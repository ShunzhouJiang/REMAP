from .ENVI import ENVI
from .fit_functions import *
from .util import *
from .sc_st_similarity import st_frac
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from anndata import concat
import numpy as np
import os
import scanpy as sc
import torch
import gc
from scipy.sparse import csr_matrix, save_npz, load_npz, issparse

def covet_init(st_data, sc_data, save_path, num_covet_genes = 100, k_nearest = 100, num_HVG = 1000, 
               covet_genes = [], stable = 1e-6, batch_key = None, epochs = 10000,
               spatial_distribution = 'pois', sc_distribution = 'nb', log_input = 0.1, lib_size = False,
               covet_batch_size = 256, train_batch_size = 2048, pve = 0.98, sample_gene = None):
    """
    Initialize neighboring gene-gene covariance estimation. For ST, they are directly computed. 
    For scRNA-seq, they are inferred using the ENVI model (Haviv et al. Nat Biotechnol (2025)).
    
    For multi-capture training, please specify `batch_key` in combined ST data to indicate capture source of cells.
    
    Parameters
    ----------
    st_data : AnnData
        Spatial transcriptomics dataset.
    sc_data : AnnData
        Single-cell RNA-seq dataset.
    save_path : str
        Directory path to save output files.
    num_covet_genes : int, default=100
        Number of genes to calculate neighboring gene-gene covariance.
    k_nearest : int, default=100
        Number of nearest neighbors for calculating neighboring gene-gene covariance.
    num_HVG : int, default=1000
        Number of highly variable genes to select for scRNA-seq.
    covet_genes : list, default=[]
        Predefined list of COVET genes (if provided).
    stable : float, default=1e-6
        Stability constant.
    batch_key : str, optional
        Key for different slices in ST data (For multi-capture training).
    epochs : int, default=10000
        Number of training epochs.
    spatial_distribution : {'pois', 'nb'}, default='pois'
        Distribution assumption for ST data (Poisson or Negative Binomial).
    sc_distribution : {'pois', 'nb'}, default='nb'
        Distribution assumption for scRNA-seq data.
    log_input : float, default=0.1
        Log-transformation scaling factor.
    lib_size : bool, default=False
        Whether to normalize by library size.
    covet_batch_size : int, optional
        Batch size for calculating neighboring gene-gene covariance.
    train_batch_size : int, default=2048
        Batch size for ENVI training.
    pve : float or bool, default=0.98
        Proportion of variance explained for PCA dimensionality reduction.
        If False, upper-triangle representation of COVET is used.
    sample_gene : int, optional
        If provided, subsample this many HVGs from ST data.
    
    Returns
    -------
    st_data_raw : AnnData
        Original ST dataset with inferred COVET embeddings stored in `.obsm['covet']`.
    sc_data_raw : AnnData
        Original scRNA dataset with inferred COVET embeddings stored in `.obsm['covet']`.
    """
    
    os.makedirs(save_path, exist_ok=True)
    st_data_raw = st_data.copy(); sc_data_raw = sc_data.copy()
    st_data_raw.X = st_data_raw.X.A.astype(np.float32) if issparse(st_data_raw.X) else st_data_raw.X.astype(np.float32)
    sc_data_raw.X = sc_data_raw.X.A.astype(np.float32) if issparse(sc_data_raw.X) else sc_data_raw.X.astype(np.float32)

    st_cov_path = f"{save_path}/st_covariance.npy"
    sc_cov_path = f"{save_path}/sc_covariance.npy"

    if 'covet' in st_data.obsm.keys() and 'covet' in sc_data.obsm.keys():
        print("Loading precomputed neighboring gene-gene covariance")
        covet_train_pca = st_data.obsm['covet']
        covet_test_pca = sc_data.obsm['covet'] if 'covet' in sc_data.obsm.keys() else None
        return st_data_raw, sc_data_raw
    if os.path.exists(st_cov_path) and os.path.exists(sc_cov_path):
        print("Loading precomputed neighboring gene-gene covariance")
        covet_train_pca = np.load(st_cov_path)
        covet_test_pca = np.load(sc_cov_path)
    else:
        if st_data_raw.shape[0] > 200000:
            if sample_gene is not None:
                sample_gene = min(300, sample_gene)
            else:
                sample_gene = 300
        if sample_gene is not None:
            st_data.layers['log'] = np.log(st_data.X + 1)
            sc.pp.highly_variable_genes(st_data, n_top_genes = sample_gene, layer = 'log')
            st_data = st_data[:, st_data.var['highly_variable']].copy()
        envi_model = ENVI(spatial_data = st_data, sc_data = sc_data, num_covet_genes=num_covet_genes, covet_genes = covet_genes, 
                            k_nearest=k_nearest, spatial_distribution = spatial_distribution, sc_distribution=sc_distribution, batch_key = batch_key,
                        num_HVG=num_HVG, covet_batch_size = covet_batch_size, log_input=log_input, lib_size = lib_size, stable = stable)
        envi_model.train(batch_size = train_batch_size, epochs = epochs)
        envi_model.infer_covet()
        del st_data, sc_data
        torch.cuda.empty_cache()
        
        # Dimensionality reduction: PCA if pve given, else flatten covariance matrix upper triangle part
        if pve:
            covet_train_pca, pca_model = covet_pca(envi_model.spatial_data.obsm['COVET'], pve = pve, flat=True)
            covet_test_pca = covet_pcafit(envi_model.sc_data.obsm['COVET'], pca_model, flat=True)
            np.save(st_cov_path, covet_train_pca)
            np.save(sc_cov_path, covet_test_pca)
            print("Saved neighboring gene-gene covariance PCA embeddings.")
            del pca_model
        else:
            covet_train_pca = np.array(list(map(covet_upper, envi_model.spatial_data.obsm['COVET'])))
            covet_test_pca = np.array(list(map(covet_upper, envi_model.sc_data.obsm['COVET'])))
        del envi_model
    st_data_raw.obsm['covet'] = covet_train_pca
    sc_data_raw.obsm['covet'] = covet_test_pca
    del covet_train_pca, covet_test_pca
    gc.collect()
    torch.cuda.empty_cache()
    return st_data_raw, sc_data_raw


def Fit_cord_single(Rdata, location_data, Qdata, path_name, hidden_dims = [400, 200, 100], n_iter = 3, batch_size = 256, harmony = True):
    """
    Iteratively update location estimates and neighboring gene-gene covariance estimates.
    If no batch effect correction is applied before, harmony = True is recommended.

    Parameters
    ----------
    Rdata : AnnData
        Reference dataset (with known spatial coordinates).
    location_data : pd.DataFrame or np.ndarray
        True spatial coordinates for Rdata.
    Qdata : AnnData
        Query dataset (to be aligned into reference coordinate space).
    path_name : str
        Path to save model and predictions.
    hidden_dims : list, default=[400,200,100]
        Hidden layer sizes for the neural network.
    n_iter : int, default=3
        Number of  iterations.
    harmony : bool, default=True
        Whether to run Harmony for batch effect correction between ST and scRNA-seq gene expression.
    """
    # Whether to run Harmony for batch effect correction
    
    cuda = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"Model is on {cuda}")
    
    norm = True
    if harmony:
        files = os.listdir(path_name)
        if "st_harmony.npy" in files and "sc_harmony.npy" in files:
            print("Loading Harmony corrected expression")
            Rdata.obsm["corrected"] = np.load(f"{path_name}/st_harmony.npy")
            Qdata.obsm["corrected"] = np.load(f"{path_name}/sc_harmony.npy")
        else:
            Rdata, Qdata = run_harmony(Rdata, Qdata, norm = True, st_all = True)
            np.save(f"{path_name}/st_harmony.npy", Rdata.obsm["corrected"])
            np.save(f"{path_name}/sc_harmony.npy", Qdata.obsm["corrected"])
        norm = False
    
    # Use corrected expression values for both datasets
    Rdata = use_corrected_expression(Rdata)
    Qdata = use_corrected_expression(Qdata)
    
    # Initialized neighboring gene-gene covariance embeddings (from obsm)
    covet_train_pca = Rdata.obsm['covet']
    covet_test_pca = Qdata.obsm['covet']
    if norm:
        Rdata, Qdata = preprocess(Rdata, Qdata, normalize = True, scale = True)
    location_data = np.array(location_data)
        
    # Iterative refinement loop
    for i in range(n_iter):
        # Step 1: Fit model to predict coordinates from gene expression + neighboring gene-gene covariance
        file_name_loc = f"location_iter{i}"
        model_train = Fit_cord (data_train = Rdata, location_data=location_data, covet_train=covet_train_pca, hidden_dims = hidden_dims, num_epochs_max = 500, number_error_try=15, 
                                    batch_size = batch_size, path = path_name, filename = file_name_loc)
        pred_loc_test = Predict_cord (data_test = Qdata, covet_test=covet_test_pca, out_dim = location_data.shape[1], path = path_name, filename = file_name_loc)
        pred_loc_train = Predict_cord (data_test = Rdata, covet_test=covet_train_pca, out_dim = location_data.shape[1], path = path_name, filename = file_name_loc)
            
        pred_loc_test_transform = pred_transform(pred_cord = pred_loc_test, train_cord = np.array(location_data))
        pred_loc_train_transform = pred_transform(pred_cord = pred_loc_train, train_cord = np.array(location_data))

        # Step 2: Fit model to update neighboring gene-gene covariance estimates from gene expression + coordinates
        if i != n_iter - 1:
            file_name_covet = f"covariance_iter{i}"
            model_train = Fit_covet (data_train = Rdata, location_data=pred_loc_train_transform, covet_train=covet_train_pca, hidden_dims = hidden_dims, num_epochs_max = 500, number_error_try=15, 
                                    batch_size = batch_size, path = path_name, filename = file_name_covet)
            pred_covet_test = Predict_covet (data_test = Qdata, location_test = pred_loc_test_transform, out_dim=covet_train_pca.shape[1], location_train = pred_loc_train_transform,
                                        path = path_name, filename = file_name_covet)
            pred_covet_train = Predict_covet (data_test = Rdata, location_test = pred_loc_train_transform, out_dim=covet_train_pca.shape[1],
                                        path = path_name, filename = file_name_covet)
            
            covet_test_pca = pred_covet_test
            covet_train_pca = pred_covet_train
    np.save(f"{path_name}/remap_loc.npy", pred_loc_test_transform)
    return pred_loc_test_transform
            
            
def Fit_cord_multi(Rdata, location_data, Qdata, path_name, source_key = "source", equal_size = False, grid_size = 50, batch_train = 1024, batch_test = 8192,
                   sample_pairs = 100000, full_pairwise = False, neighbor_fraction=0.1, harmony = True, 
                 hidden_dims = [400, 100, 100], save = True, train = True, num_workers = 8):
    """
    Train pairwise distance prediction model using multi-capture ST data and predict pairwise distances for scRNA-seq data.

    This function optionally runs Harmony batch correction, trains a model on ST data, and predicts pairwise distances for scRNA-seq.
    If no batch effect correction is applied before, harmony = True is recommended.

    Parameters
    ----------
    Rdata : AnnData or list of AnnData
        Reference dataset(s) (e.g., spatial transcriptomics).
    location_data : np.ndarray or list of np.ndarray
        Spatial coordinates corresponding to Rdata cells.
    Qdata : AnnData
        Query dataset (e.g., scRNA-seq) for which pairwise distances are predicted.
    path_name : str
        Path to save model outputs and intermediate files.
    source_key : str, optional
        Column in `Rdata.obs` indicating source/batch labels. Default is "source".
    equal_size : bool, optional
        Whether each ST capture is equal-sized. If not, we will rescale ST captures by their location ranges to match sizes.
    grid_size : int or float, optional
        Grid size for spatial sampling in each reference ST capture. Default is 50.
    batch_train : int, optional
        Batch size for model training. Default is 1024.
    batch_test : int, optional
        Batch size for model testing. Default is 8192.
    sample_pairs : int, optional
        Number of cell pairs sampled for each slice for training. Default is 100000.
    full_pairwise : logical, defauld False
        Whether to predict the full pairwise distance matrix for every cell pair, unnecessary for CN clustering
    neighbor_fraction : float
        If not predicting full pairwise distance matrix, we first filter neighbors based on feature neighbors. 
        Fraction (0-1) of feature neighbors to keep per cell, default is 0.1.
    hidden_dims : list of int, optional
        Hidden layer sizes for the model. Default is [400, 100, 100].
    sample_pairs : int, optional (default=100000)
        Number of cell pairs sampled for each slice for training.
    harmony : bool, optional
        If True, perform Harmony batch correction. Default is False.
    num_workers : int, optional
        Number of workers for parallel processing. Default is 8.
    save : bool, optional
        If True, saves the predicted distances to a .npy file. Default is True.

    Returns
    -------
    dist_pred : np.ndarray
        Predicted pairwise distances for the query dataset Qdata.
    """
    
    cuda = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"Model is on {cuda}")
    
    os.makedirs(path_name, exist_ok=True)
    Rdata.obsm['spatial'] = np.array(location_data)

    norm = True
    if harmony:
        Rdata, Qdata = run_harmony(Rdata, Qdata, norm = True, st_all = False)
        np.save(f"{path_name}/st_harmony.npy", Rdata.obsm["corrected"])
        np.save(f"{path_name}/sc_harmony.npy", Qdata.obsm["corrected"])
        norm = False
        
    Rdata = use_corrected_expression(Rdata)
    Qdata = use_corrected_expression(Qdata)
        
    if source_key in Rdata.obs.columns:
        source_uniq = np.sort(np.unique(Rdata.obs[source_key]))
        Rdata = [Rdata[Rdata.obs[source_key] == source_uniq[j]] for j in range(len(source_uniq))]
        Qdata.obs[source_key] = 'test'

        if len(source_uniq) > 1:
            adata_concat = concat(Rdata + [Qdata], axis = 0)
            weights_mean, _ = st_frac(adata_concat, test_name="test", path=path_name, sample_num=3000, source_key=source_key, norm = norm)
            weights_mean = weights_mean[source_uniq] * len(weights_mean)
            weights_mean = np.array(weights_mean)
            del adata_concat
        else:
            weights = None; weights_mean = None
    else:
        Rdata = [Rdata]
        location_data = [location_data]
        weights = None; weights_mean = None
    
    if equal_size:
        for i in range(len(Rdata)):
            Rdata[i] = norm_cord(Rdata[i], spatial_key='spatial')
    else:
        Rdata = norm_cord_scale(Rdata, spatial_key='spatial')
    
    if isinstance(grid_size, (int, float)):
        grid_size = int(grid_size)
        for i in range(len(Rdata)):
            Rdata[i] = select_sample(Rdata[i], grid_size=grid_size)

    if norm:
        Rdata, Qdata = preprocess(Rdata, Qdata, normalize = True, scale = True)
    location_data = [Rdata[j].obsm['spatial'] for j in range(len(Rdata))]
    covet_train = [Rdata[j].obsm['covet'] for j in range(len(Rdata))]
    
    if (not os.path.exists(f"{path_name}/remap_rel.pt")) or train:
        model_train = Fit_relative(Rdata, location_data = location_data, weights=weights_mean, covet_train=covet_train, hidden_dims = hidden_dims, path = path_name, 
                                   sample_pairs = sample_pairs, 
                                    filename = "remap_rel", batch_size = batch_train, num_epochs_max = 500, num_workers = num_workers)
    print("Predicting pairwise distances for scRNA-seq data.")
    dist_pred = Pred_relative(Qdata, covet_test = Qdata.obsm['covet'], batch_size=batch_test, full_pairwise = full_pairwise, 
                              neighbor_fraction=neighbor_fraction, path = path_name, filename = "remap_rel")
    if save:
        finite_mask = np.isfinite(dist_pred)
        i, j = np.where(finite_mask)
        sparse_matrix = csr_matrix((dist_pred[finite_mask], (i, j)), shape=dist_pred.shape)
        save_npz(f"{path_name}/remap_rel_dist.npz", sparse_matrix)
    return dist_pred
    

def Prefict_cord_rel(Rdata, Qdata, path_name, batch_size = 8192, full_pairwise = False, neighbor_fraction=0.1, save = True, filename = "remap_rel_dist"):
    """
    After training the model, predict pairwise distances for a query scRNA-seq data relative to reference ST data,

    Parameters
    ----------
    Rdata : AnnData
        Reference dataset (e.g., spatial transcriptomics).
    Qdata : AnnData
        Query dataset (e.g., scRNA-seq) for which pairwise distances are predicted.
    path_name : str
        Path to save the predicted distances.
    batch_size : int, optional
        Batch size for distance prediction. Default is 8192.
    full_pairwise : logical, defauld False
        Whether to predict the full pairwise distance matrix for every cell pair, unnecessary for CN clustering
    neighbor_fraction : float
        If not predicting full pairwise distance matrix, we first filter neighbors based on feature neighbors. 
        Fraction (0-1) of feature neighbors to keep per cell, default is 0.1.
    save : bool, optional
        If True, saves the predicted distances to a .npy file. Default is True.
    filename : str, optional
        Name of the saved file (without extension). Default is "remap_rel_dist".

    Returns
    -------
    dist_pred : np.ndarray
        Predicted full pairwise distance matrix (symmetric).
    """
    Rdata = use_corrected_expression(Rdata)
    Qdata = use_corrected_expression(Qdata)
    Rdata, Qdata = preprocess(Rdata, Qdata, normalize = True, scale = True)
    dist_pred = Pred_relative(Qdata, covet_test = Qdata.obsm['covet'], batch_size=batch_size, full_pairwise = full_pairwise, 
                              neighbor_fraction=neighbor_fraction, path = path_name, filename = "remap_rel")
    if save:
        finite_mask = np.isfinite(dist_pred)
        i, j = np.where(finite_mask)
        values = dist_pred[finite_mask]
        sparse_matrix = csr_matrix((values, (i, j)), shape=dist_pred.shape)
        save_npz(f"{path_name}/{filename}.npz", sparse_matrix)
    return dist_pred