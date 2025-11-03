import pandas as pd
import numpy as np
import scipy
import os
import scanpy as sc
from tqdm import tqdm
import seaborn as sns
from scipy.sparse import issparse
from anndata import AnnData
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mstats
from sklearn.metrics import pairwise_distances
from anndata import AnnData, concat
from sklearn.manifold import MDS
from scipy.spatial import procrustes
import harmonypy as hm


def get_zscore (adata, mean = None, sd = None ):
    """
    Standardize gene expression data (Z-score) for an AnnData object.

    This function computes Z-scores for each gene (column-wise) and caps
    extreme values at 6. If `mean` and `sd` are provided, they are used
    instead of computing from the data. The function stores mean, standard
    deviation, and a flag in `adata`.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing gene expression in `adata.X`.
    mean : np.ndarray, optional
        Precomputed mean per gene (default is None, compute from data).
    sd : np.ndarray, optional
        Precomputed standard deviation per gene (default is None, compute from data).
    """
    genotypedata = (adata.X.A if issparse(adata.X) else adata.X)
    if mean is None:
        genemean = np.mean(genotypedata, axis =0)
        genesd = np.std(genotypedata, axis = 0)
    else:
        genemean = mean
        genesd = sd
    try:
        if adata.standardize is not True:
                datatransform = (genotypedata - genemean) / genesd
                adata.X = datatransform
                adata.genemean = genemean
                adata.genesd = genesd
                adata.standardize = True
                adata.X[np.where(adata.X > 6)] = 6
        else:
            print("Data has already been z-scored")
    except AttributeError:
        datatransform = (genotypedata - genemean) / genesd
        adata.X = datatransform
        adata.genemean = genemean
        adata.genesd = genesd
        adata.standardize = True
        adata.X[np.where(adata.X > 6)] = 6


def lib_norm (adata):
    """
    Perform library size normalization and log-transform on an AnnData object.

    This function normalizes each cell to the same total counts, log-transforms the data,
    and ensures unique and uppercase gene names.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression data in `adata.X` and gene names in `adata.var`.

    Returns
    -------
    AnnData
        The AnnData object with normalized and log-transformed gene expression.
        The changes are also reflected in `adata.var` (unique gene names).
    """
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    adata.var_names_make_unique()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
    sc.pp.log1p(adata)
    return adata

def preprocess(st_data, sc_data, normalize = True, scale = True):
    """
    Preprocess ST and scRNA-seq data.

    This function ensures that both datasets share common genes, optionally normalizes
    library size, and optionally scales (Z-score) the data. Supports `st_data` as
    a single AnnData object or a list of AnnData objects.

    Parameters
    ----------
    st_data : AnnData or list of AnnData
        Spatial transcriptomics dataset(s). Can be a single AnnData or a list of AnnData objects.
    sc_data : AnnData
        Single-cell RNA-seq dataset.
    normalize : bool, optional
        If True, perform library size normalization. Default is True.
    scale : bool, optional
        If True, perform Z-score scaling. Default is True.

    Returns
    -------
    st_data_return : AnnData or list of AnnData
        Preprocessed spatial transcriptomics dataset(s), matching the input type.
    sc_data : AnnData
        Preprocessed single-cell dataset.
    """
    if isinstance(st_data, list):
        index_lst = [0]
        common_genes = set(st_data[0].var.index).intersection(*[set(data.var.index) for data in st_data[1:]])
        common_genes = list(common_genes)
        for i in range(len(st_data)):
            st_data[i] = st_data[i][:, common_genes]
            index_lst.append(st_data[i].shape[0] + index_lst[len(index_lst)-1])
        st_data_lst = concat(st_data, axis = 0)
    else:
        st_data_lst = st_data
    
    common_genes = np.intersect1d(st_data_lst.var.index, sc_data.var.index)
    st_data_lst = st_data_lst[:, common_genes]
    sc_data = sc_data[:, common_genes]
    if normalize:
        lib_norm(st_data_lst)
        lib_norm(sc_data)
    if scale:
        get_zscore(st_data_lst)
        # get_zscore(sc_data, mean=st_data_lst.genemean, sd=st_data_lst.genesd)
        get_zscore(sc_data)
    
    if isinstance(st_data, list):
        st_data_return = []
        for i in range(len(st_data)):
            st_data_return.append(st_data_lst[index_lst[i]:index_lst[i+1], :])
    else:
        st_data_return = st_data_lst
    return st_data_return, sc_data

def covet_upper(covet):
    return covet[np.triu_indices(covet.shape[0])]

def covet_pca(covet_train, pve = 0.98, flat = True):
    """
    Perform PCA (Principal Component Analysis) on a covariate matrix.

    Optionally flattens upper-triangular covariate matrices before PCA,
    and retains components explaining a given proportion of variance.

    Parameters
    ----------
    covet_train : array-like
        Training covariate data. Each entry can be a matrix or vector.
    pve : float, optional
        Proportion of variance explained to retain. Default is 0.98 (retain 98% variance).
    flat : bool, optional
        If True, applies `covet_upper` to flatten upper-triangular matrices into vectors.
        Default is True.

    Returns
    -------
    covet_train : np.ndarray
        PCA-transformed covariate matrix.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model (can be used to transform new data).
    """
    if flat == True:
        covet_train = np.array(list(map(covet_upper, covet_train)))
    pca_model = PCA(n_components=pve, random_state=42)
    covet_train = pca_model.fit_transform(covet_train)
    return covet_train, pca_model


def covet_pcafit(covet_test, pca_model, flat=True):
    """
    Transform covariate data using a pre-fitted PCA model.

    Optionally flattens upper-triangular covariate matrices before applying the PCA transformation.

    Parameters
    ----------
    covet_test : array-like
        Covariate data to transform. Each entry can be a matrix or vector.
    pca_model : sklearn.decomposition.PCA
        Pre-fitted PCA model returned by `covet_pca`.
    flat : bool, optional
        If True, applies `covet_upper` to flatten upper-triangular matrices into vectors.
        Default is True.

    Returns
    -------
    covet_test : np.ndarray
        PCA-transformed covariate data.
    """
    if flat == True:
        covet_test = np.array(list(map(covet_upper, covet_test)))
    covet_test = pca_model.transform(covet_test)
    return covet_test


def use_corrected_expression(adata, obsm_key="corrected", var_names_key="corrected_var_names"):
    """
    Return a new AnnData object using corrected expression stored in obsm,
    while keeping obs and uns from the original AnnData.
    """
    if obsm_key not in adata.obsm:
        print("No batch corrected expression found, use raw expression adata.")
        return adata

    corrected_expr = adata.obsm[obsm_key]
    if var_names_key in adata.uns:
        corrected_var_names = adata.uns[var_names_key]
    else:
        corrected_var_names = range(adata.obsm[obsm_key].shape[1])

    if corrected_expr.shape[0] != adata.n_obs:
        raise ValueError("Number of cells in corrected expression does not match adata.")

    new_adata = AnnData(
        X=corrected_expr,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[str(v) for v in corrected_var_names]),
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
    )
    return new_adata

def merge_adata(Rdata):
    """
    Merge a list of AnnData objects into a single AnnData object
    """
    # 'keys' will add a new column in .obs named 'batch' by default
    merged = concat(Rdata, axis=0, merge="same", keys=[f"s{i}" for i in range(len(Rdata))])
    merged.uns = Rdata[0].uns.copy()
    return merged

def MDSTransform(pred_dist):
    """
    Perform Multidimensional Scaling (MDS) on a distance matrix.

    Parameters
    ----------
    pred_dist : np.ndarray
        A square (n x n) distance matrix representing pairwise dissimilarities 
        between samples.

    Returns
    -------
    embedding : np.ndarray
        A 2D embedding (n x 2) of the input distances in Euclidean space.
    """
    
    if np.allclose(pred_dist, pred_dist.T) is False:
        pred_dist = pred_dist + pred_dist.T - np.diag(np.diag(pred_dist))
    pred_dist[np.eye(pred_dist.shape[0], dtype=bool)] = 0
    # pred_pcoa = skbio.stats.ordination.pcoa(pred_dist).samples
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(pred_dist)
    # return pred_pcoa
    return embedding

def MDSalign(loc_true, Qdata_loc):
    """
    Align MDS embedding to true spatial coordinates using Procrustes analysis (Only for visualization).

    Parameters
    ----------
    loc_true : np.ndarray
        Ground-truth 2D coordinates (n x 2).
    Qdata_loc : np.ndarray
        MDS embedding (n x 2) to be aligned.

    Returns
    -------
    mtx2 : np.ndarray
        The aligned version of Qdata_loc (scaled, rotated, translated).
    """
    loc_true = np.array(loc_true)
    Qdata_loc = np.array(Qdata_loc)
    loc_true = (loc_true - np.min(loc_true, axis=0)) / (np.max(loc_true, axis=0) - np.min(loc_true, axis=0))
    mds_norm = (Qdata_loc - np.min(Qdata_loc, axis=0)) / (np.max(Qdata_loc, axis=0) - np.min(Qdata_loc, axis=0))
    mtx1, mtx2, disparity = procrustes(loc_true, mds_norm)
    return mtx2
    

def select_sample(adata, grid_size=50, spatial_key='spatial'):
    """
    Select one cell per grid by maximum UMI count.
    Coordinates are taken from adata.obsm[spatial_key].
    """
    # Compute UMI counts
    if issparse(adata.X):
        adata.obs['UMI'] = adata.X.sum(axis=1).A1 
    else:
        adata.obs['UMI'] = adata.X.sum(axis=1)

    coords = adata.obsm[spatial_key].copy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size

    x_grid = ((coords[:, 0] - x_min) // x_step).astype(int).astype(str)
    y_grid = ((coords[:, 1] - y_min) // y_step).astype(int).astype(str)
    
    adata.obs["xbin"] = x_grid; adata.obs["ybin"] = y_grid
    adata.obs["bin"] = adata.obs["xbin"] + "_" + adata.obs["ybin"]

    # Keep the cell with max UMI per bin
    idx = adata.obs.groupby('bin')['UMI'].idxmax()
    adata = adata[idx].copy()
    return adata
    
    
def norm_cord(Qdata, spatial_key='spatial'):
    """
    Normalize coordinates stored in obsm[spatial_key] for a single AnnData object.
    """
    Qdata_loc = Qdata.obsm[spatial_key].copy()
    Qdata_loc = (Qdata_loc - Qdata_loc.min(axis=0)) / (Qdata_loc.max(axis=0) - Qdata_loc.min(axis=0))
    Qdata = Qdata.copy()
    Qdata.obsm[spatial_key] = Qdata_loc
    return Qdata


def norm_cord_scale(Rdata, spatial_key='spatial'):
    """
    Normalize coordinates for a list of AnnData objects in Rdata, scale them relative to
    the maximum x/y ranges across all datasets, using obsm[spatial_key].
    """
    x_ranges = [adata.obsm[spatial_key][:, 0].max() - adata.obsm[spatial_key][:, 0].min() for adata in Rdata]
    y_ranges = [adata.obsm[spatial_key][:, 1].max() - adata.obsm[spatial_key][:, 1].min() for adata in Rdata]
    
    x_range_max = np.max(x_ranges)
    y_range_max = np.max(y_ranges)
    
    Rdata_lst = []
    for adata, x_range, y_range in zip(Rdata, x_ranges, y_ranges):
        Qdata_loc = adata.obsm[spatial_key].copy()
        Qdata_loc = (Qdata_loc - Qdata_loc.min(axis=0)) / (Qdata_loc.max(axis=0) - Qdata_loc.min(axis=0))
        Qdata_loc[:, 0] *= x_range / x_range_max
        Qdata_loc[:, 1] *= y_range / y_range_max
        adata.obsm[spatial_key] = Qdata_loc
        Rdata_lst.append(adata)
    return Rdata_lst

def load_dist(path_name, filename):
    """
    Load a pairwise distance matrix saved as upper-triangular entries and reconstruct the full matrix.

    The distance matrix is assumed to be symmetric, and only the upper-triangular elements
    (excluding the diagonal) were saved to save memory.

    Parameters
    ----------
    path_name : str
        Path to the folder containing the saved .npy file.
    filename : str
        Name of the .npy file (without extension) storing the upper-triangular distances.

    Returns
    -------
    dist_full : np.ndarray
        Reconstructed full symmetric pairwise distance matrix of shape (n, n).
    """
    dist_upper = np.load(f"{path_name}/{filename}.npy").astype(np.float32)
    n = int((1 + np.sqrt(1 + 8*len(dist_upper))) / 2)  # number of rows/cols
    dist_full = np.zeros((n, n), dtype=np.float32)
    triu_indices = np.triu_indices(n, k=1)
    dist_full[triu_indices] = dist_upper
    dist_full = dist_full + dist_full.T
    return dist_full


def run_harmony(Rdata_raw, Qdata_raw, norm = True, st_all = True):
    """
    Integrate two datasets (e.g., spatial transcriptomics and single-cell RNA-seq)
    using Harmony batch correction.

    This function optionally normalizes the data, computes PCA, and runs Harmony
    to correct for batch effects between `Rdata_raw` and `Qdata_raw`.

    Parameters
    ----------
    Rdata_raw : AnnData
        Reference dataset (e.g., spatial transcriptomics).
    Qdata_raw : AnnData
        Query dataset (e.g., single-cell RNA-seq).
    norm : bool, optional
        If True, preprocess both datasets with normalization and scaling. Default is True.
    st_all : bool, optional
        If True, all cells in Rdata are labeled as 'st' and in Qdata as 'sc'.
        Otherwise, uses existing 'source' column if present. Default is False.
    """
    if norm:
        Rdata_raw, Qdata_raw = preprocess(Rdata_raw, Qdata_raw, normalize=True, scale=True)
    Rdata = Rdata_raw.copy(); Qdata = Qdata_raw.copy()
    if st_all or 'source' not in Rdata.obs.columns:
        Rdata.obs['source_harmony'] = 'st'; Qdata.obs['source_harmony'] = 'sc'
    else:
        Rdata.obs['source_harmony'] = Rdata.obs['source']; Qdata.obs['source_harmony'] = Qdata.obs['source']
    
    adata_combined = concat([Rdata, Qdata], axis=0)

    adata_combined.obsm['X_pca'] = PCA(n_components=0.98, random_state=42).fit_transform(adata_combined.X)
    print(adata_combined.obsm['X_pca'].shape)
    ho = hm.run_harmony(adata_combined.obsm['X_pca'], adata_combined.obs, 'source_harmony')
    adata_combined.obsm["corrected"] = ho.Z_corr.T

    Rdata_raw.obsm["corrected"] = adata_combined[adata_combined.obs['source_harmony'] != 'sc'].obsm["corrected"]
    Qdata_raw.obsm["corrected"] = adata_combined[adata_combined.obs['source_harmony'] == 'sc'].obsm["corrected"]
    return Rdata_raw, Qdata_raw