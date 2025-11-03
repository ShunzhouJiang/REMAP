import random
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from . datasetgenemap import wrap_gene_location, wrap_gene_covet, wrap_gene_rel, wrap_gene_rel_test
from . DNN import DNN, DNN_rel, TrainerExe
import pickle
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Fit_cord (data_train, covet_train, location_data, out_dim = 2, hidden_dims = [400, 200, 100], num_epochs_max = 500, path = "output", filename = "remap", 
              batch_size = 128, number_error_try = 15, initial_learning_rate = 0.0001, seednum = 2025):
    """
    Fit a location predictor to learn spatial coordinates from training data.

    Parameters
    ----------
    data_train : AnnData or similar
        Training data object containing gene expression (sparse or dense).
    covet_train : array-like
        Covariates associated with training data.
    location_data : array-like
        Target spatial coordinates for each sample (num_samples * out_dim).
    out_dim : int, 2 or 3
        Output dimension of the predicted coordinates (2D or 3D).
    hidden_dims : list of int, optional (default=[400, 200, 100])
        Sizes of hidden layers in the DNN.
    num_epochs_max : int, optional (default=500)
        Maximum number of training epochs.
    path : str
        Directory path to save the trained model.
    filename : str
        Filename for the saved model (without extension).
    batch_size : int, optional 
        Batch size for the DataLoader.
    number_error_try : int, optional (default=15)
        Number of times to tolerate non-improving loss before reducing learning rate.
    initial_learning_rate : float, optional (default=0.0001)
        Initial learning rate for the optimizer.
    seednum : int, optional (default=2025)
        Random seed for reproducibility.
    """
    
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    tdata_rs = (data_train.X.A if issparse(data_train.X) else data_train.X)
    DataTra = wrap_gene_location(tdata_rs, location_data, covet_train)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    DNNmodel = DNN( in_channels = DataTra[1][0].shape[0], hidden_dims = hidden_dims, out_channels=location_data.shape[1], type_pred = "loc")
    DNNmodel = DNNmodel.float()

    CoOrg = TrainerExe(filename=filename)
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)

    os.makedirs(path, exist_ok=True)
    torch.save(DNNmodel, f"{path}/{filename}.pt")
    return DNNmodel


def Predict_cord (data_test, covet_test, out_dim = 2,  path = "output", filename = "remap"):
    """
    Predict spatial coordinates for test data using a trained location predictor.
    
    Parameters
    ----------
    data_test : AnnData 
        Testing anndata
    covet_test : array-like
        Neighboring gene-gene covariance for test data
    out_dim : int, optional (default=2)
        Dimension of spatial dimensions 
    path : str, optional (default="output")
        Folder where the trained model (pickle file) is saved.
    filename : str, optional (default="remap")
        Name of the trained model file (without extension).
    """
    
    # Create placeholder location data
    location_data = pd.DataFrame(np.ones((data_test.shape[0], out_dim)))
    vdata_rs = data_test.X.A if issparse(data_test.X) else data_test.X

    # Wrap data into dataset and dataloader
    DataVal = wrap_gene_location(vdata_rs, location_data, covet_test)
    Val_loader = torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers=0)

    DNNmodel = torch.load(f"{path}/{filename}.pt")
    DNNmodel = DNNmodel.to("cpu")
    DNNmodel.eval()  # evaluation mode

    # Predict coordinates
    coords_predict = np.zeros((data_test.obs.shape[0], out_dim))
    with torch.no_grad():
        for i, img in enumerate(Val_loader):
            # Move tensors to CPU
            img = tuple(x.to("cpu") if isinstance(x, torch.Tensor) else x for x in img)
            recon = DNNmodel(img)
            coords_predict[i, :] = recon[0].numpy()
    return coords_predict


def pred_transform(pred_cord, train_cord = None):
    """
    Transform the predicted coordinates to match the scale of the reference coordinates.
    """
    if train_cord is None:
        return pred_cord

    coord_dim = pred_cord.shape[1]
    pred_cord_transform = np.zeros_like(pred_cord)

    for i in range(coord_dim):
        coord_min = np.min(train_cord[:, i])
        coord_max = np.max(train_cord[:, i])
        pred_cord_transform[:, i] = pred_cord[:, i] * (coord_max - coord_min) + coord_min

    return pred_cord_transform


def Fit_covet (data_train, covet_train, location_data, hidden_dims = [400, 200, 100], num_epochs_max = 500, path = "output", filename = "remap", 
              batch_size = 128, number_error_try = 15, initial_learning_rate = 0.0001, seednum = 2025):
    """
    Fit a neighboring gene-gene covariance predictor to learn neighboring gene-gene covariance from expression and locations.

    Parameters
    ----------
    data_train : AnnData or similar
        Training data object containing gene expression (sparse or dense).
    covet_train : array-like
        Covariates associated with training data.
    location_data : array-like
        Target spatial coordinates for each sample (num_samples * out_dim).
    out_dim : int, 2 or 3
        Output dimension of the predicted coordinates (2D or 3D).
    hidden_dims : list of int, optional (default=[400, 200, 100])
        Sizes of hidden layers in the DNN.
    num_epochs_max : int, optional (default=500)
        Maximum number of training epochs.
    path : str
        Directory path to save the trained model.
    filename : str
        Filename for the saved model (without extension).
    batch_size : int, optional 
        Batch size for the DataLoader.
    number_error_try : int, optional (default=15)
        Number of times to tolerate non-improving loss before reducing learning rate.
    initial_learning_rate : float, optional (default=0.0001)
        Initial learning rate for the optimizer.
    seednum : int, optional (default=2025)
        Random seed for reproducibility.
    """
    
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)

    tdata_rs = (data_train.X.A if issparse(data_train.X) else data_train.X)
    DataTra = wrap_gene_covet(tdata_rs, location_data, covet_train,  trainloc=None)
    t_loader= torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = 4, shuffle = True, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    DNNmodel = DNN( in_channels = DataTra[1][0].shape[0], out_channels=DataTra[1][1].shape[0], hidden_dims = hidden_dims, type_pred = "covet") 
    DNNmodel = DNNmodel.float()

    CoOrg = TrainerExe(filename=filename)
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    os.makedirs(path, exist_ok=True)
    torch.save(DNNmodel, f"{path}/{filename}.pt")
    return DNNmodel


def Predict_covet (data_test, location_test, out_dim, location_train = None, path = "output", filename = "remap"):
    """
    Predict neighboring gene-gene covariance for test data.

    Parameters
    ----------
    data_test : AnnData
        Test dataset containing gene expression (sparse or dense) and sample metadata.
    location_test : array-like
        Spatial coordinates of the test samples.
    out_dim : int
        Dimension of the predicted covariance features.
    location_train : array-like, optional (default=None)
        Training locations used during model training (for reference in dataset wrapper).
    path : str, optional (default="output")
        Folder where the trained model (pickle file) is saved.
    filename : str, optional (default="remap")
        Name of the trained model file (without extension).

    """

    covet_data = np.ones((data_test.shape[0], out_dim))
    tdata_rs = data_test.X.A if issparse(data_test.X) else data_test.X
    DataVal = wrap_gene_covet(tdata_rs, location_test, covet_data, trainloc=location_train)
    Val_loader = torch.utils.data.DataLoader(DataVal, batch_size=1, num_workers=4)

    DNNmodel = torch.load(f"{path}/{filename}.pt")
    DNNmodel = DNNmodel.to("cpu")
    DNNmodel.eval()

    covet_predict = np.zeros((data_test.obs.shape[0], out_dim))
    with torch.no_grad():
        for i, img in enumerate(Val_loader):
            img = tuple(x.to("cpu") if isinstance(x, torch.Tensor) else x for x in img)
            recon = DNNmodel(img)
            covet_predict[i, :] = recon[0].numpy()
    return covet_predict


def Fit_relative(data_train, covet_train, location_data, weights = None, out_dim = 1, hidden_dims = [400, 200, 100], num_epochs_max = 500, path = "output", filename = "remap_rel", 
              batch_size = 1024, sample_pairs = 100000, number_error_try = 15, initial_learning_rate = 0.0001, num_workers = 8, seednum = 2025):
    """
    Train a relative DNN model to predict pairwise distances between cells based on gene expression and gene-gene covariance.

    Parameters
    ----------
    data_train : AnnData 
        Training data containing gene expression (sparse or dense).
    covet_train : array-like
        Covariates associated with training data.
    location_data : array-like
        Spatial coordinates or reference values for the training samples.
    weights : array-like, optional (default=None)
        Weights for each sample used in the loss function.
    out_dim : int, optional (default=1)
        Dimension of the output (usually 1 for relative distance/pairwise score).
    hidden_dims : list of int, optional (default=[400, 200, 100])
        Sizes of hidden layers in the DNN_rel model.
    num_epochs_max : int, optional (default=500)
        Maximum number of training epochs.
    path : str, optional (default="output")
        Directory path where the trained model will be saved.
    filename : str, optional (default="remap")
        Filename for the saved model (without extension).
    batch_size : int, optional (default=1024)
        Batch size for the training.
    sample_pairs : int, optional (default=100000)
        Number of cell pairs sampled for each slice for training.
    number_error_try : int, optional (default=15)
        Number of times to tolerate non-improving loss before reducing learning rate.
    initial_learning_rate : float, optional (default=0.0001)
        Initial learning rate for the optimizer.
    seednum : int, optional (default=2025)
        Random seed for reproducibility.
    """
    
    random.seed(seednum)
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    g = torch.Generator()
    g.manual_seed(seednum)
    DataTra = wrap_gene_rel(data_train, location_data, covet_train, weights, sample_pairs=sample_pairs)
    # DataTra = wrap_gene_rel(data_train, location_data, covet_train, weights)
    
    t_loader = torch.utils.data.DataLoader(DataTra, batch_size=batch_size, num_workers = num_workers, shuffle = True, worker_init_fn=seed_worker)
    
    DNNmodel = DNN_rel(in_channels=DataTra[0][0].shape[0], out_channels=out_dim, hidden_dims=hidden_dims)
    DNNmodel = DNNmodel.float()
    CoOrg = TrainerExe(filename=filename)
    CoOrg.train(model = DNNmodel, train_loader = t_loader, num_epochs= num_epochs_max, RCcountMax = number_error_try, learning_rate = initial_learning_rate)
    os.makedirs(path, exist_ok=True)
    torch.save(DNNmodel, f"{path}/{filename}.pt")
    return DNNmodel


def Pred_relative(data_test, covet_test, batch_size = 8192, full_pairwise = False, neighbor_fraction=0.1, path = "output", filename = "remap_rel", num_workers = 8):
    """
    Predict pairwise distances for testing cells.

    Parameters
    ----------
    data_test : AnnData or array-like
        Test dataset containing gene expression (sparse or dense) and sample metadata.
    covet_test : array-like
        Covariates associated with the test dataset.
    batch_size : int, optional (default=8192)
        Batch size for the DataLoader during prediction.
    full_pairwise : logical, defauld False
        Whether to predict the full pairwise distance matrix for every cell pair, unnecessary for CN clustering
    neighbor_fraction : float
        If not predicting full pairwise distance matrix, we first filter neighbors based on feature neighbors. 
        Fraction (0-1) of feature neighbors to keep per cell
    path : str
        Folder where the trained model (pickle file) is saved.
    filename : str
        Name of the trained model file (without extension).
    """
    
    loc_test = np.zeros((data_test.shape[0], 2), dtype=np.float32)
    data_test = wrap_gene_rel_test(data_test, loc_test, covet_test, full_pairwise = full_pairwise, neighbor_fraction=neighbor_fraction)
    DNNmodel = torch.load(f"{path}/{filename}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DNNmodel = DNNmodel.to(device)
    DNNmodel.eval()

    loader= torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory = True)
    result = np.zeros((data_test.index_lst.shape[0], 3), dtype=np.float32)
    result[:, :2] = data_test.index_lst

    for i, img in enumerate(tqdm(loader)):
        img = tuple(x.to("cuda") if isinstance(x, torch.Tensor) else x for x in img)
        with torch.no_grad():
            recon = DNNmodel(img)
        recon = recon[0].cpu().numpy()
        start_idx = i * batch_size
        end_idx = start_idx + len(recon)
        result[start_idx:end_idx, 2] = recon.flatten()
    
    size = int(np.max(result[:, 0])) + 1
    pair_dist = np.zeros((size, size), dtype=np.float32)
    pair_dist[result[:, 0].astype(int), result[:, 1].astype(int)] = result[:, 2]
    # pair_dist = pair_dist + pair_dist.T - np.diag(np.diag(pair_dist))
    if full_pairwise is False:
        pair_dist[pair_dist == 0] = np.inf
    # pair_dist[np.eye(pair_dist.shape[0], dtype=bool)] = 0
    return pair_dist