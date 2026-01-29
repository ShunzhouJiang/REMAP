import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import TensorDataset
from sklearn.neighbors import NearestNeighbors


class wrap_gene_location(TensorDataset):
    """
    Dataset for gene + covet features and spatial coordinates (2D or 3D).
    
    Args:
        datainput (np.ndarray): [num_spots, num_genes]
        label (pd.DataFrame or np.ndarray): [num_spots, 2] or [num_spots, 3]
        covetseq (np.ndarray): [num_spots, covet_dim]
    """
    def __init__(self, datainput, label, covetseq):
        # self.data_tensor = torch.from_numpy(datainput).float()  # [num_spots, num_genes]
        self.data_tensor = torch.from_numpy(datainput.astype('float32')) 

        try:
            cord = label.to_numpy().astype('float32')
        except:
            cord = label.astype('float32')
        self.coord_dim = cord.shape[1]  # 2 or 3

        self.covetseq = torch.from_numpy(covetseq.astype('float32')) 
        self.num_spots = self.data_tensor.shape[0]
        
        cmin = cord.min(axis=0)
        cmax = cord.max(axis=0)
        cord_norm = (cord - cmin) / (cmax - cmin)
        self.coord_tensor = torch.from_numpy(cord_norm.astype('float32')) 

        # Normalize coordinates
        # self.coord_norm = []
        # for i in range(self.coord_dim):
        #     coord_i = cord[:, i]
        #     cmin = coord_i.min() ; cmax = coord_i.max() 
        #     normed = (coord_i - cmin) / (cmax - cmin)
        #     self.coord_norm.append(normed)

    def __getitem__(self, index):
        # geneseq = self.data_tensor[index]  # shape: [num_genes]
        # coords = torch.tensor([self.coord_norm[i][index] for i in range(self.coord_dim)])
        # covet = self.covetseq[index]  # shape: [covet_dim]
        # feature_vec = torch.cat((geneseq, covet))  # final shape: [num_genes + covet_dim]
        # return feature_vec, coords
        geneseq = self.data_tensor[index]        # [num_genes]
        coords = self.coord_tensor[index]        # [coord_dim]
        covet = self.covetseq[index]             # [covet_dim]
        feature_vec = torch.cat((geneseq, covet))  # [num_genes + covet_dim]
        return feature_vec, coords

    def __len__(self):
        return self.num_spots


class wrap_gene_covet(TensorDataset):
    """
    Dataset for gene + coordinate and covet features.

    Args:
        datainput (np.ndarray): [num_spots, num_genes]
        label (np.ndarray or pd.DataFrame): [num_spots, 2 or 3] spatial coordinates
        covetseq (np.ndarray): [num_spots, covet_dim]
        trainloc (optional): another wrap_gene_covet_flex instance to provide mean/std
    """
    def __init__(self, datainput, label, covetseq, trainloc=None):
        # self.data_tensor = torch.from_numpy(datainput).float()  # [num_spots, num_genes]
        self.data_tensor = torch.from_numpy(datainput.astype("float32"))

        try:
            cord = label.to_numpy().astype('float32')
        except:
            cord = label.astype('float32')
            
        if trainloc is not None:
            try:
                trainloc = trainloc.to_numpy().astype('float32')
            except:
                trainloc = trainloc.astype('float32')

        self.coord_dim = cord.shape[1]  # 2 or 3
        self.covet_tensor = torch.from_numpy(covetseq.astype("float32"))
        self.num_spots = datainput.shape[0]

        means = cord.mean(axis=0) if trainloc is None else trainloc.mean(axis = 0)
        stds = cord.std(axis=0) * 0.25 if trainloc is None else trainloc.std(axis = 0) * 0.25

        # self.coords_norm = (cord - means) / stds
        self.coords_norm = torch.from_numpy(((cord - means) / stds).astype("float32"))


    def __getitem__(self, index):
        # geneseq = self.data_tensor[index]  # [num_genes]
        # coords = torch.tensor(self.coords_norm[index])  # [2 or 3]
        # covet = self.covetsq[index]  # [covet_dim]
        # feature_vec = torch.cat((geneseq, coords))  # [num_genes + 2/3]
        # return feature_vec, covet
        geneseq = self.data_tensor[index]        # [num_genes]
        coords = self.coords_norm[index]         # [2 or 3]
        covet = self.covet_tensor[index]         # [covet_dim]
        feature_vec = torch.cat((geneseq, coords))  # [num_genes + 2/3]
        return feature_vec, covet

    def __len__(self):
        return self.num_spots
    
    
class wrap_gene_rel(TensorDataset):
    """
    Dataset for gene + neighboring gene-gene covaraince features and relative distance.
    
    Args:
        datainput (np.ndarray): [num_spots, num_genes]
        label (pd.DataFrame or np.ndarray): [num_spots, 2] or [num_spots, 3]
        covetseq (np.ndarray): [num_spots, covet_dim]
    """
    def __init__(self, datainput_lst, label_lst, covetseq_lst, weights = None, sample_pairs=None, seed = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if not isinstance(datainput_lst, list):
            datainput_lst = [datainput_lst]
            label_lst = [label_lst]
            covetseq_lst = [covetseq_lst]
            weights = [1] if weights is None else [weights]
            
        self.expr_lst = []
        self.cord_lst = []
        self.covet_lst = []
        self.index_lst = []
        self.index_num_lst = [0]
        self.weight_lst = []
        if weights is None:
            weights = [1] * len(datainput_lst)
        for i in range(len(datainput_lst)):
            datainput = (datainput_lst[i].X.A if issparse(datainput_lst[i].X ) else datainput_lst[i].X)
            self.expr_lst.append(datainput)
            self.covet_lst.append(covetseq_lst[i])
            label = np.array(label_lst[i])
            self.cord_lst.append(label)
            rows, cols = np.tril_indices(datainput.shape[0])
            if sample_pairs is not None and len(rows) > sample_pairs:
                idx = np.random.choice(len(rows), sample_pairs, replace=False)
                rows = rows[idx]
                cols = cols[idx]
            rows += self.index_num_lst[-1]
            cols += self.index_num_lst[-1]
            self.index_lst.append(np.column_stack((rows, cols)))            
            self.index_num_lst.append(datainput.shape[0] + self.index_num_lst[-1] )
            self.weight_lst.extend([weights[i]] * len(rows))
            
        self.cord_lst = np.vstack(self.cord_lst)
        self.expr_lst = torch.from_numpy(np.vstack(self.expr_lst)).float()
        self.covet_lst = torch.from_numpy(np.vstack(self.covet_lst)).float()
        self.feature_lst = torch.concat((self.expr_lst, self.covet_lst), dim = 1)
        self.weight_lst = torch.from_numpy(np.array(self.weight_lst)).float()
        self.index_lst = np.vstack(self.index_lst).astype(np.int32)
        self.size = self.index_lst.shape[0]
        
    def __getitem__(self, index):
        indexcomb = self.index_lst[index, :]
        loc0 = self.cord_lst[indexcomb[0], :]; loc1 = self.cord_lst[indexcomb[1], :]
        dist = np.sqrt(np.sum((loc0 - loc1)**2))
        return self.feature_lst[indexcomb[0],:], self.feature_lst[indexcomb[1],:], torch.tensor([dist], dtype=torch.float32), self.weight_lst[index]
    def __len__(self):
        return self.size
    
    

class wrap_gene_rel_test(TensorDataset):
    """
    Dataset for gene + neighboring gene-gene covariance features and relative distance.
    Pairs are formed based on top feature-space neighbors (using NearestNeighbors).

    Args:
        datainput : AnnData
            Gene expression data [num_spots, num_genes]
        label : np.ndarray
            Spatial coordinates [num_spots, 2] or [num_spots, 3]
        covetseq : np.ndarray
            Covariates [num_spots, covet_dim]
        full_pairwise : logical, defauld False
            Whether to predict the full pairwise distance matrix for every cell pair, unnecessary for CN clustering
        neighbor_fraction : float
            If not predicting full pairwise distance matrix, we first filter neighbors based on feature neighbors. 
            Fraction (0-1) of feature neighbors to keep per cell
    """

    def __init__(
        self, datainput, label, covetseq, full_pairwise = False, neighbor_fraction=0.1):

        # --- Extract data ---
        datainput = datainput.X.A if issparse(datainput.X) else datainput.X
        covet = covetseq
        label = np.array(label)
        n = datainput.shape[0]

        feature_mat = np.concatenate([datainput, covet], axis=1)

        # --- Find top neighbors in feature space ---
        if full_pairwise:
            rows, cols = np.tril_indices(n)
            index_lst = np.column_stack((rows, cols)).astype(np.int32)
        else:
            n_neighbors = min(10000, int(n * neighbor_fraction))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
            nbrs.fit(feature_mat)
            _, indices = nbrs.kneighbors(feature_mat)

            # Remove self-pairs if they appear
            indices = indices[:, 1:] if np.any(indices[:, 0] == np.arange(n)) else indices
            n_neighbors = indices.shape[1]

            # --- Form index pairs (i, neighbor_j) ---
            rows = np.repeat(np.arange(n), n_neighbors)
            cols = indices.flatten()
            index_lst = np.column_stack((rows, cols)).astype(np.int32)

        # --- Convert to tensors ---
        expr_tensor = torch.from_numpy(datainput).float()
        covet_tensor = torch.from_numpy(covet).float()
        feature_tensor = torch.cat((expr_tensor, covet_tensor), dim=1)
        weight_tensor = torch.full((index_lst.shape[0],), 1.0, dtype=torch.float32)

        # --- Store ---
        self.cord = label
        self.feature = feature_tensor
        self.index_lst = index_lst
        self.weight = weight_tensor
        self.size = index_lst.shape[0]

    def __getitem__(self, index):
        i0, i1 = self.index_lst[index, :]
        loc0, loc1 = self.cord[i0, :], self.cord[i1, :]
        dist = np.sqrt(np.sum((loc0 - loc1) ** 2))
        return (
            self.feature[i0, :],
            self.feature[i1, :],
            torch.tensor([dist], dtype=torch.float32),
            self.weight[index]
        )

    def __len__(self):
        return self.size

