import os
import sys
import math
import numpy as np
import scanpy as sc
import scipy.sparse
import scipy.special
import sklearn.neighbors
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import Poisson, NegativeBinomial, Normal


# def matrix_square_root(mats):
#     """
#     Computes the matrix square root for a batch of symmetric positive semi-definite matrices.
#     Equivalent to sqrtm(M) = V diag(sqrt(e)) V^T using eigen decomposition.
#     Args:
#         mats: Tensor of shape (B, N, N) on GPU or CPU
#     Returns:
#         Tensor of shape (B, N, N) representing sqrt of input matrices
#     """
#     # Compute eigenvalues and eigenvectors
#     mats = torch.from_numpy(mats).to(dtype=torch.float32, device="cuda")
#     e, v = torch.linalg.eigh(mats)  # e: (B, N), v: (B, N, N)

#     # Clamp eigenvalues to remove negatives (numerical stability)
#     e = torch.clamp(e, min=0.0)
#     sqrt_e = torch.sqrt(e)

#     # Form diagonal matrix of sqrt eigenvalues
#     diag_sqrt_e = torch.diag_embed(sqrt_e)  # (B, N, N)

#     # Reconstruct sqrt matrix: V diag(sqrt(e)) V^T
#     sqrt_mats = v @ diag_sqrt_e @ v.transpose(-2, -1)
#     return sqrt_mats.cpu().numpy()

def matrix_square_root(mats) -> torch.Tensor:
    """
    Computes the matrix square root for a batch of symmetric positive semi-definite matrices.
    Equivalent to sqrtm(M) = V diag(sqrt(e)) V^T using eigen decomposition.
    Args:
        mats: Tensor of shape (B, N, N) on GPU or CPU
    Returns:
        Tensor of shape (B, N, N) representing sqrt of input matrices
    """
    # Compute eigenvalues and eigenvectors
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    mats = torch.from_numpy(mats).to(dtype=torch.float32, device=device)
    e, v = torch.linalg.eigh(mats)  # e: (B, N), v: (B, N, N)

    # Clamp eigenvalues to remove negatives (numerical stability)
    e = torch.clamp(e, min=0.0)
    sqrt_e = torch.sqrt(e)

    # Form diagonal matrix of sqrt eigenvalues
    diag_sqrt_e = torch.diag_embed(sqrt_e)  # (B, N, N)

    # Reconstruct sqrt matrix: V diag(sqrt(e)) V^T
    sqrt_mats = v @ diag_sqrt_e @ v.transpose(-2, -1)
    return sqrt_mats.cpu().numpy()


def batch_knn(data, batch, k):
    """
    Computes kNN matrix for spatial data from multiple batches

    Args:
        data (array): Data to compute kNN on
        batch (array): Batch allocation per sample in Data
        k (int): number of neighbors for kNN matrix
    Return:
        knn_graph_index (np.array): indices of each sample's k nearest-neighbors
        weighted_index (np.array): Weighted (softmax) distance to each nearest-neighbors
    """

    knn_graph_index = np.zeros(shape=(data.shape[0], k))
    weighted_index = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )

        batch_knn_weight = scipy.special.softmax(
            -np.reshape(batch_knn.data, [data[val_ind].shape[0], k]), axis=-1
        )

        knn_graph_index[val_ind] = val_ind[batch_knn_ind]
        weighted_index[val_ind] = batch_knn_weight
    return (knn_graph_index.astype("int"), weighted_index)


def get_niche_expression(
    spatial_data, k, spatial_key="spatial", batch_key=-1, data_key=None
):
    """
    Computing Niche mean expression based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        data_key (str): obsm key to compute niche mean across
            (default None, uses gene expression .X)

    Return:
        niche_expression: Average gene expression in niche
        knn_graph_index: indices of nearest spatial neighbors per cell
    """

    if data_key is None:
        Data = spatial_data.X
    else:
        Data = spatial_data.obsm[data_key]

    if batch_key == -1:
        knn_graph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        knn_graph = scipy.sparse.coo_matrix(
            (np.ones_like(knn_graph.data), (knn_graph.row, knn_graph.col)),
            shape=knn_graph.shape,
        )
        knn_graph_index = np.reshape(
            np.asarray(knn_graph.col), [spatial_data.obsm[spatial_key].shape[0], k]
        )
    else:
        knn_graph_index, _ = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], k
        )

    return Data[knn_graph_index[np.arange(spatial_data.obsm[spatial_key].shape[0])]]


def compute_covet(
    spatial_data,
    k,
    spatial_key="spatial",
    batch_key=-1,
    mean_expression=None,
    weighted=False,
    covet_pseudocount=1,
):
    """
    Wrapper to compute COVET based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        mean_expression (np.array): expression vector to shift COVET with
        weighted (bool): if True, weights COVET by spatial distance
    Return:
        covet: COVET matrices
        knn_graph_index: indices of nearest spatial neighbors per cell
    """
    expression_data = spatial_data[:, spatial_data.var.highly_variable].X

    if covet_pseudocount > 0:
        expression_data = np.log(expression_data + covet_pseudocount)

    if batch_key == -1 or batch_key not in spatial_data.obs.columns:
        knn_graph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        knn_graph.data = knn_graph.data.astype(np.float32)
        knn_graph_index = np.reshape(
            np.asarray(knn_graph.col), [spatial_data.obsm[spatial_key].shape[0], k]
        )
        weighted_index = scipy.special.softmax(
            -np.reshape(knn_graph.data, [spatial_data.obsm[spatial_key].shape[0], k]),
            axis=-1,
        )
    else:
        knn_graph_index, weighted_index = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], k
        )

    if not weighted:
        weighted_index = np.ones_like(weighted_index) / k

    if mean_expression is None:
        weighted_distance_matrix = (
            (
                expression_data.mean(axis=0, dtype=np.float32)[None, None, :]
                - expression_data[knn_graph_index[np.arange(expression_data.shape[0])]]
            ).astype(np.float32)
            * np.sqrt(weighted_index.astype(np.float32))[:, :, None]
            * np.sqrt(1.0 / (1.0 - np.sum(np.square(weighted_index.astype(np.float32)), axis=-1)))[
                :, None, None
            ]
        )
    else:
        weighted_distance_matrix = (
        (
            mean_expression.astype(np.float32)[:, None, :]
            - expression_data[knn_graph_index[np.arange(expression_data.shape[0])]].astype(np.float32)
        )
        * np.sqrt(weighted_index.astype(np.float32))[:, :, None]
        * np.sqrt(1.0 / (1.0 - np.sum(np.square(weighted_index.astype(np.float32)), axis=-1)))[
            :, None, None
        ])

    covet = np.matmul(
        weighted_distance_matrix.transpose([0, 2, 1]), weighted_distance_matrix
    )
    covet = covet.astype('float32')
    covet = covet + covet.mean() * 0.00001 * np.expand_dims(
        np.identity(covet.shape[-1], dtype=np.float32), axis=0
    )
    del weighted_distance_matrix, weighted_index
    return covet



def get_covet(
    spatial_data,
    k,
    g,
    genes,
    covet_distribution,
    spatial_key="spatial",
    batch_key=-1,
    covet_pseudocount=1,
    covet_batch_size = None
):
    """
    Compute COVET matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        g (int): number of HVG to compute COVET matrices
        genes (list of str): list of genes to keep for COVET
        covet_distribution (str): distribution to transform COVET matrices to fit into
        batch_key (str): obs key for batch information (default -1, for no batch)

    Return:
        covet: raw, untransformed COVET matrices
        covet_sqrt: COVET transformed for covet_distribution
        niche_expression: Average gene expression in niche
        covet_genes: Genes used for COVET
    """

    spatial_data = spatial_data.copy()
    spatial_data.layers["log"] = np.log(spatial_data.X + 1)

    if g == -1:
        covet_gene_set = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
        if g == 0:
            spatial_data.var.highly_variable = False
        if len(genes) > 0:
            spatial_data.var["highly_variable"][genes] = True

    covet_gene_set = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    covet_genes = spatial_data.var_names[covet_gene_set]
    print(f"Number of genes used for calculating neighboring gene-gene covariance: {len(covet_genes)}")

    covet = compute_covet(
        spatial_data,
        k,
        spatial_key=spatial_key,
        batch_key=batch_key,
        weighted=False,
        covet_pseudocount=covet_pseudocount,
    )

    # if covet_distribution == "norm":
    #     covet_sqrt = covet.reshape([covet.shape[0], -1])
    #     covet_sqrt = (
    #         covet_sqrt - covet_sqrt.mean(axis=0, keepdims=True)
    #     ) / covet_sqrt.std(axis=0, keepdims=True)
    if covet_distribution == "OT":
        if covet_batch_size is None or covet_batch_size >= covet.shape[0]:
            covet_sqrt = matrix_square_root(covet)
        else:
            # Process matrix square root in batches too
            n_cells = covet.shape[0]
            covet_sqrt = np.zeros_like(covet)
            
            # Split into batches
            batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / covet_batch_size))
            
            for batch_idx in tqdm(batch_indices, desc="Computing matrix square roots"):
                # Process this batch of matrices
                batch_sqrt = matrix_square_root(covet[batch_idx])
                covet_sqrt[batch_idx] = batch_sqrt
    else:
        covet_sqrt = np.copy(covet)

    return (
        covet,
        covet_sqrt,
        covet_genes,
    )
    
    
def normal_kl(mean, log_std):
    """
    Compute KL divergence between N(mean, std) and N(0, 1)
    """
    return 0.5 * (mean.pow(2) + torch.exp(log_std).pow(2) - 2 * log_std).mean(dim=-1)



def log_pos_pdf(sample, l):
    rate = l
    sample = torch.clamp(sample, min=1e-8)
    log_prob = (
        sample * torch.log(rate)
        - rate
        - torch.lgamma(sample + 1.0)
    )

    return log_prob.mean(dim=-1)

def log_nb_pdf(sample, r, p_logits):
    total_count = r; logits = p_logits
    # Ensure sample is float and positive
    sample = torch.clamp(sample, min=1e-8)

    # Use log-Gamma-based formulation of NB log-prob (safe for floats)
    log_r = torch.log(total_count + 1e-8)
    log_1_minus_p = torch.nn.functional.softplus(logits)  # log(1 + exp(logits))
    log_p = -torch.nn.functional.softplus(-logits)        # log(sigmoid(logits))

    log_unnormalized_prob = (
        torch.lgamma(sample + total_count)
        - torch.lgamma(sample + 1.0)
        - torch.lgamma(total_count)
        + total_count * torch.nn.functional.logsigmoid(-logits)  # log(1 - p)
        + sample * torch.nn.functional.logsigmoid(logits) 
        # + total_count * log_1_minus_p
        # + sample * log_p
    )
    return log_unnormalized_prob.mean(dim=-1)

def log_zinb_pdf(sample, r, p_logits, d_logits):
    """
    Log-likelihood under a Zero-Inflated Negative Binomial (ZINB) distribution
    r: total_count
    p_logits: logits for NB
    d_logits: logits for dropout (zero inflation)
    """
    nb_dist = NegativeBinomial(total_count=r, logits=p_logits)
    pi = torch.sigmoid(d_logits)  # probability of zero inflation

    nb_log_prob = nb_dist.log_prob(sample)
    zero_mask = (sample == 0).float()

    # Combine inflated (zero) and regular NB
    log_prob = torch.log(
        pi * zero_mask + (1 - pi) * torch.exp(nb_log_prob) + 1e-8  # to avoid log(0)
    )
    return log_prob.mean(dim=-1)

def log_normal_pdf(sample, mean, std=1.0):
    """
    Log-likelihood under a Normal distribution
    """
    dist = Normal(loc=mean, scale=std)
    log_prob = dist.log_prob(sample)
    return log_prob.mean(dim=-1)

def ot_distance(sample, mean):
    """
    Negative squared difference (Average Over Time distance analog)
    """
    sample = sample.view(sample.shape[0], -1)
    mean = mean.view(mean.shape[0], -1)
    log_prob = -torch.square(sample - mean)
    return log_prob.mean(dim=-1)

def fill_triangular(vec):
    """
    Reconstruct a lower-triangular matrix from a vector.

    Args:
        vec: Tensor of shape [B, L], where L = d * (d + 1) // 2

    Returns:
        Tensor of shape [B, d, d], where d is inferred
    """
    B, L = vec.shape
    # Solve L = d * (d + 1) // 2 → d = (sqrt(8L + 1) - 1) / 2
    d = int((math.sqrt(8 * L + 1) - 1) / 2)

    tril_indices = torch.tril_indices(row=d, col=d, offset=0, device=vec.device)
    mat = torch.zeros((B, d, d), device=vec.device, dtype=vec.dtype)
    mat[:, tril_indices[0], tril_indices[1]] = vec
    return mat


def truncated_normal_(tensor, mean=0.0, std=1.0):
    """Fills the tensor with values from a truncated normal distribution."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
        

def init_weights(layer, std):
    """Applies truncated normal to both weight and bias of a Linear layer."""
    truncated_normal_(layer.weight, std=std)
    if layer.bias is not None:
        truncated_normal_(layer.bias, std=std)

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, weight_std, bias_std):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # init_weights(self.linear, weight_std, bias_std)
        truncated_normal_(self.linear.weight, std=weight_std)
        truncated_normal_(self.linear.bias, std=bias_std)

    def forward(self, x):
        return self.linear(x)


class ConstantLayer(nn.Module):
    def __init__(self, output_dim, bias_std):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(output_dim))
        truncated_normal_(self.bias, std=bias_std)

    def forward(self, x):
        # Expand bias to match batch size
        return self.bias.expand(x.size(0), -1)
    
    
class ENVIOutputLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_init,  # This should be std for truncated_normal_
        bias_init,    # Same as above
        spatial_distribution="pois",
        sc_distribution="nb",
        share_dispersion=False,
        const_dispersion=False,
        name="dec_exp_output",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_distribution = spatial_distribution
        self.sc_distribution = sc_distribution
        self.share_dispersion = share_dispersion
        self.const_dispersion = const_dispersion
        self._name = name
        self.kernel_std = kernel_init
        self.bias_std = bias_init

        # Required output: rate (r)
        self.r = LinearLayer(input_dim, output_dim, self.kernel_std, self.bias_std)

        # Conditionally build other parameter layers
        self.init_dispersion_layers()

    def dist_has_p(self, mode="spatial"):
        p_dists = {"zinb", "nb", "full_norm"}
        if self.share_dispersion:
            return self.spatial_distribution in p_dists or self.sc_distribution in p_dists
        return getattr(self, f"{mode}_distribution") in p_dists

    def dist_has_d(self, mode="spatial"):
        d_dists = {"zinb"}
        if self.share_dispersion:
            return self.spatial_distribution in d_dists or self.sc_distribution in d_dists
        return getattr(self, f"{mode}_distribution") in d_dists

    def init_layer(self, name_suffix):
        if self.const_dispersion:
            return ConstantLayer(self.output_dim, self.bias_std)
        return LinearLayer(self.input_dim, self.output_dim, self.kernel_std, self.bias_std)

    def init_dispersion_layers(self):
        if self.dist_has_p("spatial"):
            self.p_spatial = self.init_layer("_p_spatial")
            if self.share_dispersion:
                self.p_sc = self.p_spatial
        if self.dist_has_d("spatial"):
            self.d_spatial = self.init_layer("_d_spatial")
            if self.share_dispersion:
                self.d_sc = self.d_spatial

        if not self.share_dispersion:
            if self.dist_has_p("sc"):
                self.p_sc = self.init_layer("_p_sc")
            if self.dist_has_d("sc"):
                self.d_sc = self.init_layer("_d_sc")

    def forward(self, x, mode="spatial"):
        r = self.r(x)

        dist = self.spatial_distribution if mode == "spatial" else self.sc_distribution

        if dist == "zinb":
            p = getattr(self, f"p_{mode}")(x)
            d = getattr(self, f"d_{mode}")(x)
            return r, p, d

        if dist in {"nb", "full_norm"}:
            p = getattr(self, f"p_{mode}")(x)
            return r, p

        return r