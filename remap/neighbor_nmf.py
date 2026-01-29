import numpy as np
import pandas as pd
import seaborn as sns
import os
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import MiniBatchNMF, NMF


def make_celltype_mat(celltype_annotation):
    celltype_annotation = pd.Series(celltype_annotation)
    celltypes = celltype_annotation.unique()
    mat = np.zeros((len(celltype_annotation), len(celltypes)), dtype=int)
    for i, ctype in enumerate(celltypes):
        mat[:, i] = (celltype_annotation == ctype).astype(int)
    return pd.DataFrame(mat, columns=celltypes)


## neighbor_radius: define three radii for 3-hop analysis
def find_neighbors(loc_pred, spatial, neighbor_radius = [40, 80, 120]):
    """
    Compute k-hop neighborhood-aggregated cell-type compositions around each spot/cell,
    using radius-based neighbor graphs at multiple radii.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        Must contain a column 'Cluster' giving the predicted cluster/cell type label
        for each observation. Index should align with `spatial` rows.

    spatial : np.ndarray
        Array of shape (n_obs, 2) giving (x, y) spatial coordinates for each observation.

    neighbor_radius : list[int | float]
        Radii used to construct neighborhood graphs. Interpreted here as:
        - "1-hop" graph: neighbors within neighbor_radius[0]
        - "2-hop" graph: neighbors within neighbor_radius[1]
        - "3-hop" graph: neighbors within neighbor_radius[2]

    Returns
    -------
    neighbor_khop_pred_all : pd.DataFrame
        DataFrame indexed by observation names with concatenated neighborhood
        compositions for each radius-shell.
        Columns are prefixed as "1-hop_", "2-hop_", "3-hop_".
        Each block is row-normalized to sum to 1 (unless a row has no neighbors,
        in which case it stays all zeros).
    """
    ct_lst = np.unique(loc_pred['Cluster'])
    # Initialize container
    neighbor_khop_pred = {
        '1-hop': pd.DataFrame(columns=ct_lst),
        '2-hop': pd.DataFrame(columns=ct_lst),
        '3-hop': pd.DataFrame(columns=ct_lst)
    }
    # Prepare AnnData
    ct_pred = make_celltype_mat(loc_pred['Cluster'])
    adata = sc.AnnData(X=ct_pred.values)
    adata.var_names = ct_pred.columns.astype(str)
    adata.obs_names = loc_pred.index.astype(str)
    adata.obsm['spatial'] = spatial
    # Build neighbor graphs for 0–3 hops
    
    adata.obsp['0-hop'] = sp.csr_matrix((adata.shape[0], adata.shape[0]), dtype=np.float32)
    coords = adata.obsm['spatial']
    
    for i, r in enumerate(neighbor_radius, start=1):
        adj = radius_neighbors_graph(coords, radius=r, mode="connectivity", include_self=False)
        adata.obsp[f"{i}-hop"] = adj

    # Aggregate k-hop predictions
    for khop in range(len(neighbor_radius)):
        neighbor_mat = ((adata.obsp[f"{khop+1}-hop"] - adata.obsp[f"{khop}-hop"]) @ adata.X)
        neighbor_mat = np.array(neighbor_mat)
        neighbor_mat[neighbor_mat < 0] = 0
        
        # neighbor_mat = neighbor_mat / neighbor_mat.sum(axis=1, keepdims=True)
        row_sums = neighbor_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        neighbor_mat = neighbor_mat / row_sums
        
        neighbor_mat = pd.DataFrame(neighbor_mat, columns=ct_pred.columns, index=adata.obs_names)
        neighbor_mat = neighbor_mat.reindex(columns=ct_lst, fill_value=0)
        neighbor_khop_pred[f"{khop+1}-hop"] = neighbor_mat
    # Add prefixes and concatenate
    neighbor_khop_pred['1-hop'] = neighbor_khop_pred['1-hop'].add_prefix("1-hop_")
    neighbor_khop_pred['2-hop'] = neighbor_khop_pred['2-hop'].add_prefix("2-hop_")
    neighbor_khop_pred['3-hop'] = neighbor_khop_pred['3-hop'].add_prefix("3-hop_")
    neighbor_khop_pred_all = pd.concat(
        [neighbor_khop_pred['1-hop'],
         neighbor_khop_pred['2-hop'],
         neighbor_khop_pred['3-hop']],
        axis=1)
    return neighbor_khop_pred_all


def nmf_neighbor(neighbor_khop_pred_all, ct_lst, num_factor, verbose = 0, random_state=123):
    """
    Factorize neighborhood composition features (concatenated 1/2/3-hop blocks) using NMF.

    This is used to extract a small number of latent "neighborhood patterns" (factors)
    from high-dimensional neighbor composition vectors.

    Parameters
    ----------
    neighbor_khop_pred_all : pd.DataFrame
        Shape (n_obs, 3 * n_ct) if you concatenated 1-hop/2-hop/3-hop blocks.
        Columns are expected to be ordered in blocks:
            [1-hop_{ct1..ctK}, 2-hop_{ct1..ctK}, 3-hop_{ct1..ctK}]
        Rows correspond to observations (cells/spots/patches).

    ct_lst : array-like
        List/array of cluster/cell-type names (length = n_ct). Used only to infer
        block boundaries (each block assumed size = len(ct_lst)).

    num_factor : int
        Number of latent NMF components (rank) to learn.

    verbose : int
        Verbosity level passed to MiniBatchNMF.

    random_state : int
        Random seed for NMF initialization and minibatch updates.

    Returns
    -------
    W_pred_rescaled : pd.DataFrame
        Shape (num_factor, n_features). This corresponds to the *components* matrix
        (often called "W" or "H" depending on convention; sklearn uses `components_`).
        Here each row is a factor, each column is a feature (e.g., 1-hop_celltypeA).

    H_pred_rescaled : pd.DataFrame
        Shape (n_obs, num_factor). This is the *activation/usage* matrix for each
        observation (sklearn's transform output).
    """
    model_pred = MiniBatchNMF(n_components=num_factor, init='nndsvd', max_no_improvement=10, batch_size=50000,
                                random_state=random_state, max_iter=10000, tol=1e-4, verbose=verbose)
    # model_pred = NMF(n_components=num_factor, init='nndsvd', random_state=random_state, max_iter=5000, verbose=verbose)
    model_pred.fit(neighbor_khop_pred_all.values)
    H_pred = model_pred.transform(neighbor_khop_pred_all.values)
    W_pred = model_pred.components_
    W_pred = pd.DataFrame(W_pred, columns=neighbor_khop_pred_all.columns)
    vlines = [len(ct_lst), 2*len(ct_lst)]
    W_pred_rescaled = W_pred.copy()
    block_edges = [0] + vlines + [W_pred.shape[1]]
    for i in range(len(block_edges) - 1):
        start, end = block_edges[i], block_edges[i+1]
        block = W_pred.iloc[:, start:end]
        block_sum = block.sum(axis=1).replace(0, np.nan)  # avoid divide-by-zero
        W_pred_rescaled.iloc[:, start:end] = block.div(block_sum, axis=0)
    W_pred_rescaled = W_pred_rescaled.fillna(0)
    H_pred = pd.DataFrame(H_pred)
    H_pred.index = neighbor_khop_pred_all.index
    row_sums = H_pred.sum(axis=1).replace(0, np.nan)
    H_pred_rescaled = H_pred.div(row_sums, axis=0).fillna(0) 
    return W_pred_rescaled, H_pred_rescaled


def nmf_bar(W_pred, plot_path, ct_lst, colors):
    column_colors = [colors[col.split('_')[-1]] for col in W_pred]
    vlines = [len(ct_lst), 2*len(ct_lst)]
    os.makedirs(plot_path, exist_ok = True)
    fig, axes = plt.subplots(W_pred.shape[0], 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0})
    for i, ax in enumerate(axes):
        W_pred.iloc[i].plot(kind='bar', ax=ax, color=column_colors)
        ax.set_ylabel(f"Factor {i}")
        ax.tick_params(axis='x', rotation=90)
        for x in vlines:
            ax.axvline(x - 0.5, color='black', linestyle='--', linewidth=1)
    fig.savefig(f"{plot_path}/H_nmf_bar.png", dpi= 300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    

def nmf_main(loc_pred, spatial, path, colors, ct_key = 'Cluster', neighbor_radius = [40, 80, 120], ct_interest = None, num_factor = 6, random_state=123):
    """
    End-to-end pipeline for neighborhood-based NMF analysis.

    This function:
      1) Computes multi-radius (1/2/3-hop) neighborhood cell-type compositions
         from spatial coordinates.
      2) Applies NMF to extract latent neighborhood patterns.
      3) Visualizes the learned patterns and saves results to disk.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        Must contain a column 'Cluster' with predicted cluster / cell-type labels.
        Index corresponds to observations (cells/spots/patches).

    spatial : array-like
        Spatial coordinates of shape (n_obs, 2). Will be converted to np.ndarray
        and aligned with loc_pred.index.

    path : str
        Base directory where NMF results will be saved.
        A subdirectory named:
            nmf_<r1>_<r2>_<r3>
        will be created automatically.

    colors : dict or list
        Color mapping used by `nmf_bar` to visualize cell-type compositions.
        Typically maps ct_lst entries to colors.
        
    ct_interest : str or None, default None
        If provided, restricts the analysis to **anchor observations**
        whose cell type equals `ct_interest`.

        Specifically:
          - NMF is performed only on rows corresponding to observations
            with `loc_pred[ct_key] == ct_interest`.

        This allows learning neighborhood patterns **centered around a specific cell type. If None, all observations are used as anchors.

    neighbor_radius : list[int | float], default [40, 80, 120]
        Radii (in um) used to define neighborhood shells for 1-hop, 2-hop, and 3-hop
        aggregation. These are increasing spatial radii, not graph-theoretic hops.

    num_factor : int, default 6
        Number of NMF factors (latent neighborhood patterns) to learn.
        
    ct_interest : str
        Number of NMF factors (latent neighborhood patterns) to learn.

    random_state : int, default 123
        Random seed controlling NMF initialization and minibatch updates.

    Outputs
    -------
    Files written to disk:
      - <path>/nmf_<r1>_<r2>_<r3>/H_nmf.csv
            Row-normalized factor activations per observation
            (n_obs x num_factor).
      - <path>/nmf_<r1>_<r2>_<r3>/W_nmf.csv
            Block-wise normalized factor patterns
            (num_factor x n_features).
      - Plots generated by nmf_bar(...) summarizing factor compositions.
    """
    os.makedirs(path, exist_ok=True)
    path = f"{path}/nmf_{neighbor_radius[0]}_{neighbor_radius[1]}_{neighbor_radius[2]}"
    ct_lst = np.unique(loc_pred[ct_key])
    spatial = np.array(spatial)

    neighbor_khop_pred_all = find_neighbors(loc_pred, spatial = spatial, neighbor_radius = neighbor_radius)
    if ct_interest is not None:
        neighbor_khop_pred_all = neighbor_khop_pred_all[loc_pred[ct_key].isin([ct_interest]) ]
    W_pred, H_pred = nmf_neighbor(neighbor_khop_pred_all, ct_lst = ct_lst, num_factor = num_factor, random_state=random_state)
    
    nmf_bar(W_pred, plot_path = path, ct_lst = ct_lst, colors = colors)
    H_pred.to_csv(f"{path}/H_nmf.csv")
    W_pred.to_csv(f"{path}/W_nmf.csv")
