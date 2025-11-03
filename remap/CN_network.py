import networkx as nx
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.stats import mstats
from scipy.optimize import linear_sum_assignment
import matplotlib.colors as mcolors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform


def compute_avg_distance(A, B):
    distances = np.sqrt(((A[:, np.newaxis, :] - B[np.newaxis, :, :])**2).sum(axis=2))
    # avg_distances = distances.mean(axis=1)
    return np.mean(distances)

def get_frac(loc, loc_pred, knn = 100, ct_key = "Cluster"):
    """
    Compute local neighborhood cell type fractions for each cell.

    Parameters
    ----------
    loc : array-like, shape (n_cells, n_dims)
        Spatial coordinates of cells (e.g., predicted or true locations).
    loc_pred : DataFrame
        Must contain a column 'Cluster' with categorical cluster assignments.
    knn : int, default=100
        Number of nearest neighbors to use when computing neighborhood composition.

    Returns
    -------
    fractions : ndarray, shape (n_cells, n_clusters)
        For each cell, the fraction of neighbors belonging to each cluster type.
    """
    loc = np.array(loc)
    unique_types, encoded_types = np.unique(loc_pred[ct_key], return_inverse=True)
    num_types = len(unique_types)
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='auto').fit(loc)
    _, indices = nbrs.kneighbors(loc)
    indices = indices[:, 1:]

    one_hot_neighbors = np.eye(num_types)[encoded_types[indices]]
    type_counts = one_hot_neighbors.sum(axis=1)
    fractions = type_counts / type_counts.sum(axis=1, keepdims=True)
    return fractions


def cn_cluster_loc(loc_pred, location, ct_key = "Cluster", n_clust = 10, knn = 100, random_state = 1):
    """
    Cluster cells into neighborhood-based clusters (CN clusters) based on local cell type composition.

    Parameters
    ----------
    loc_pred : pandas.DataFrame
        DataFrame containing cell-level predictions, must have a column 'Cluster' for cell types.
    location : array-like, shape (n_cells, n_dims)
        Spatial coordinates of cells.
    cn_key : str
        Column name to store the resulting neighborhood cluster assignments.
    n_clust : int, default=10
        Number of CN clusters.
    knn : int, default=100
        Number of nearest neighbors used to compute local cell type fractions.
    random_state : int, default=1
        Random seed for KMeans clustering.

    Returns
    -------
    loc_pred : pandas.DataFrame
        Original DataFrame with an additional column (`cn_key`) containing the neighborhood cluster labels.
    """
    location = np.array(location)
    location = (location - np.min(location, axis=0)) / (np.max(location, axis=0) - np.min(location, axis=0))
    km = MiniBatchKMeans(n_clusters = n_clust, random_state=random_state)
    # km = KMeans(n_clusters = n_clust, random_state=random_state)
    frac_true = get_frac(location, loc_pred, ct_key=ct_key, knn=knn)
    labelskm = km.fit_predict(frac_true) + 1
    return labelskm.astype(str)


def cn_cluster_dist(loc_pred, dist_matrix, ct_key = "Cluster", n_clust=10, knn=100, random_state=1):
    """
    Cluster cells based on the pairwise distance matrix.

    This function computes a k-nearest neighbor (kNN) cluster composition for each cell
    and then performs KMeans clustering on these compositions to assign new clusters.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame containing cluster assignments for each observation (e.g., cell types).
    dist_matrix : np.ndarray
        Square distance matrix (n_samples x n_samples) representing pairwise distances.
    ct_key : str, optional
        Column in `loc_pred` containing initial cluster labels. Default is "Cluster".
    n_clust : int, optional
        Number of new clusters to generate via KMeans. Default is 10.
    knn : int, optional
        Number of nearest neighbors to consider for cluster composition. Default is 100.
    random_state : int, optional
        Random seed for reproducibility in KMeans. Default is 1.

    Returns
    -------
    np.ndarray
        Array of new cluster assignments (as strings) for each cell.
    """
    # dist_matrix = squareform(dist_matrix)
    clusters = loc_pred[ct_key].values
    n = dist_matrix.shape[0]
    dist_matrix = dist_matrix.A if issparse(dist_matrix) else dist_matrix
    dist_matrix = np.where(np.isfinite(dist_matrix ), dist_matrix, np.nanmax(dist_matrix[np.isfinite(dist_matrix)]) * 2)
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :knn]

    try:
        cluster_uniq = np.sort(np.unique(clusters).astype(int)).astype(str)
    except:
        cluster_uniq = np.sort(np.unique(clusters))
    cluster_fraction_matrix = np.zeros((n, len(cluster_uniq)))
 
    for i, neighbors in enumerate(knn_indices):
        neighbor_clusters = clusters[neighbors]
        for j, uc in enumerate(cluster_uniq):
            cluster_fraction_matrix[i, j] = np.sum(neighbor_clusters == uc) / knn
   
    kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=random_state)
    new_clusters = kmeans.fit_predict(cluster_fraction_matrix) + 1
    return new_clusters.astype(str)


def dist_center(loc_pred, Qdata_loc, key="CN_true"):
    """
    Compute pairwise distances between the centroids of clusters in normalized spatial coordinates.

    Parameters
    ----------
    loc_pred : pandas.DataFrame
        DataFrame containing cell information and cluster assignments (e.g., CN clusters).
    Qdata_loc : array-like, shape (n_cells, 2)
        Spatial coordinates of cells corresponding to `loc_pred`.
    key : str, default="CN_true"
        Column in `loc_pred` used to group cells into clusters.

    Returns
    -------
    dist : numpy.ndarray, shape (n_clusters, n_clusters)
        Pairwise distance matrix between cluster centroids.
        The distance is computed as the inverse of Euclidean distance between centroids.
    loc_mean : pandas.DataFrame
        DataFrame containing the mean (centroid) coordinates for each cluster.
        Columns: ['x_norm', 'y_norm'], indexed by cluster labels.
    """
    loc_pred_copy = loc_pred.copy()
    Qdata_loc = np.array(Qdata_loc)
    Qdata_loc = (Qdata_loc - np.min(Qdata_loc, axis=0)) / (np.max(Qdata_loc, axis=0) - np.min(Qdata_loc, axis=0))
    loc_pred_copy[['x_norm', 'y_norm']] = Qdata_loc
    loc_mean = loc_pred_copy.groupby(by=key)[['x_norm', 'y_norm']].mean()
    dist = squareform(1/pdist(loc_mean.values, metric='euclidean'))
    return dist, loc_mean

def adjust_xy_locations(points_df, min_distance=0.08, max_iterations=100):
    """
    Adjust XY coordinates to prevent points from overlapping or being too close.

    This function iteratively moves points apart if their pairwise distance is
    less than `min_distance`. The movement is symmetric between the two points.

    Parameters
    ----------
    points_df : pd.DataFrame
        DataFrame of point coordinates (n_points x 2 or n_points x 3). Index corresponds
        to point IDs.
    min_distance : float, optional
        Minimum allowed distance between any two points. Default is 0.08.
    max_iterations : int, optional
        Maximum number of iterations to adjust points. Default is 100.

    Returns
    -------
    pd.DataFrame
        Adjusted point coordinates with the same index and columns as `points_df`.
    """
    points = np.array(points_df)
    n = points.shape[0]
    
    for _ in range(max_iterations):  # Prevent infinite loops
        adjusted = False
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(points[i] - points[j])
                if distance < min_distance:  # If too close, adjust
                    move_dist = (min_distance - distance) / 2
                    direction = (points[i] - points[j])  # Vector direction
                    norm = np.linalg.norm(direction) + 1e-7  # Avoid division by zero
                    move_vector = (direction / norm) * move_dist
                    
                    points[i] += move_vector
                    points[j] -= move_vector
                    adjusted = True
        
        if not adjusted:
            break  # Stop if no adjustments were made
    loc_adj = pd.DataFrame(points, index=points_df.index, columns=points_df.columns)
    return loc_adj

def draw_graph_loc(loc_mean, dist_node, thre, sd_node, colors, coef = 1000, lw_coef=0.5, title = "", save_path = None, filename = "CN_network"):
    """
    Construct CN spatial network.

    Parameters
    ----------
    loc_mean : pandas.DataFrame
        DataFrame containing cluster centroids with normalized coordinates.
        Shape: (n_clusters, 2), columns correspond to ['x', 'y'].
    dist_node : numpy.ndarray
        Pairwise distance matrix between cluster centroids.
        Shape: (n_clusters, n_clusters).
    thre : float, optional
        Threshold for drawing edges. If <1, interpreted as quantile of non-zero distances; 
        if >=1, interpreted as absolute distance threshold.
    sd_node : dict-like
        Dictionary mapping cluster index to node size (e.g., standard deviation of cluster spread).
    colors : dict-like
        Dictionary mapping cluster index to node color.
    coef : float, default=1000
        Scaling factor for node sizes.
    lw_coef : float, default=0.5
        Exponent to scale edge width based on distance: width = distance**lw_coef.
    """
    
    if thre < 1:
        thre_num = np.quantile(dist_node[dist_node != 0], thre)
        pos = np.where(dist_node > thre_num)
    else:
        pos = np.where(dist_node > thre)
    fig = plt.figure(figsize=(5, 5))
    plt.xlim(0,1)
    plt.ylim(0,1)
    loc_mean = adjust_xy_locations(loc_mean, min_distance=0.08)
    for i in range(loc_mean.shape[0]):
        loc = np.array(loc_mean.iloc[i, :])
        plt.scatter([loc[0]],[loc[1]],c = colors[loc_mean.index[i]], s = sd_node[loc_mean.index[i]] * coef, zorder = 3)
        
    for j in range(len(pos[0])):
        e0 = pos[0][j]; e1 = pos[1][j]; p = dist_node[e0, e1]
        pos_0 = loc_mean.iloc[e0, :]; pos_1 = loc_mean.iloc[e1, :]
        plt.plot([ pos_0.iloc[0], pos_1.iloc[0] ],[ pos_0.iloc[1], pos_1.iloc[1] ], c= 'black',linewidth = p**lw_coef)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Adjust line width
        spine.set_edgecolor('black')  # Adjust frame color
    plt.show()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)


def node_var(loc_pred, location, key):
    """
    Compute the standard deviation of pairwise distances within each cluster

    Parameters
    ----------
    loc_pred : pandas.DataFrame
        DataFrame containing spatial coordinates and cluster assignments.
        Must have columns corresponding to `x_{method}`, `y_{method}`, and `CN_{name}`.
    clust_uniq : array-like
        List or array of unique cluster identifiers to compute pairwise distance variability for.
    method : str
        String used to select the coordinate columns, e.g., 'true' or 'pred'. 
        The function will use columns `x_{method}` and `y_{method}`.
    key : str
        CN cluster key name

    Returns
    -------
    pair_std_lst : dict
        Dictionary mapping each cluster identifier to the standard deviation of pairwise distances 
        among points within that cluster. Distances are winsorized at the top 5% to reduce outlier effects.
    """
    
    pair_std_lst = {}
    # clust_uniq = loc_pred[key].cat.categories
    clust_uniq = loc_pred[key].unique()
    for i in range(len(clust_uniq)):
        pair_dist = pdist(np.array(location[loc_pred[key] == clust_uniq[i]] ), metric='euclidean')
        pair_dist = mstats.winsorize(pair_dist, limits=[0.0, 0.05])
        pair_std = np.std(pair_dist)
        pair_std_lst[clust_uniq[i]] = pair_std
    return pair_std_lst
    
    
def match_color(loc_pred, key, colors, colors_ref = 'CN_true'):
    """
    Match cluster colors to a reference set of cluster labels using the Hungarian algorithm.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame containing cluster assignments for observations.
    key : str
        Column in `loc_pred` for which colors need to be matched.
    colors : dict
        Color mapping for the reference clusters (keys = reference cluster labels).
    colors_ref : str, optional
        Column in `loc_pred` to use as reference for matching (default is 'CN_true').

    Returns
    -------
    dict
        Dictionary mapping the clusters in `key` to colors based on the best match
        to the reference clusters.
    """
    matrix = pd.crosstab(loc_pred[colors_ref], loc_pred[key])
    cluster_uniq = matrix.index
    cost_matrix = -matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    new_col = { cluster_uniq[col_indices[i]]: colors[cluster_uniq[i]] for i in range(len(col_indices))}
    return new_col
    
def squareform_index(n, i, j):
    """Return index in condensed distance vector for pair (i,j)."""
    if i == j:
        raise ValueError("No diagonal in condensed matrix")
    if i > j:
        i, j = j, i
    return n*i - i*(i+1)//2 + (j - i - 1)


def match_clusters(loc_pred, key_ref, key_new):
    """
    Compute cluster correspondence between set 1 and set 2 
    using the Hungarian algorithm.

    Parameters
    ----------
    set1_labels : array-like
        Cluster assignments for set 1 (reference).
    set2_labels : array-like
        Cluster assignments for set 2 (to be matched).

    Returns
    -------
    correspondence : dict
        Mapping {cluster_id_in_set2: cluster_id_in_set1}
    """
    set1_labels = np.array(loc_pred[key_ref].values)
    set2_labels = np.array(loc_pred[key_new].values)
    matrix = pd.crosstab(set1_labels, set2_labels)
    clusters1 = matrix.index.to_numpy()
    clusters2 = matrix.columns.to_numpy()
    cost_matrix = -matrix.values
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    correspondence = {clusters2[c]: clusters1[r] for r, c in zip(row_ind, col_ind)}
    return correspondence
        
    
def assign_loc(loc_mean, col_true, col_new):
    color_to_old_index = {v: k for k, v in col_true.items()}
    new_loc_mean = pd.DataFrame(columns=loc_mean.columns)
    for new_idx, color in col_new.items():
        old_idx = color_to_old_index[color]
        new_loc_mean.loc[new_idx] = loc_mean.loc[old_idx]
    new_loc_mean = new_loc_mean.sort_index(key=lambda x: x.astype(int))
    return new_loc_mean

def cn_cluster(loc_pred, loc_matrix, ct_key = "Cluster", n_clust=10, knn=100, random_state=1):
    """
    Assign clusters to nodes (e.g., cell types) based on spatial coordinates or pairwise distance matrix.

    This function decides whether to cluster using a **distance matrix** or a
    **coordinate matrix** and calls the appropriate clustering function.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame containing cluster assignments for each observation.
    loc_matrix : np.ndarray
        Either a square distance matrix (n_samples x n_samples) or a coordinate matrix
        (n_samples x 2 or n_samples x 3) representing spatial locations.
    ct_key : str, optional
        Column in `loc_pred` containing initial cluster labels. Default is "Cluster".
    n_clust : int, optional
        Number of clusters to generate. Default is 10.
    knn : int, optional
        Number of nearest neighbors to consider in distance-based clustering. Default is 100.

    Returns
    -------
    np.ndarray
        Array of new cluster assignments (as strings) for each node.
    """
    if loc_matrix.shape[0] == loc_matrix.shape[1]:
        new_clusters = cn_cluster_dist(loc_pred, loc_matrix, ct_key=ct_key, n_clust=n_clust, knn=knn, random_state=random_state)
    elif loc_matrix.shape[1] in [2, 3]:
        new_clusters = cn_cluster_loc(loc_pred, loc_matrix, ct_key=ct_key, n_clust=n_clust, knn=knn, random_state=random_state)
    else:
        raise ValueError("loc_matrix must be either a distance matrix or a coordinate matrix with 2/3 columns.")
    return new_clusters

def cn_network_loc(loc_pred, location, cn_key, colors, thre = 0.75, title = "", coef = 1000, lw_coef=0.5, save_path = None, filename = "CN_network"):
    """
    For single slice training setting, generate a CN network based on node locations.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame containing cluster assignments for each observation.
    location : np.ndarray
        Coordinates of each observation (n_samples x 2 or x3).
    cn_key : str
        Column name in `loc_pred` containing cluster labels.
    colors : dict or pd.Series
        Color mapping for each cluster.
    thre : float, optional
        Quantile threshold (0-1) to filter edges; default is 0.75.
    title : str, optional
        Title for the plot.
    coef : float, optional
        Scaling factor for node sizes; default is 1000.
    lw_coef : float, optional
        Exponent for edge linewidth scaling; default is 0.5.
    """
    location = (location - np.min(location, axis=0)) / (np.max(location, axis=0) - np.min(location, axis=0))
    loc_pred[cn_key] = loc_pred[cn_key].astype('category').astype(str)
    dist_node, loc_mean = dist_center(loc_pred, location, key=cn_key)
    sd_node = node_var(loc_pred, location, key=cn_key)
    draw_graph_loc(loc_mean, dist_node, thre, sd_node, colors, coef = coef, lw_coef=lw_coef, title=title, save_path = save_path, filename = filename)
    
    
    
def cn_spatial_proximity(loc_pred, dist_matrix, key, k=2000, r=None):
    """
    Compute CN spatial proximity by computing the fraction of neighbors from each CN cluster.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame with a column `key` giving cluster labels for each cell.
    dist_matrix : np.ndarray
        Pairwise distance matrix (n x n), may contain np.inf.
    key : str
        Column name in loc_pred for cluster labels.
    k : int
        Number of nearest neighbors to consider (ignored if r is given).
    r : float or None
        Distance threshold; if provided, consider neighbors within distance <= r.

    Returns
    -------
    pd.DataFrame
        DataFrame where entry (i, j) = average fraction of j-neighbors 
        for cells in cluster i, considering *only cross-cluster* neighbors.
    """
    from tqdm import tqdm
    clusters = loc_pred[key].values
    uniq = np.sort(np.unique(clusters))
    n_clusters = len(uniq)
    proximity = np.full((n_clusters, n_clusters), np.nan)

    for i, ci in tqdm(enumerate(uniq)):
        i_idx = np.where(clusters == ci)[0]
        frac_mat = np.zeros((len(i_idx), n_clusters))

        for idx_pos, u in enumerate(i_idx):
            dists = dist_matrix[u, :].copy()
            dists[u] = np.inf  # exclude self
            finite_mask = np.isfinite(dists)

            if finite_mask.sum() == 0:
                continue

            if r is not None:
                neighbor_idx = np.where((finite_mask) & (dists <= r))[0]
            else:
                k_eff = min(k, finite_mask.sum())
                neighbor_idx = np.argpartition(dists[finite_mask], k_eff - 1)[:k_eff]
                neighbor_idx = np.where(finite_mask)[0][neighbor_idx]

            if len(neighbor_idx) == 0:
                continue

            neighbor_clusters = clusters[neighbor_idx]
            for j, cj in enumerate(uniq):
                frac_mat[idx_pos, j] = np.mean(neighbor_clusters == cj)

        proximity[i, :] = np.nanmean(frac_mat, axis=0)
    return pd.DataFrame(proximity, index=uniq, columns=uniq)