import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy
from sklearn.metrics import pairwise_distances

def distCompute(Qdata, key_true = "loc_true", key_pred = "loc_pred"):
    """
    Compute pairwise Euclidean distances between all points in true vs. predicted locations.

    Args:
        Qdata (AnnData or dict-like): Object containing location arrays in `.obsm`.
        key_true (str, optional): Key for the ground-truth locations in Qdata.obsm.
        key_pred (str, optional): Key for the predicted locations in Qdata.obsm.

    Returns:
        tuple:
            true_dist (np.ndarray, float32) Flattened array of pairwise distances 
                                              between ground-truth points.
            pred_dist (np.ndarray, float32) Flattened array of pairwise distances 
                                              between predicted points.
    """
    pred_dist = []
    true_dist = []
    Qdata_loc = np.array(Qdata.obsm[key_true])
    pred_loc = np.array(Qdata.obsm[key_pred])

    for i in tqdm(range(Qdata_loc.shape[0])):
        pred_i = pred_loc[i, :]
        pred_points = pred_loc[i+1:, :]
        pred_dist.extend(np.sqrt(np.sum((pred_points - pred_i)**2, axis=1)))

        true_i = Qdata_loc[i, :]
        true_points = Qdata_loc[i+1:, :]
        true_dist.extend(np.sqrt(np.sum((true_points - true_i)**2, axis=1)))
    true_dist = np.float32(true_dist)
    pred_dist = np.float32(pred_dist)
    return true_dist, pred_dist
        
def pairwise_corr(Qdata, key1 = 'loc_true', key2 = 'loc_pred'):
    """
    Compute pairwise distance correlation.

    Args:
        Qdata (AnnData or dict-like): Object containing location arrays in `.obsm`.
        key_true (str, optional): Key for the ground-truth locations in Qdata.obsm.
        key_pred (str, optional): Key for the predicted locations in Qdata.obsm.

    """
    Qdata_loc = np.array(Qdata.obsm[key1])
    pred_loc = np.array(Qdata.obsm[key2])
    Qdata_loc = (Qdata_loc - np.min(Qdata_loc, axis=0)) / (np.max(Qdata_loc, axis=0) - np.min(Qdata_loc, axis=0))
    pred_loc = (pred_loc - np.min(pred_loc, axis=0)) / (np.max(pred_loc, axis=0) - np.min(pred_loc, axis=0))         
    Qdata.obsm[f'{key1}_norm'] = Qdata_loc
    Qdata.obsm[f'{key2}_norm'] = pred_loc
    true_dist, pred_dist = distCompute(Qdata, key_true=f'{key1}_norm', key_pred=f'{key2}_norm')
    corr = scipy.stats.pearsonr(true_dist, pred_dist)
    print(corr.statistic)
    
def scatter_plot(Qdata, colors, spatial_key, cluster_key = "Cluster", title = "", s = 1, save_path = None, filename = "remap_pred"):
    """
    Create a 2D scatter plot of spatial transcriptomics data.

    Args:
        Qdata (AnnData): AnnData object containing `.obs` (metadata) and `.obsm` (embeddings).
        colors (list or dict): Color palette for cluster labels.
        spatial_key (str): Key in `Qdata.obsm` with spatial coordinates (e.g., ["x", "y"]).
        cluster_key (str, optional): Column name in `Qdata.obs` for cluster labels. Default is "Cluster".
        title (str, optional): Plot title. Default is "".
        s (float, optional): Point size for scatter plot. Default is 1.
    """
    try:
        cluster_uniq = np.sort(np.unique(Qdata.obs[cluster_key].astype(int))).astype(str)
    except:
        cluster_uniq = np.sort(np.unique(Qdata.obs[cluster_key]))
    Qdata.obs[['x_loc', 'y_loc']] = np.array(Qdata.obsm[spatial_key])
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(data=Qdata.obs, x='x_loc', y='y_loc', s=s, hue=cluster_key, hue_order=cluster_uniq,
                    palette=colors, ax=axes, legend=False).set(title=title, xlabel=None, ylabel=None)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_linewidth(1.5)  
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    

def pair_corr_rel(loc_true, dist_pred):
    """
    Compute pairwise distance correlation.

    Args:
        loc_true (str, optional): True locations.
        dist_pred (str, optional): Predicted pairwise distances.
    """
    loc_true = np.array(loc_true)
    loc_true = (loc_true - np.min(loc_true, axis=0)) / (np.max(loc_true, axis=0) - np.min(loc_true, axis=0))
    dist_true = pairwise_distances(np.array(loc_true), metric='euclidean')
    dist_pred = dist_pred[np.tril_indices(dist_pred.shape[0], k=-1)]
    dist_true = dist_true[np.tril_indices(dist_true.shape[0], k=-1)]
    corr = scipy.stats.pearsonr(dist_true, dist_pred)
    print(corr.statistic)
    return corr.statistic