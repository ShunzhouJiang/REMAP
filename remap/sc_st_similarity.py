import pandas as pd
import numpy as np
import os
from anndata import concat
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
import scanpy as sc

def sample_group(group, key = 'Cluster', sample_num = 5000):
    group_indices = group.index
    group_class = group[key]
    class_count = group_class.value_counts()
    class_keep = class_count[class_count > 1].index.tolist()
    group = group[group[key].isin(class_keep)].copy()
    group_indices = group.index
    group_class = group[key]
    if sample_num >= len(group_indices):
        return group_indices.tolist()
    group_sampled, _ = train_test_split(group_indices, train_size=min(sample_num, len(group_indices)), stratify=group_class, random_state=1)
    return group_sampled

def compute_fractions(row, source_uniq):
    counts = row.value_counts(normalize=True)  # Normalize counts to get fractions
    return pd.Series([counts.get(i, 0) for i in source_uniq])  # Fraction for categories 1, 2, 3, 4

def st_frac(adata, test_name, path, source_key='source', sample_key = 'Cluster', sample_num = 3000, norm = True):
    adata.obs_names_make_unique()
    adata_i = adata[adata.obs[source_key] == test_name].copy()
    adata_no = adata[adata.obs[source_key] != test_name].copy()
    
    val_table = adata_no.obs[source_key].value_counts()
    if sample_num is None or sample_num > val_table.min():
        sample_num = val_table.min()
    sampled_indices = adata_no.obs.groupby(source_key).apply(lambda group: sample_group(group, key=sample_key, sample_num=sample_num))
    sampled_indices = sampled_indices.explode().values
    adata_sampled = adata_no[sampled_indices].copy()
    
    adata_sampled_tot = concat([adata_i, adata_sampled], axis = 0)

    if norm:
        sc.pp.normalize_per_cell(adata_sampled_tot, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_sampled_tot)
    sc.pp.pca(adata_sampled_tot, n_comps=50)
    sc.pp.neighbors(adata_sampled_tot, n_pcs=50, use_rep='X_pca', metric='euclidean')
    
    adata_i = adata_sampled_tot[adata_sampled_tot.obs[source_key] == test_name].copy()
    adata_no = adata_sampled_tot[adata_sampled_tot.obs[source_key] != test_name].copy()

    kdB = BallTree(adata_no.obsm['X_pca'], metric="euclidean")      ## reference
    knn_index = kdB.query(adata_i.obsm['X_pca'], k=int(sample_num * 0.1))[1] 
    sample_ind= np.array(adata_no.obs[source_key].values[knn_index])    
    
    source_uniq = np.sort(np.unique(adata_sampled.obs[source_key]))
    source_uniq = source_uniq[source_uniq != test_name]
    weights = pd.DataFrame(sample_ind).apply(lambda row: compute_fractions(row, source_uniq=source_uniq), axis = 1)
    weights.columns = source_uniq
    weights_mean = np.mean(weights, axis = 0)
    weights_mean.index = source_uniq
    return weights_mean, weights