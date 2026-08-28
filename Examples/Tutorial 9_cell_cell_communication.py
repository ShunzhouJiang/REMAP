"""
Performed SpatialDM (https://www.nature.com/articles/s41467-023-39608-w) cell-cell communication on the mouse brain data.

Runs the same analysis twice -- once on the true spatial coordinates, once on the
REMAP-predicted ones -- and compares them. Three stages, each reading and writing
under RESULT_DIR:

    run_spatialdm()   global + local SpatialDM per method, writes the result tables
                      and a slim h5ad. This is the expensive stage (hours).
    plot_lr_maps()    spatial map of the selected spots, one panel per LR pair

The stages run from main(); comment out the ones you do not need. Stages 2 and 3
only need the files stage 1 wrote, so they can be rerun on their own.
"""

import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdm as sdm
from scipy import sparse
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
THRE = 0.05                 # an LR pair must be expressed in this fraction of cells
RESULT_DIR = f"result_local/brain_hd_remap_test"
PREFIX = "CCC"              # legacy filename prefix, kept so existing results load

SC_DATA_FULL = "../REMAP_final/data/brain_hd/sc_data.h5ad"
REMAP_LOC = "../REMAP_final/remap_output/brain_hd/remap_loc.npy"

METHODS = ("true", "remap")     # obsm keys holding the two coordinate sets


def result_csv(kind, name):
    return f"{RESULT_DIR}/{PREFIX}_{kind}_{name}.csv"


def result_h5ad(name):
    return f"{RESULT_DIR}/{PREFIX}_{name}.h5ad"


def minmax(loc):
    """Scale coordinates into the unit square."""
    return (loc - np.min(loc, axis=0)) / (np.max(loc, axis=0) - np.min(loc, axis=0))


# --------------------------------------------------------------------------- #
# stage 1: run SpatialDM
# --------------------------------------------------------------------------- #
def load_query():
    """
    Half of the cells, gene-filtered and log-normalised, carrying both coordinate
    sets in obsm so the two runs differ only in which one they use.
    """
    Qdata = ad.read_h5ad(SC_DATA_FULL)
    Qdata.obsm["true"] = np.array(Qdata.obs[["x_cord", "y_cord"]])
    Qdata.obsm["remap"] = np.load(REMAP_LOC)

    _, keep, _, _ = train_test_split(range(Qdata.shape[0]), Qdata.obs["Cluster"],
                                     test_size=0.5, random_state=1,
                                     stratify=Qdata.obs["Cluster"])
    Qdata = Qdata[np.sort(keep), :].copy()
    print("after subsampling:", Qdata.shape)

    Qdata.raw = ad.AnnData(Qdata.X, obs=Qdata.obs, var=Qdata.var)
    sc.pp.filter_genes(Qdata, min_cells=3000)
    sc.pp.normalize_per_cell(Qdata, counts_per_cell_after=10000)
    sc.pp.log1p(Qdata)
    print("after gene filtering:", Qdata.shape)

    sdm.extract_lr(Qdata, "mouse", min_cell=int(Qdata.shape[0] * THRE))
    Qdata.X = Qdata.X.A
    return Qdata


def ccc_local(Qdata, loc, name):
    """Global and local SpatialDM for one coordinate set; writes the tables and a slim h5ad."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    Qdata.obsm["spatial"] = minmax(loc)

    sdm.weight_matrix(Qdata, l=0.05, cutoff=0.75, n_neighbors=500, single_cell=True)
    sdm.spatialdm_global(Qdata, 1000, specified_ind=None, method="z-score", nproc=4)
    sdm.sig_pairs(Qdata, method="z-score", fdr=True, threshold=0.1)
    Qdata.uns["global_res"].to_csv(result_csv("global", name))

    sdm.spatialdm_local(Qdata, n_perm=1000, method="z-score", specified_ind=None, nproc=4)
    sdm.sig_spots(Qdata, method="z-score", fdr=False, threshold=0.1)
    Qdata.uns["local_z_p"].to_csv(result_csv("local_z_p", name))
    Qdata.uns["selected_spots"].to_csv(result_csv("selected_spots", name))

    # The big tables are on disk as csv now, so keep only one gene in the h5ad.
    del (Qdata.uns["global_I"], Qdata.uns["global_stat"],
         Qdata.uns["local_z_p"], Qdata.uns["selected_spots"])
    Qdata.X = sparse.csr_matrix(Qdata.X)
    sdm.write_spatialdm_h5ad(Qdata[:, :1], result_h5ad(name))


def run_spatialdm():
    Qdata = load_query()
    for method in METHODS:
        print(f"--- SpatialDM on {method} coordinates ---")
        ccc_local(Qdata.copy(), loc=Qdata.obsm[method], name=method)


# --------------------------------------------------------------------------- #
# stage 2 and 3 shared: reload results
# --------------------------------------------------------------------------- #
def load_result(name, spatial=None):
    """
    Re-attach the saved SpatialDM tables to the log-normalised expression.

    The slim h5ad written by stage 1 keeps one gene, so expression is reloaded
    from the full dataset and subset to the cells that were analysed. `spatial`
    overrides the coordinates stored in the h5ad.
    """
    adata = ad.read_h5ad(result_h5ad(name))
    adata.uns["local_z_p"] = pd.read_csv(result_csv("local_z_p", name), index_col=0)
    adata.uns["selected_spots"] = pd.read_csv(result_csv("selected_spots", name), index_col=0)
    adata.uns["receptor"].replace("NA", np.nan, inplace=True)
    adata.uns["ligand"].replace("NA", np.nan, inplace=True)

    expr = ad.read_h5ad(SC_DATA_FULL)
    expr.X = expr.X.astype(np.float32)
    if spatial is not None:
        expr.obsm["spatial"] = minmax(spatial)
    expr = expr[adata.obs.index, :]
    sc.pp.normalize_per_cell(expr, counts_per_cell_after=10000)
    sc.pp.log1p(expr)

    expr.obs = adata.obs
    if spatial is None:
        expr.obsm = adata.obsm
    expr.uns = adata.uns
    expr.obsp = adata.obsp
    return expr


def load_results():
    """Both methods, ready to plot."""
    true = load_result("true")
    true.obsm["spatial"] = 1 - true.obsm["spatial"]      # match the tissue orientation
    remap = load_result("remap", spatial=np.load(REMAP_LOC, allow_pickle=True))
    return {"true": true, "remap": remap}


def shared_lr_pairs(results):
    """LR pairs called by every method, so the panels are comparable."""
    called = [set(adata.uns["local_z_p"].index) for adata in results.values()]
    return np.sort(list(set.intersection(*called)))


# --------------------------------------------------------------------------- #
# stage 2: per-LR-pair spatial maps
# --------------------------------------------------------------------------- #
def plot_lr(adata, pair, out_dir, method):
    """Spatial map of one LR pair, shaded by 1 - local p-value."""
    score = 1 - adata.uns["local_z_p"].loc[pair]
    loc = adata.obsm["spatial"]

    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=loc[:, 0], y=loc[:, 1], hue=score, palette="Greens",
                    size=1, sizes=(1, 1), legend=False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"{out_dir}/{pair}_{method}.png", format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_lr_maps(results, pairs, subdir="lr_plot_own_new"):
    out_dir = f"{RESULT_DIR}/{subdir}"
    os.makedirs(out_dir, exist_ok=True)
    for pair in pairs:
        for method, adata in results.items():
            plot_lr(adata, pair, out_dir, method)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def main():
    # WARNING: this rewrites crc_true.h5ad / crc_remap.h5ad in RESULT_DIR. The
    # existing crc_true.h5ad there is from Apr 2025 and takes hours to regenerate --
    # comment this line out to plot from the results already on disk.
    run_spatialdm()

    results = load_results()
    lr_select = shared_lr_pairs(results)
    print(f"{len(lr_select)} LR pairs called by both methods")

    plot_lr_maps(results, lr_select)


if __name__ == "__main__":
    main()
