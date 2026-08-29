import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
from scipy.sparse import issparse


def _cluster_codes(clusters, uniq):
    """
    Column index of every cell's cluster. Labels matching no entry of `uniq` are
    sent to a spare bin that the caller drops, which keeps the counting loop
    branch-free.
    """
    codes = np.full(len(clusters), len(uniq), dtype=np.intp)
    for j, cj in enumerate(uniq):
        codes[clusters == cj] = j
    return codes


def _keep_neighbors(dists, k, r, all_finite):
    """
    Positions of the neighbors kept for one cell.

    `dists` holds the candidate distances, +inf wherever the slot is not a real
    neighbor -- absent, non-finite, or the cell itself. With `r`, everything
    within the radius; otherwise the `k` smallest, which degrades on its own to
    "all of them" for a cell with fewer than `k` neighbors. `all_finite` skips the
    per-cell finiteness test when the caller has already ruled it out.
    """
    if r is not None:
        return np.flatnonzero(dists <= r)            # +inf and nan are never <= r
    if dists.size <= k:
        return np.arange(dists.size) if all_finite else np.flatnonzero(np.isfinite(dists))
    pos = np.argpartition(dists, k - 1)[:k]
    return pos if all_finite else pos[np.isfinite(dists[pos])]


def _neighbor_fractions(dist_matrix, codes, n_clusters, k, r):
    """
    Fraction of each cluster among every cell's neighbors, one row per cell.

    A sparse matrix is read straight from its CSR buffers -- the stored entries
    of a row are exactly its finite distances -- so no dense n x n matrix is ever
    built. For 50k cells that alone would be over 10 GB.
    """
    n = dist_matrix.shape[0]
    frac = np.zeros((n, n_clusters))
    sparse = issparse(dist_matrix)

    if sparse:
        M = dist_matrix.tocsr()
        if M.nnz and (M.data == 0).any():
            # A stored zero means "not a neighbor" here, never "distance 0".
            M = M.copy()
            M.eliminate_zeros()
        indptr, indices, data = M.indptr, M.indices, M.data
        # Both settled once for the whole matrix instead of row by row. A REMAP
        # prediction stores neither the diagonal nor any non-finite distance, so
        # the loop below normally skips both tests.
        stores_self = bool(M.diagonal().any())
        all_finite = not stores_self and bool(np.isfinite(data).all())
    else:
        all_finite = False

    for u in range(n):
        if sparse:
            lo, hi = indptr[u], indptr[u + 1]
            cand, dists = indices[lo:hi], data[lo:hi]
            if stores_self:
                dists = np.where(cand == u, np.inf, dists)
        else:
            dists = np.array(dist_matrix[u], dtype=float)
            dists[u] = np.inf                        # exclude self
        pos = _keep_neighbors(dists, k, r, all_finite)
        if pos.size == 0:                            # no neighbor: leave the row at 0
            continue
        nbr = cand[pos] if sparse else pos
        frac[u] = np.bincount(codes[nbr], minlength=n_clusters + 1)[:n_clusters] / pos.size
    return frac


def cluster_neighbor_fraction(loc_pred, dist_matrix, cn_key, k_neighbor=3000, r=None):
    """
    Compute cross-cluster spatial proximity by averaging, over the cells of each
    cluster, the fraction of their neighbors belonging to every cluster.

    Parameters
    ----------
    loc_pred : pd.DataFrame
        DataFrame with a column `cn_key` giving cluster labels for each cell.
    dist_matrix : np.ndarray or scipy.sparse matrix
        Pairwise distance matrix (n x n), may contain np.inf. A sparse matrix is
        read directly, as saved by `Fit_cord_multi`: a stored entry is a predicted
        neighbor distance and everything else counts as infinite, which is what
        the dense `dist[dist == 0] = np.inf` matrix expressed. No densification
        is needed.
    cn_key : str
        Column name in loc_pred for cluster labels.
    k_neighbor : int
        Number of nearest neighbors to consider (ignored if r is given). A cell
        with fewer than `k` neighbors uses all of them.
    r : float or None
        Distance threshold; if provided, consider neighbors within distance <= r.

    Returns
    -------
    pd.DataFrame
        DataFrame where entry (i, j) = average fraction of j-neighbors for cells
        in cluster i. Row i includes cluster i's own share on the diagonal, so
        each row is a composition over all clusters and sums to 1.
    """
    clusters = loc_pred[cn_key].values
    uniq = np.sort(np.unique(clusters))
    codes = _cluster_codes(clusters, uniq)
    frac = _neighbor_fractions(dist_matrix, codes, len(uniq), k_neighbor, r)
    proximity = np.vstack([frac[codes == i].mean(axis=0) for i in range(len(uniq))])
    return pd.DataFrame(proximity, index=uniq, columns=uniq)


def get_block_num_affinity(img, n_clust=5, method='kmeans'):
    """
    Group the clusters of a proximity matrix into `n_clust` blocks by clustering
    its rows, i.e. by how similar their neighborhood compositions are.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clust, random_state=42)
    elif method == 'minik':
        clustering = MiniBatchKMeans(n_clusters=n_clust, random_state=42)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clust, metric='euclidean',
                                             linkage='average')
    return clustering.fit_predict(img.values)


def get_blocks_plot(labels, order=None):
    """
    Order the clusters so that block members sit together, and return the
    (start, end) span each block occupies in that order.

    `order` supplies a grouping order to use instead of sorting by label value,
    for callers that have already decided how the blocks should be arranged.
    """
    idx_sorted = np.argsort(labels, kind='stable') if order is None else np.asarray(order)
    edges = np.flatnonzero(np.diff(labels[idx_sorted])) + 1
    bounds = np.concatenate(([0], edges, [len(labels)]))
    return idx_sorted, list(zip(bounds[:-1], bounds[1:]))


def _similarity_order(img):
    """
    A leaf order of the clusters in which proximate ones sit next to each other.

    The proximity matrix is very nearly symmetric but not exactly, so it is
    symmetrized and turned into a dissimilarity before the tree is built. Used
    only to order rows for display -- it never changes block membership.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    sym = (img.values + img.values.T) / 2
    dissim = 1 - sym / max(sym.max(), 1e-12)
    np.fill_diagonal(dissim, 0)
    condensed = squareform(dissim, checks=False)
    if condensed.size == 0:
        return np.arange(len(img))
    tree = optimal_leaf_ordering(linkage(condensed, method='average'), condensed)
    return leaves_list(tree)


def compute_block(img, n_clust, method='kmeans'):
    """
    Split a cluster-proximity matrix into blocks and reorder it for plotting.

    Parameters
    ----------
    img : pd.DataFrame
        Square proximity matrix, as returned by `cluster_neighbor_fraction`.
    n_clust : int
        Number of blocks to split the clusters into.
    method : {'kmeans', 'minik', 'agglomerative'}
        Algorithm used to group the clusters.

    Returns
    -------
    block_label : pd.DataFrame
        One row per cluster, with the block it was assigned to.
    img_reorder : pd.DataFrame
        `img` with rows and columns reordered so that each block is contiguous.
        Blocks, and the clusters inside them, are ordered by similarity, which
        makes the block structure read down the diagonal.
    blocks : list of (int, int)
        Half-open (start, end) span of every block in the reordered matrix.
    """
    # Cluster on a canonical row order so the blocks depend only on the matrix,
    # not on the order its rows happen to arrive in: KMeans seeding is order
    # sensitive, and the same matrix can reach here row-permuted.
    canon = np.argsort(np.asarray(img.index, dtype=str), kind='stable')
    labels = np.empty(len(img), dtype=int)
    labels[canon] = get_block_num_affinity(img.iloc[canon, canon], n_clust=n_clust,
                                           method=method)

    # Rank blocks, and their members, along a similarity ordering so the diagonal
    # reads as a gradient instead of following arbitrary KMeans label numbers.
    rank = np.empty(len(labels), dtype=int)
    rank[_similarity_order(img)] = np.arange(len(labels))
    block_rank = {b: rank[labels == b].mean() for b in np.unique(labels)}
    order = np.lexsort((rank, [block_rank[b] for b in labels]))

    idx_sorted, blocks = get_blocks_plot(labels, order=order)

    img_reorder = img.iloc[idx_sorted, idx_sorted]
    block_label = pd.DataFrame({"cluster": img.index.to_list(), "label": labels})
    return block_label, img_reorder, blocks


def _color_strip(ax, groups, colors, orient):
    """Draw the per-cluster color key as a strip aligned cell-for-cell with the map."""
    rgb = np.array([mcolors.to_rgb(colors[g]) for g in groups])
    ax.imshow(rgb[None, :, :] if orient == 'h' else rgb[:, None, :],
              aspect='auto', interpolation='nearest',
              extent=((-0.5, len(groups) - 0.5, 0.5, -0.5) if orient == 'h'
                      else (-0.5, 0.5, len(groups) - 0.5, -0.5)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which='both', length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def cn_heatmap_draw(img_reorder, colors, save_path, filename, blocks=None,
                    mask_diagonal=False, cmap='Blues', vmax=None, title=None,
                    cell=0.36, dpi=300):
    """
    Draw a cluster-proximity heatmap with a color key and block structure.

    Parameters
    ----------
    img_reorder : pd.DataFrame
        Square proximity matrix, already reordered by `compute_block`.
    colors : dict or pd.Series
        Color for each cluster label.
    save_path, filename : str
        The figure is written to `{save_path}/{filename}.pdf`.
    cn_num : int, optional
        Ignored; the number of clusters is taken from `img_reorder`. Kept so
        existing calls keep working -- passing a value smaller than the matrix
        used to silently drop color chips.
    blocks : list of (int, int), optional
        Block spans from `compute_block`. Drawn as separators and outlines.
    mask_diagonal : bool
        Leave out the diagonal, which is each cluster's own share of its
        neighborhood. It is always the largest entry by far, so keeping it sets
        the color scale and flattens every cross-cluster value to near-white.
        Set False to show it.
    cmap : str
        Matplotlib colormap.
    vmax : float, optional
        Upper end of the color scale. Defaults to the largest plotted value.
    labels : bool
        Write the cluster names alongside the color strips.
    cell : float
        Size of one matrix cell in inches; the figure scales with the matrix.
    dpi : int
        Raster resolution for the saved figure.
    """
    os.makedirs(save_path, exist_ok=True)

    groups = list(img_reorder.index)
    n = len(groups)
    values = np.array(img_reorder, dtype=float)
    if mask_diagonal:
        values[np.diag_indices(n)] = np.nan
    if vmax is None:
        vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0

    # Everything is laid out in cell units, with the gaps as real grid columns, so
    # the map stays square and the strips line up cell for cell at any size.
    strip, pad, bar = float(np.clip(0.05 * n, 0.4, 0.8)), 0.35, 0.28
    cols = [strip, pad, n, pad, bar]
    rows = [n, pad, strip]

    fig = plt.figure(figsize=(sum(cols) * cell, sum(rows) * cell))
    grid = fig.add_gridspec(3, 5, width_ratios=cols, height_ratios=rows,
                            wspace=0, hspace=0,
                            left=0.005, right=0.995, top=0.995, bottom=0.005)
    # Deliberately not shared axes: sharing would tie the tick locators together
    # and leak the strip labels onto the map. The grid geometry aligns them.
    ax = fig.add_subplot(grid[0, 2])
    ax_row = fig.add_subplot(grid[0, 0])
    ax_col = fig.add_subplot(grid[2, 2])
    ax_bar = fig.add_subplot(grid[0, 4].subgridspec(3, 1, height_ratios=[1, 2, 1])[1])

    palette = plt.get_cmap(cmap).copy()
    palette.set_bad('#f2f2f2')           # masked diagonal, not "value zero"
    mesh = ax.imshow(values, cmap=palette, vmin=0, vmax=vmax,
                     interpolation='nearest', aspect='auto')

    # Hairline gaps between cells, so equal neighbors stay countable.
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='both', length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    for side in ax.spines.values():
        side.set_color('#c8c8c8')
        side.set_linewidth(0.8)

    if blocks:
        for start, end in blocks:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), end - start, end - start,
                                       fill=False, edgecolor='#1a1a1a', linewidth=1.4,
                                       zorder=3))
        for start, _ in blocks[1:]:      # separators across the full matrix
            ax.axvline(start - 0.5, color='white', linewidth=2.2, zorder=2)
            ax.axhline(start - 0.5, color='white', linewidth=2.2, zorder=2)

    _color_strip(ax_row, groups, colors, 'v')
    _color_strip(ax_col, groups, colors, 'h')
    if title is not None:
        ax.set_title(title, fontsize=15, pad=2)

    cbar = fig.colorbar(mesh, cax=ax_bar)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, width=0.6, labelsize=7, pad=2)
    cbar.set_ticks([0, vmax])
    cbar.set_ticklabels(["Min", "Max"])
    cbar.set_label("Spatial proximity", fontsize=7, labelpad=3)

    fig.savefig(f"{save_path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight')
    # plt.close(fig)
