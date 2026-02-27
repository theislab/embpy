"""Visualization functions for perturbation embeddings.

Provides UMAP / t-SNE scatter plots, similarity and distance heatmaps,
correlation matrices, KNN-overlap panels, hierarchical-clustering
dendrograms, embedding distribution plots, and Leiden-clustering
overview figures.

All public functions accept an ``AnnData`` whose ``.obsm`` contains one
or more embedding matrices.  Unless a specific key is given, the helper
:func:`_get_embedding_keys` discovers all suitable keys automatically.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from embpy import tl

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = ("X_umap", "X_tsne", "X_pca", "X_diffmap", "X_draw_graph")


def _get_embedding_keys(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
) -> list[str]:
    """Discover embedding keys in *adata.obsm*.

    Selects 2-D numeric arrays and excludes common reduced-coordinate keys
    (``X_umap``, ``X_tsne``, …) unless they are explicitly requested.
    """
    if obsm_keys is not None:
        for k in obsm_keys:
            if k not in adata.obsm:
                raise KeyError(f"'{k}' not in adata.obsm. Available: {list(adata.obsm.keys())}")
        return list(obsm_keys)

    keys: list[str] = []
    for k in adata.obsm:
        arr = adata.obsm[k]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2:
            continue
        if any(k.startswith(p) for p in _SKIP_PREFIXES):
            continue
        keys.append(k)
    return keys


def _labels_from_adata(adata: AnnData) -> list[str] | None:
    """Try to extract short observation labels from common columns."""
    for col in ("identifier", "drug_id", "gene_symbol"):
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).tolist()
    return list(adata.obs_names)


# ---------------------------------------------------------------------------
# 2-D embedding scatter
# ---------------------------------------------------------------------------

def plot_embedding_space(
    adata: AnnData,
    obsm_key: str | None = None,
    color: str | None = None,
    method: str = "umap",
    basis: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Plot perturbation embeddings in a 2-D reduced space.

    If *basis* is not provided, UMAP or t-SNE coordinates are computed
    automatically from *obsm_key* using the corresponding ``tl`` function.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Key in ``.obsm`` with the high-dimensional embedding.  Used to
        derive 2-D coordinates when *basis* is ``None``.
        Defaults to the first discovered key.
    color
        Column in ``adata.obs`` to color points by.
    method
        Reduction method when *basis* is ``None``: ``"umap"`` or ``"tsne"``.
    basis
        Explicit key in ``.obsm`` for 2-D coordinates.  Overrides *method*.
    title
        Plot title.
    figsize
        Figure size as ``(width, height)``.
    ax
        Optional matplotlib ``Axes`` to draw into.
    **kwargs
        Passed to ``matplotlib.axes.Axes.scatter``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]

    if basis is not None:
        coords_key = basis
    elif method == "umap":
        coords_key = f"X_umap_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_umap(adata, obsm_key=obsm_key)
    elif method == "tsne":
        coords_key = f"X_tsne_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_tsne(adata, obsm_key=obsm_key)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'umap' or 'tsne'.")

    if coords_key not in adata.obsm:
        raise KeyError(f"'{coords_key}' not found in adata.obsm.")
    coords = np.asarray(adata.obsm[coords_key])

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter_kw: dict[str, Any] = {"s": 40, "alpha": 0.8, "edgecolors": "k", "linewidth": 0.3}
    scatter_kw.update(kwargs)

    if color and color in adata.obs.columns:
        cats = adata.obs[color]
        if cats.dtype.name == "category" or cats.nunique() <= 20:
            cats = cats.astype("category")
            palette = sns.color_palette("husl", n_colors=cats.cat.categories.size)
            color_map = {c: palette[i] for i, c in enumerate(cats.cat.categories)}
            c_vals = [color_map[v] for v in cats]
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=c_vals, **scatter_kw)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                           markersize=8, label=str(c))
                for c in cats.cat.categories
            ]
            ax.legend(handles=handles, title=color, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        else:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cats.values, cmap="viridis", **scatter_kw)
            plt.colorbar(scatter, ax=ax, shrink=0.8, label=color)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kw)

    method_name = method.upper() if basis is None else basis
    ax.set_xlabel(f"{method_name}1")
    ax.set_ylabel(f"{method_name}2")
    ax.set_title(title or f"{obsm_key} ({method_name})")

    if created_fig:
        fig.tight_layout()
    return fig


def all_embeddings(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
    method: str = "umap",
    color: str | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5, 4),
) -> Figure:
    """Grid of 2-D scatter plots, one per embedding key.

    Parameters
    ----------
    adata
        AnnData containing multiple embeddings in ``.obsm``.
    obsm_keys
        Embedding keys to plot.  ``None`` → all discovered keys.
    method
        ``"umap"`` or ``"tsne"``.
    color
        Column in ``adata.obs`` to color points by.
    ncols
        Number of columns in the grid.
    figsize_per_panel
        Size of each individual panel.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    keys = _get_embedding_keys(adata, obsm_keys)
    if not keys:
        raise ValueError("No embedding keys found in adata.obsm.")

    n = len(keys)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        plot_embedding_space(adata, obsm_key=key, color=color, method=method, ax=axes[r][c])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Similarity / distance / correlation heatmaps
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(
    similarity_matrix: np.ndarray | None = None,
    adata: AnnData | None = None,
    obsm_key: str | None = None,
    metric: str = "cosine",
    labels: list[str] | None = None,
    title: str = "Perturbation Similarity",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    **kwargs: Any,
) -> Figure:
    """Heatmap of pairwise perturbation similarities.

    Either provide a pre-computed *similarity_matrix*, or pass *adata*
    and *obsm_key* to compute it on the fly.

    Parameters
    ----------
    similarity_matrix
        Pre-computed square similarity matrix.
    adata
        AnnData (used when *similarity_matrix* is ``None``).
    obsm_key
        Embedding key (used with *adata*).
    metric
        Similarity metric (``"cosine"``, ``"pearson"``, ``"spearman"``).
    labels
        Tick labels for rows/columns.
    title
        Plot title.
    figsize
        Figure size.
    cmap
        Colormap.
    **kwargs
        Passed to ``seaborn.heatmap``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if similarity_matrix is None:
        if adata is None or obsm_key is None:
            raise ValueError("Provide either 'similarity_matrix' or both 'adata' and 'obsm_key'.")
        similarity_matrix = tl.compute_similarity(adata, obsm_key=obsm_key, metric=metric)
        if labels is None:
            labels = _labels_from_adata(adata)

    fig, ax = plt.subplots(figsize=figsize)
    hm_kw: dict[str, Any] = {"vmin": -1, "vmax": 1, "center": 0, "square": True, "linewidths": 0.5}
    hm_kw.update(kwargs)
    sns.heatmap(
        similarity_matrix, ax=ax, cmap=cmap,
        xticklabels=labels or False,
        yticklabels=labels or False,
        **hm_kw,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def distance_heatmap(
    adata: AnnData,
    obsm_key: str,
    metric: str = "euclidean",
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    **kwargs: Any,
) -> Figure:
    """Heatmap of pairwise distances.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key.
    metric
        ``"euclidean"``, ``"cosine"``, or ``"wasserstein"``.
    labels
        Tick labels.
    title
        Plot title.
    figsize
        Figure size.
    cmap
        Colormap.
    **kwargs
        Passed to ``seaborn.heatmap``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    D = tl.compute_distance_matrix(adata, obsm_key=obsm_key, metric=metric)
    if labels is None:
        labels = _labels_from_adata(adata)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        D, ax=ax, cmap=cmap, square=True, linewidths=0.5,
        xticklabels=labels or False,
        yticklabels=labels or False,
        **kwargs,
    )
    ax.set_title(title or f"Pairwise {metric.title()} Distance ({obsm_key})")
    fig.tight_layout()
    return fig


def correlation_matrix(
    adata: AnnData,
    obsm_key: str,
    method: str = "pearson",
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    **kwargs: Any,
) -> Figure:
    """Heatmap of pairwise Pearson or Spearman correlation.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key.
    method
        ``"pearson"`` or ``"spearman"``.
    labels
        Tick labels.
    title
        Plot title.
    figsize
        Figure size.
    cmap
        Colormap.
    **kwargs
        Passed to ``seaborn.heatmap``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    sim = tl.compute_similarity(adata, obsm_key=obsm_key, metric=method)
    if labels is None:
        labels = _labels_from_adata(adata)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sim, ax=ax, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
        xticklabels=labels or False,
        yticklabels=labels or False,
        **kwargs,
    )
    ax.set_title(title or f"{method.title()} Correlation ({obsm_key})")
    fig.tight_layout()
    return fig


def cross_embedding_correlation(
    adata: AnnData,
    obsm_key_a: str,
    obsm_key_b: str,
    method: str = "pearson",
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """Scatter comparing pairwise similarities from two embedding spaces.

    For each pair of observations the similarity (cosine) is computed in
    both spaces.  The scatter shows their relationship, annotated with
    the Pearson or Spearman correlation.

    Parameters
    ----------
    adata
        AnnData with both embeddings.
    obsm_key_a
        First embedding key.
    obsm_key_b
        Second embedding key.
    method
        Correlation method (``"pearson"`` or ``"spearman"``).
    figsize
        Figure size.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    from scipy.stats import pearsonr as _pr, spearmanr as _sr

    sim_a = tl.compute_similarity(adata, obsm_key=obsm_key_a, metric="cosine")
    sim_b = tl.compute_similarity(adata, obsm_key=obsm_key_b, metric="cosine")

    iu = np.triu_indices(sim_a.shape[0], k=1)
    flat_a = sim_a[iu]
    flat_b = sim_b[iu]

    if method == "pearson":
        corr, pval = _pr(flat_a, flat_b)
    elif method == "spearman":
        corr, pval = _sr(flat_a, flat_b)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(flat_a, flat_b, s=4, alpha=0.3)
    ax.set_xlabel(f"Cosine similarity ({obsm_key_a})")
    ax.set_ylabel(f"Cosine similarity ({obsm_key_b})")
    ax.set_title(f"{method.title()} r = {corr:.4f}  (p = {pval:.2e})")
    lims = [min(flat_a.min(), flat_b.min()), max(flat_a.max(), flat_b.max())]
    ax.plot(lims, lims, "r--", lw=0.8, alpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# KNN overlap
# ---------------------------------------------------------------------------

def knn_overlap(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
    k: int = 15,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "YlGnBu",
) -> Figure:
    """Heatmap of mean KNN-Jaccard overlap between embedding spaces.

    For every pair of embedding spaces, the mean per-observation Jaccard
    index of *k*-nearest-neighbor sets is computed and displayed.

    Parameters
    ----------
    adata
        AnnData with multiple embeddings.
    obsm_keys
        Keys to compare.  ``None`` → all discovered keys.
    k
        Number of nearest neighbors.
    figsize
        Figure size.
    cmap
        Colormap.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    keys = _get_embedding_keys(adata, obsm_keys)
    if len(keys) < 2:
        raise ValueError("At least two embedding keys are needed for KNN overlap comparison.")

    n = len(keys)
    overlap = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            _, mean_j = tl.compute_knn_overlap(adata, keys[i], keys[j], k=k)
            overlap[i, j] = mean_j
            overlap[j, i] = mean_j

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        overlap, ax=ax, cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".3f",
        square=True, linewidths=0.5,
        xticklabels=keys, yticklabels=keys,
    )
    ax.set_title(f"KNN Overlap (Jaccard, k={k})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Perturbation ranking
# ---------------------------------------------------------------------------

def plot_perturbation_ranking(
    rankings: list[tuple[str, float]] | None = None,
    adata: AnnData | None = None,
    query: str | np.ndarray | None = None,
    obsm_key: str | None = None,
    query_label: str = "Query",
    top_k: int = 20,
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Figure:
    """Horizontal bar plot of top-ranked perturbations by similarity.

    Either provide pre-computed *rankings*, or pass *adata*, *query*,
    and *obsm_key* to compute them.

    Parameters
    ----------
    rankings
        Pre-computed list of ``(id, score)`` tuples.
    adata
        AnnData (used when *rankings* is ``None``).
    query
        Query identifier or vector (used with *adata*).
    obsm_key
        Embedding key (used with *adata*).
    query_label
        Label for the query perturbation in the title.
    top_k
        Number of results to show.
    figsize
        Figure size.
    **kwargs
        Passed to ``matplotlib.axes.Axes.barh``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if rankings is None:
        if adata is None or query is None or obsm_key is None:
            raise ValueError("Provide either 'rankings' or all of 'adata', 'query', 'obsm_key'.")
        rankings = tl.rank_perturbations(adata, query=query, obsm_key=obsm_key, top_k=top_k)

    rankings = rankings[:top_k]
    names = [r[0] for r in rankings]
    scores = [r[1] for r in rankings]

    fig, ax = plt.subplots(figsize=figsize)
    bar_kw: dict[str, Any] = {"color": sns.color_palette("viridis", n_colors=len(names))}
    bar_kw.update(kwargs)
    y_pos = range(len(names))
    ax.barh(y_pos, scores, **bar_kw)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top {len(names)} perturbations similar to {query_label}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Clustering visualizations
# ---------------------------------------------------------------------------

def plot_cluster_composition(
    adata: AnnData,
    cluster_key: str = "cluster",
    color_by: str = "perturbation_type",
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Figure:
    """Stacked bar chart of cluster composition by a categorical variable.

    Parameters
    ----------
    adata
        AnnData with cluster assignments and metadata.
    cluster_key
        Column in ``obs`` with cluster labels.
    color_by
        Column in ``obs`` to break down each cluster by.
    figsize
        Figure size.
    **kwargs
        Passed to ``pandas.DataFrame.plot.bar``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"'{cluster_key}' not found in adata.obs.")
    if color_by not in adata.obs.columns:
        raise KeyError(f"'{color_by}' not found in adata.obs.")

    ct = adata.obs.groupby([cluster_key, color_by], observed=True).size().unstack(fill_value=0)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    ct_norm.plot.bar(stacked=True, ax=ax, **kwargs)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Cluster")
    ax.set_title(f"Cluster Composition by {color_by}")
    ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def leiden_overview(
    adata: AnnData,
    obsm_key: str,
    resolution: float = 1.0,
    color_by: str | None = None,
    plots: list[str] | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """Multi-panel Leiden clustering overview.

    Runs Leiden clustering (if not already done) and produces a figure
    with selectable panels.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key for clustering and UMAP.
    resolution
        Leiden resolution parameter.
    color_by
        Optional metadata column for an additional UMAP panel.
    plots
        Which panels to include.  Options: ``"umap_cluster"``,
        ``"umap_metadata"``, ``"composition"``, ``"cluster_sizes"``.
        ``None`` → all applicable panels.
    figsize
        Overall figure size.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    leiden_key = f"leiden_{obsm_key}"
    if leiden_key not in adata.obs.columns:
        tl.leiden(adata, obsm_key=obsm_key, resolution=resolution, key_added=leiden_key)

    umap_key = f"X_umap_{obsm_key}"
    if umap_key not in adata.obsm:
        tl.compute_umap(adata, obsm_key=obsm_key)

    available: list[str] = ["umap_cluster", "cluster_sizes"]
    if color_by and color_by in adata.obs.columns:
        available.insert(1, "umap_metadata")
        available.append("composition")

    selected = plots if plots is not None else available
    n_panels = len(selected)
    ncols = min(n_panels, 2)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for idx, panel in enumerate(selected):
        ax = flat_axes[idx]

        if panel == "umap_cluster":
            plot_embedding_space(adata, obsm_key=obsm_key, color=leiden_key,
                                basis=umap_key, title=f"Leiden Clusters ({obsm_key})", ax=ax)

        elif panel == "umap_metadata":
            if color_by and color_by in adata.obs.columns:
                plot_embedding_space(adata, obsm_key=obsm_key, color=color_by,
                                    basis=umap_key, title=f"{color_by} ({obsm_key})", ax=ax)
            else:
                ax.text(0.5, 0.5, "No metadata column", ha="center", va="center", transform=ax.transAxes)

        elif panel == "composition":
            if color_by and color_by in adata.obs.columns:
                ct = adata.obs.groupby([leiden_key, color_by], observed=True).size().unstack(fill_value=0)
                ct_norm = ct.div(ct.sum(axis=1), axis=0)
                ct_norm.plot.bar(stacked=True, ax=ax)
                ax.set_ylabel("Fraction")
                ax.set_xlabel("Cluster")
                ax.set_title("Cluster Composition")
                ax.legend(title=color_by, fontsize=6, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No color_by provided", ha="center", va="center", transform=ax.transAxes)

        elif panel == "cluster_sizes":
            sizes = adata.obs[leiden_key].value_counts().sort_index()
            sizes.plot.bar(ax=ax, color=sns.color_palette("husl", n_colors=len(sizes)))
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            ax.set_title("Cluster Sizes")

        else:
            logging.warning("Unknown panel '%s'; skipping.", panel)
            ax.set_visible(False)

    for idx in range(len(selected), nrows * ncols):
        flat_axes[idx].set_visible(False)

    fig.suptitle(f"Leiden Overview — {obsm_key} (resolution={resolution})", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------

def dendrogram(
    adata: AnnData,
    obsm_key: str,
    metric: str = "cosine",
    linkage_method: str = "average",
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    leaf_font_size: int = 8,
) -> Figure:
    """Hierarchical-clustering dendrogram of perturbation embeddings.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key.
    metric
        Distance metric (``"cosine"``, ``"euclidean"``, ``"correlation"``).
    linkage_method
        Linkage method (``"average"``, ``"complete"``, ``"single"``, ``"ward"``).
        Note: ``"ward"`` requires ``metric="euclidean"``.
    labels
        Leaf labels.
    title
        Plot title.
    figsize
        Figure size.
    leaf_font_size
        Font size for leaf labels.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    D = tl.compute_distance_matrix(adata, obsm_key=obsm_key, metric=metric)
    np.fill_diagonal(D, 0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if labels is None:
        labels = _labels_from_adata(adata)

    fig, ax = plt.subplots(figsize=figsize)
    _scipy_dendrogram(Z, labels=labels, ax=ax, leaf_font_size=leaf_font_size, leaf_rotation=90)
    ax.set_title(title or f"Hierarchical Clustering ({obsm_key}, {metric})")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Embedding distribution / norms
# ---------------------------------------------------------------------------

def embedding_distributions(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
    n_dims: int = 10,
    figsize_per_panel: tuple[float, float] = (10, 3),
) -> Figure:
    """Violin plots of embedding dimension values.

    For each embedding key, shows the distribution of the first *n_dims*
    dimensions across all observations.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_keys
        Keys to plot.  ``None`` → all discovered.
    n_dims
        Number of leading dimensions to visualize.
    figsize_per_panel
        Size per row.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    keys = _get_embedding_keys(adata, obsm_keys)
    if not keys:
        raise ValueError("No embedding keys found.")

    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx, 0]
        X = np.asarray(adata.obsm[key], dtype=np.float64)
        ndims = min(n_dims, X.shape[1])
        parts = ax.violinplot([X[:, d] for d in range(ndims)], showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, ndims + 1))
        ax.set_xticklabels([f"d{d}" for d in range(ndims)], fontsize=7)
        ax.set_title(f"{key} (first {ndims} dims)")
        ax.set_ylabel("Value")

    fig.tight_layout()
    return fig


def embedding_norms(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """Box plot of L2 norms across embedding spaces.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_keys
        Keys to compare.  ``None`` → all discovered.
    figsize
        Figure size.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    keys = _get_embedding_keys(adata, obsm_keys)
    if not keys:
        raise ValueError("No embedding keys found.")

    data: list[np.ndarray] = []
    for key in keys:
        X = np.asarray(adata.obsm[key], dtype=np.float64)
        norms = np.linalg.norm(X, axis=1)
        data.append(norms)

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, tick_labels=keys, patch_artist=True, showmeans=True)
    palette = sns.color_palette("husl", n_colors=len(keys))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("L2 Norm")
    ax.set_title("Embedding Norms")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig
