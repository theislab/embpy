"""Clustering visualizations: Leiden overview, composition, dendrogram."""

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

from ._helpers import _labels_from_adata
from .embedding_space import plot_embedding_space


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
        ``None`` -> all applicable panels.
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

    fig.suptitle(f"Leiden Overview -- {obsm_key} (resolution={resolution})", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


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
