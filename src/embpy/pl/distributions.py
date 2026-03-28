"""Embedding distribution, norm, and perturbation ranking plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure

from embpy import tl

from ._helpers import _get_embedding_keys


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
        Keys to plot.  ``None`` -> all discovered.
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
        Keys to compare.  ``None`` -> all discovered.
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
