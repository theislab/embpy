"""2-D embedding scatter plots (UMAP, t-SNE) and feature-colored panels."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure

from embpy import tl

from ._helpers import _get_embedding_keys


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
        Embedding keys to plot.  ``None`` -> all discovered keys.
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


def umap_feature_panel(
    adata: AnnData,
    obsm_key: str,
    features: list[str],
    ncols: int = 3,
    point_size: float = 6,
    cmap: str = "viridis",
    figsize_per_panel: tuple[float, float] = (5, 4.5),
    title: str | None = None,
) -> Figure:
    """Grid of UMAPs colored by continuous features from ``adata.obs``.

    UMAP coordinates are taken from ``obsm[X_umap_{obsm_key}]``.  If
    they do not exist yet, they are computed automatically.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key used for the UMAP layout.
    features
        Column names in ``adata.obs`` to color by.
    ncols
        Number of columns in the grid.
    point_size
        Scatter point size.
    cmap
        Colormap for continuous values.
    figsize_per_panel
        Size of each individual panel.
    title
        Overall figure title.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    umap_key = f"X_umap_{obsm_key}"
    if umap_key not in adata.obsm:
        tl.compute_umap(adata, obsm_key=obsm_key)
    coords = np.asarray(adata.obsm[umap_key])

    n = len(features)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for idx, feat in enumerate(features):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if feat not in adata.obs.columns:
            ax.text(0.5, 0.5, f"'{feat}' not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        vals = adata.obs[feat].values.copy().astype(float)
        valid = ~np.isnan(vals)
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=1, c="lightgray", alpha=0.3)
        sc = ax.scatter(
            coords[valid, 0], coords[valid, 1],
            s=point_size, c=vals[valid], cmap=cmap, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
