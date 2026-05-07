"""2-D embedding scatter plots (PCA, UMAP, t-SNE) and feature-colored panels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

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
    method: Literal["pca", "umap", "tsne"] = "umap",
    basis: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Any = None,
    annotate: bool = False,
    annotate_col: str | None = None,
    palette: str = "husl",
    **kwargs: Any,
) -> Figure:
    """Plot embeddings in a 2-D reduced space.

    If *basis* is not provided, PCA / UMAP / t-SNE coordinates are computed
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
        Reduction method when *basis* is ``None``:
        ``"pca"``, ``"umap"``, or ``"tsne"``.
    basis
        Explicit key in ``.obsm`` for 2-D coordinates.  Overrides *method*.
    title
        Plot title.
    figsize
        Figure size as ``(width, height)``.
    ax
        Optional matplotlib ``Axes`` to draw into.
    annotate
        If ``True``, draw a text label next to each point. Useful for
        small panels (< ~50 points). Label text comes from
        ``adata.obs[annotate_col]`` if set, otherwise ``adata.obs_names``.
    annotate_col
        Optional ``adata.obs`` column to use for point labels when
        ``annotate=True``.
    palette
        Name of the seaborn / matplotlib palette for categorical colors.
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

    method = method.lower()
    if basis is not None:
        coords_key = basis
    elif method == "pca":
        coords_key = f"X_pca_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_pca(adata, obsm_key=obsm_key)
    elif method == "umap":
        coords_key = f"X_umap_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_umap(adata, obsm_key=obsm_key)
    elif method == "tsne":
        coords_key = f"X_tsne_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_tsne(adata, obsm_key=obsm_key)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pca', 'umap', or 'tsne'.")

    if coords_key not in adata.obsm:
        raise KeyError(f"'{coords_key}' not found in adata.obsm.")
    coords = np.asarray(adata.obsm[coords_key])

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter_kw: dict[str, Any] = {"s": 40, "alpha": 0.85, "edgecolors": "white", "linewidth": 0.5}
    scatter_kw.update(kwargs)

    if color and color in adata.obs.columns:
        cats = adata.obs[color]
        if cats.dtype.name == "category" or cats.nunique() <= 20:
            cats = cats.astype("category")
            colors = sns.color_palette(palette, n_colors=cats.cat.categories.size)
            color_map = {c: colors[i] for i, c in enumerate(cats.cat.categories)}
            c_vals = [color_map[v] for v in cats]
            ax.scatter(coords[:, 0], coords[:, 1], c=c_vals, **scatter_kw)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                           markersize=8, label=str(c))
                for c in cats.cat.categories
            ]
            ax.legend(handles=handles, title=color, bbox_to_anchor=(1.05, 1),
                      loc="upper left", fontsize=7, frameon=False)
        else:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cats.values,
                                 cmap="viridis", **scatter_kw)
            plt.colorbar(scatter, ax=ax, shrink=0.8, label=color)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kw)

    if annotate:
        labels = (
            adata.obs[annotate_col].astype(str).tolist()
            if annotate_col and annotate_col in adata.obs.columns
            else list(adata.obs_names)
        )
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, coords[i], textcoords="offset points",
                        xytext=(4, 3), fontsize=7, alpha=0.85)

    if method == "pca" and basis is None:
        var_key = f"{coords_key}_variance_ratio"
        if var_key in adata.uns:
            ratios = np.asarray(adata.uns[var_key])
            ax.set_xlabel(f"PC1 ({ratios[0] * 100:.0f}%)")
            ax.set_ylabel(f"PC2 ({ratios[1] * 100:.0f}%)")
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    else:
        method_name = method.upper() if basis is None else basis
        ax.set_xlabel(f"{method_name}1")
        ax.set_ylabel(f"{method_name}2")

    method_disp = method.upper() if basis is None else basis
    ax.set_title(title or f"{obsm_key} ({method_disp})")
    ax.spines[["top", "right"]].set_visible(False)

    if created_fig:
        fig.tight_layout()
    return fig


def embedding_color_panel(
    adata: AnnData,
    color_keys: Sequence[str],
    obsm_key: str | None = None,
    method: Literal["pca", "umap", "tsne"] = "pca",
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.5, 4.5),
    palette: str = "husl",
    annotate: bool = False,
    annotate_col: str | None = None,
    title: str | None = None,
) -> Figure:
    """Grid of 2-D scatters of *the same* embedding, colored by different ``obs`` columns.

    Sister of :func:`all_embeddings` (which plots different embeddings) --
    here the layout is computed once and you compare how different
    annotations paint the same manifold.

    Parameters
    ----------
    adata
        AnnData with the embedding in ``obsm[obsm_key]``.
    color_keys
        Columns in ``adata.obs`` to color each panel by.
    obsm_key
        Embedding key. Defaults to the first discovered key.
    method
        2-D reduction method (``"pca"`` / ``"umap"`` / ``"tsne"``).
    ncols
        Number of columns in the grid.
    figsize_per_panel
        Size of each panel.
    palette
        Color palette for categorical labels.
    annotate
        Forwarded to :func:`plot_embedding_space`.
    annotate_col
        Forwarded to :func:`plot_embedding_space`.
    title
        Overall figure title.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]

    n = len(color_keys)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for idx, col in enumerate(color_keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        plot_embedding_space(
            adata, obsm_key=obsm_key, color=col, method=method,
            ax=ax, annotate=annotate, annotate_col=annotate_col,
            palette=palette, title=col,
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def highlight_gene_sets(
    adata: AnnData,
    gene_sets: Mapping[str, Sequence[str]],
    obsm_key: str | None = None,
    method: Literal["pca", "umap", "tsne"] = "pca",
    label_col: str | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    palette: str = "tab10",
    annotate: bool = True,
    title: str | None = None,
) -> Figure:
    """Multi-panel scatter; each panel highlights members of one named set.

    Members of the set are drawn in color on top of all other observations
    in light gray. Useful for asking *"do the genes in this curated list
    form a tight neighborhood?"*.

    Parameters
    ----------
    adata
        AnnData with the embedding in ``obsm[obsm_key]``.
    gene_sets
        Mapping ``{set_name: [member1, member2, ...]}``.
    obsm_key
        Embedding key. Defaults to the first discovered key.
    method
        2-D reduction method.
    label_col
        Column in ``adata.obs`` whose values match the strings in
        *gene_sets*. Defaults to ``adata.obs_names``.
    ncols
        Number of columns.
    figsize_per_panel
        Size of each panel.
    palette
        Color palette (one color per set).
    annotate
        If ``True``, draw labels next to highlighted points.
    title
        Overall figure title.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]

    method = method.lower()
    if method == "pca":
        coords_key = f"X_pca_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_pca(adata, obsm_key=obsm_key)
    elif method == "umap":
        coords_key = f"X_umap_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_umap(adata, obsm_key=obsm_key)
    elif method == "tsne":
        coords_key = f"X_tsne_{obsm_key}"
        if coords_key not in adata.obsm:
            tl.compute_tsne(adata, obsm_key=obsm_key)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    coords = np.asarray(adata.obsm[coords_key])
    names = (
        adata.obs[label_col].astype(str).tolist()
        if label_col and label_col in adata.obs.columns
        else list(adata.obs_names)
    )
    name_to_idx = {n: i for i, n in enumerate(names)}

    keys = list(gene_sets.keys())
    n = len(keys)
    nrows = math.ceil(n / ncols)
    colors = sns.color_palette(palette, n_colors=max(1, n))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        members = [g for g in gene_sets[key] if g in name_to_idx]
        member_idx = np.array([name_to_idx[g] for g in members], dtype=int)
        bg_mask = np.ones(len(names), dtype=bool)
        bg_mask[member_idx] = False
        ax.scatter(coords[bg_mask, 0], coords[bg_mask, 1],
                   s=35, color=(0.85, 0.85, 0.85), alpha=0.6,
                   edgecolors="white", linewidths=0.3)
        if len(member_idx):
            ax.scatter(coords[member_idx, 0], coords[member_idx, 1],
                       s=110, color=colors[idx % len(colors)],
                       edgecolors="black", linewidths=0.6, alpha=0.95, zorder=3)
            if annotate:
                for g in members:
                    i = name_to_idx[g]
                    ax.annotate(g, coords[i], textcoords="offset points",
                                xytext=(4, 3), fontsize=7, alpha=0.9)
        ax.set_title(f"{key}  (n={len(member_idx)})", fontweight="bold", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
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
