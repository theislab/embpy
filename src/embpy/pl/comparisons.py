"""Comparison visualizations: parallel coordinates, radar charts,
star coordinates, and t-SNE feature panels."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure

from embpy import tl


# -----------------------------------------------------------------------
# Parallel coordinates
# -----------------------------------------------------------------------

def parallel_coordinates(
    adata: AnnData,
    obsm_key: str,
    obs_features: list[str],
    n_dims: int = 5,
    color_by: str = "leiden",
    figsize: tuple[float, float] = (14, 6),
    title: str | None = None,
    alpha: float = 0.3,
    **kwargs: Any,
) -> Figure:
    """Parallel coordinates plot of embedding dimensions and annotations.

    Each vertical axis represents either a leading PCA dimension or a
    numeric annotation from ``adata.obs``.  Lines are colored by a
    categorical column, making it easy to spot multi-variate patterns
    that separate clusters.

    Parameters
    ----------
    adata
        AnnData with embeddings and annotations.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    obs_features
        Numeric columns from ``adata.obs`` to include as axes.
    n_dims
        Number of leading embedding dimensions to include (set 0 to
        use only ``obs_features``).
    color_by
        Categorical column in ``adata.obs`` for line coloring.
    figsize
        Figure size.
    title
        Plot title.
    alpha
        Line transparency.
    **kwargs
        Passed to ``pandas.plotting.parallel_coordinates``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    from pandas.plotting import parallel_coordinates as _pd_parallel

    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not in adata.obsm.")
    if color_by not in adata.obs.columns:
        raise KeyError(f"'{color_by}' not in adata.obs.")

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    actual_dims = min(n_dims, X.shape[1])

    cols: list[str] = []
    data_parts: list[np.ndarray] = []

    if actual_dims > 0:
        dim_data = X[:, :actual_dims]
        mu = dim_data.mean(axis=0)
        sd = dim_data.std(axis=0)
        sd[sd == 0] = 1.0
        dim_data = (dim_data - mu) / sd
        for i in range(actual_dims):
            cols.append(f"dim_{i}")
        data_parts.append(dim_data)

    for feat in obs_features:
        if feat not in adata.obs.columns:
            continue
        vals = adata.obs[feat].values.astype(float)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 2:
            continue
        mu_f = np.nanmean(vals)
        sd_f = np.nanstd(vals)
        if sd_f == 0:
            sd_f = 1.0
        vals = np.where(valid_mask, (vals - mu_f) / sd_f, 0.0)
        cols.append(feat.replace("_", " ").title())
        data_parts.append(vals.reshape(-1, 1))

    if not cols:
        raise ValueError("No valid columns to plot.")

    mat = np.hstack(data_parts)
    df = pd.DataFrame(mat, columns=cols)
    df[color_by] = adata.obs[color_by].values

    df = df.dropna(subset=[color_by]).copy()
    if len(df) == 0:
        raise ValueError("No rows remain after filtering.")

    fig, ax = plt.subplots(figsize=figsize)
    pc_kw: dict[str, Any] = {"alpha": alpha, "linewidth": 0.8}
    pc_kw.update(kwargs)

    n_groups = df[color_by].nunique()
    palette = sns.color_palette("tab20", n_colors=max(1, n_groups))
    _pd_parallel(df, color_by, ax=ax, color=palette, **pc_kw)

    ax.legend(
        title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, ncol=max(1, n_groups // 10),
    )
    ax.set_title(title or f"Parallel coordinates ({obsm_key})", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Radar / spider chart
# -----------------------------------------------------------------------

def radar_chart(
    adata: AnnData,
    properties: list[str],
    group_by: str = "leiden",
    groups: list[str] | None = None,
    z_score: bool = True,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    alpha: float = 0.15,
) -> Figure:
    """Radar (spider) chart comparing mean property profiles across groups.

    Each spoke of the radar represents a numeric property from
    ``adata.obs``.  Groups (e.g. Leiden clusters) are overlaid as
    separate polygons, making it easy to see which clusters are
    enriched for specific molecular properties.

    Parameters
    ----------
    adata
        AnnData with numeric properties in ``obs``.
    properties
        Numeric columns from ``adata.obs`` to plot as spokes.
    group_by
        Categorical column defining groups.
    groups
        Subset of group labels to show.  ``None`` picks the top 6
        largest groups.
    z_score
        Whether to z-score each property across all observations
        before computing group means.
    figsize
        Figure size.
    title
        Plot title.
    alpha
        Fill transparency for each polygon.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if group_by not in adata.obs.columns:
        raise KeyError(f"'{group_by}' not in adata.obs.")
    valid_props = [p for p in properties if p in adata.obs.columns]
    if len(valid_props) < 3:
        raise ValueError("Need at least 3 valid properties for a radar chart.")

    df = pd.DataFrame(adata.obs[valid_props + [group_by]]).dropna()

    if groups is None:
        groups = list(df[group_by].value_counts().head(6).index)
    df = df[df[group_by].isin(groups)]

    if z_score:
        for p in valid_props:
            mu = df[p].mean()
            sd = df[p].std()
            if sd == 0:
                sd = 1.0
            df[p] = (df[p] - mu) / sd

    means = df.groupby(group_by, observed=True)[valid_props].mean()

    labels = [p.replace("_", " ").title() for p in valid_props]
    n_props = len(valid_props)
    angles = np.linspace(0, 2 * np.pi, n_props, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    palette = sns.color_palette("tab10", n_colors=len(groups))

    for i, grp in enumerate(groups):
        if grp not in means.index:
            continue
        values = means.loc[grp].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=str(grp), color=palette[i])
        ax.fill(angles, values, alpha=alpha, color=palette[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.legend(title=group_by, loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title(title or f"Radar chart by {group_by}", fontsize=13, y=1.08)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Star coordinates
# -----------------------------------------------------------------------

def star_coordinates(
    adata: AnnData,
    obsm_key: str,
    n_dims: int = 10,
    color_by: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 6,
    cmap: str = "viridis",
    title: str | None = None,
) -> Figure:
    """Star coordinates projection of high-dimensional embeddings.

    Places ``n_dims`` feature axes as evenly-spaced rays from the
    origin.  Each observation's 2-D position is the weighted sum of
    the unit direction vectors, weighted by its (standardized)
    embedding values.  This preserves axis interpretability -- the
    direction of each ray shows where high values of that dimension
    pull observations.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_dims
        Number of leading dimensions to use as star axes.
    color_by
        Column in ``adata.obs`` to color points by (continuous or
        categorical).  ``None`` for uniform color.
    figsize
        Figure size.
    point_size
        Scatter point size.
    cmap
        Colormap for continuous coloring.
    title
        Plot title.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not in adata.obsm.")

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    actual_dims = min(n_dims, X.shape[1])
    X = X[:, :actual_dims]

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    angles = np.linspace(0, 2 * np.pi, actual_dims, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    coords = X @ dirs

    fig, ax = plt.subplots(figsize=figsize)

    if color_by and color_by in adata.obs.columns:
        vals = adata.obs[color_by]
        if vals.dtype.name == "category" or vals.nunique() <= 20:
            cats = vals.astype("category")
            palette = sns.color_palette("tab20", n_colors=cats.cat.categories.size)
            color_map = {c: palette[i] for i, c in enumerate(cats.cat.categories)}
            c_vals = [color_map[v] for v in cats]
            ax.scatter(coords[:, 0], coords[:, 1], s=point_size, c=c_vals, alpha=0.6)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[c], markersize=7, label=str(c))
                for c in cats.cat.categories
            ]
            ax.legend(handles=handles, title=color_by, bbox_to_anchor=(1.05, 1),
                      loc="upper left", fontsize=7)
        else:
            c_float = vals.values.astype(float)
            valid = ~np.isnan(c_float)
            ax.scatter(coords[~valid, 0], coords[~valid, 1], s=1, c="lightgray", alpha=0.2)
            sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=point_size,
                            c=c_float[valid], cmap=cmap, alpha=0.6)
            plt.colorbar(sc, ax=ax, shrink=0.7, label=color_by)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=0.6, color="steelblue")

    ray_len = np.abs(coords).max() * 1.15
    for i in range(actual_dims):
        dx, dy = dirs[i] * ray_len
        ax.annotate(
            f"d{i}", xy=(dx, dy), fontsize=8, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gray", "alpha": 0.8},
        )
        ax.plot([0, dx], [0, dy], color="gray", linewidth=0.8, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"Star coordinates ({obsm_key}, {actual_dims} dims)", fontsize=13)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# t-SNE feature panel
# -----------------------------------------------------------------------

def tsne_feature_panel(
    adata: AnnData,
    obsm_key: str,
    features: list[str],
    ncols: int = 3,
    point_size: float = 6,
    cmap: str = "viridis",
    figsize_per_panel: tuple[float, float] = (5, 4.5),
    title: str | None = None,
) -> Figure:
    """Grid of t-SNE plots colored by continuous features from ``adata.obs``.

    t-SNE coordinates are taken from ``obsm[X_tsne_{obsm_key}]``.  If
    they do not exist yet, they are computed automatically via
    ``tl.compute_tsne``.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Embedding key used for the t-SNE layout.
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
    tsne_key = f"X_tsne_{obsm_key}"
    if tsne_key not in adata.obsm:
        tl.compute_tsne(adata, obsm_key=obsm_key)
    coords = np.asarray(adata.obsm[tsne_key])

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
