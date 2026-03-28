"""Similarity, distance, correlation, and embedding heatmaps."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure

from embpy import tl

from ._helpers import _get_embedding_keys, _labels_from_adata


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
        Keys to compare.  ``None`` -> all discovered keys.
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


def embedding_clustermap(
    adata: AnnData,
    obsm_key: str,
    n_obs: int | None = 200,
    z_score: bool = True,
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "RdBu_r",
    vmin: float = -3,
    vmax: float = 3,
    title: str | None = None,
    **kwargs: Any,
) -> sns.matrix.ClusterGrid:
    """Hierarchically-clustered heatmap of raw embedding values.

    Rows are observations, columns are embedding dimensions.  By default
    the matrix is z-scored per dimension so that the color scale is
    comparable across dimensions with different magnitudes.

    Parameters
    ----------
    adata
        AnnData with embeddings.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_obs
        Maximum number of observations to show (subsampled randomly).
        ``None`` uses all observations.
    z_score
        Whether to z-score each dimension (column).
    figsize
        Figure size.
    cmap
        Colormap.
    vmin, vmax
        Color scale limits (applied after z-scoring if enabled).
    title
        Plot title.
    **kwargs
        Passed to ``seaborn.clustermap``.

    Returns
    -------
    The seaborn ``ClusterGrid``.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not in adata.obsm. Available: {list(adata.obsm.keys())}")
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)

    if n_obs is not None and X.shape[0] > n_obs:
        idx = np.random.default_rng(0).choice(X.shape[0], size=n_obs, replace=False)
        X = X[idx]

    if z_score:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    clust_kw: dict[str, Any] = {
        "row_cluster": True, "col_cluster": True,
        "xticklabels": False, "yticklabels": False,
        "cbar_kws": {"label": "z-score" if z_score else "value"},
    }
    clust_kw.update(kwargs)
    g = sns.clustermap(
        X, cmap=cmap, center=0 if z_score else None,
        vmin=vmin, vmax=vmax, figsize=figsize, **clust_kw,
    )
    label = title or f"Embedding heatmap: {obsm_key} ({X.shape[0]} obs x {X.shape[1]} dims)"
    g.fig.suptitle(label, y=1.02, fontsize=13)
    return g


def cross_model_similarity(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
    method: str = "cosine_correlation",
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "RdBu_r",
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Heatmap comparing similarity structure across embedding models.

    For each pair of embedding spaces, computes pairwise cosine
    similarity matrices and then measures their agreement using
    Pearson correlation (``"cosine_correlation"``), Adjusted Rand Index
    of Leiden clusters (``"ari"``), or KNN Jaccard overlap (``"knn"``).

    Parameters
    ----------
    adata
        AnnData with multiple embeddings.
    obsm_keys
        Keys to compare.  ``None`` -> all discovered keys.
    method
        Agreement metric:

        - ``"cosine_correlation"`` -- Pearson r between upper-triangular
          entries of the cosine similarity matrices.
        - ``"ari"`` -- Adjusted Rand Index between Leiden clusterings.
        - ``"knn"`` -- mean Jaccard overlap of k-nearest-neighbor sets.
    labels
        Display labels for each model.  Defaults to the obsm keys.
    figsize
        Figure size.
    cmap
        Colormap.
    title
        Plot title.
    **kwargs
        Extra keyword arguments.  For ``method="knn"``, accepts ``k``
        (default 20).  For ``method="ari"``, accepts ``resolution``
        (default 0.5).

    Returns
    -------
    The matplotlib ``Figure``.
    """
    from scipy.stats import pearsonr as _pearsonr

    keys = _get_embedding_keys(adata, obsm_keys)
    if len(keys) < 2:
        raise ValueError("Need at least 2 embedding keys.")
    n = len(keys)
    display_labels = labels or keys

    mat = np.ones((n, n), dtype=np.float64)

    if method == "cosine_correlation":
        sims = {}
        for k in keys:
            sims[k] = tl.compute_similarity(adata, obsm_key=k, metric="cosine")
        iu = np.triu_indices(adata.n_obs, k=1)
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = _pearsonr(sims[keys[i]][iu], sims[keys[j]][iu])
                mat[i, j] = r
                mat[j, i] = r
        default_title = "Cross-model agreement\n(Pearson r of cosine similarity matrices)"
        vmin, vmax = -1.0, 1.0

    elif method == "ari":
        from sklearn.metrics import adjusted_rand_score
        res = kwargs.pop("resolution", 0.5)
        for k in keys:
            lk = f"leiden_{k}"
            if lk not in adata.obs.columns:
                tl.leiden(adata, obsm_key=k, resolution=res, key_added=lk)
        for i in range(n):
            for j in range(i + 1, n):
                ari = adjusted_rand_score(
                    adata.obs[f"leiden_{keys[i]}"],
                    adata.obs[f"leiden_{keys[j]}"],
                )
                mat[i, j] = ari
                mat[j, i] = ari
        default_title = f"Adjusted Rand Index between Leiden clusterings (res={res})"
        vmin, vmax = 0.0, 1.0

    elif method == "knn":
        k_nn = kwargs.pop("k", 20)
        for i in range(n):
            for j in range(i + 1, n):
                _, mean_j = tl.compute_knn_overlap(adata, keys[i], keys[j], k=k_nn)
                mat[i, j] = mean_j
                mat[j, i] = mean_j
        default_title = f"KNN overlap (Jaccard, k={k_nn})"
        vmin, vmax = 0.0, 1.0

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: cosine_correlation, ari, knn.")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat, annot=True, fmt=".3f", cmap=cmap, center=0 if method == "cosine_correlation" else None,
        xticklabels=display_labels, yticklabels=display_labels,
        ax=ax, vmin=vmin, vmax=vmax, square=True,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(title or default_title, fontsize=13)
    fig.tight_layout()
    return fig


def cluster_property_heatmap(
    adata: AnnData,
    cluster_key: str,
    properties: list[str],
    z_score: bool = True,
    top_n_clusters: int | None = 10,
    figsize: tuple[float, float] = (14, 6),
    cmap: str = "RdBu_r",
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Heatmap of mean property values per cluster.

    Parameters
    ----------
    adata
        AnnData with cluster labels and numeric properties in ``obs``.
    cluster_key
        Column in ``obs`` with cluster labels.
    properties
        Columns in ``obs`` with numeric values to summarize.
    z_score
        Whether to z-score each property across clusters.
    top_n_clusters
        Only show the *N* largest clusters.  ``None`` shows all.
    figsize
        Figure size.
    cmap
        Colormap.
    title
        Plot title.
    **kwargs
        Passed to ``seaborn.heatmap``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"'{cluster_key}' not in adata.obs.")

    valid_props = [p for p in properties if p in adata.obs.columns]
    if not valid_props:
        raise ValueError("None of the requested properties are in adata.obs.")

    means = adata.obs.groupby(cluster_key, observed=True)[valid_props].mean()
    if top_n_clusters is not None:
        top = adata.obs[cluster_key].value_counts().head(top_n_clusters).index
        means = means.loc[means.index.isin(top)]

    if z_score:
        mu = means.mean()
        sd = means.std()
        sd[sd == 0] = 1.0
        display = (means - mu) / sd
    else:
        display = means

    fig, ax = plt.subplots(figsize=figsize)
    hm_kw: dict[str, Any] = {
        "annot": True, "fmt": ".1f", "linewidths": 0.5, "linecolor": "white",
    }
    hm_kw.update(kwargs)
    sns.heatmap(
        display.T, cmap=cmap, center=0 if z_score else None, ax=ax,
        yticklabels=[p.replace("_", " ").title() for p in display.columns],
        **hm_kw,
    )
    ax.set_xlabel("Cluster")
    ax.set_title(title or f"Property {'z-scores' if z_score else 'means'} per cluster ({cluster_key})")
    fig.tight_layout()
    return fig
