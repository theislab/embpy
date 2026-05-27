"""Similarity, distance, perturbation ranking, and pseudobulk aggregation tools."""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _get_embedding(adata: AnnData, obsm_key: str) -> np.ndarray:
    """Extract an embedding matrix from *adata.obsm* with validation."""
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array in adata.obsm['{obsm_key}'], got shape {X.shape}.")
    return X


def compute_similarity(
    adata: AnnData,
    obsm_key: str,
    metric: str = "cosine",
    *,
    plot: bool = False,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "RdBu_r",
    show: bool = True,
    **plot_kwargs,
) -> np.ndarray:
    """Compute pairwise similarity between perturbation embeddings.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    metric
        Similarity metric: ``"cosine"``, ``"pearson"``,
        ``"spearman"``, or ``"correlation"`` (alias for pearson).
    plot
        If ``True``, render an inline heatmap via
        :func:`embpy.pl.plot_similarity_heatmap`.
    labels
        Row / column labels for the heatmap.  Defaults to
        ``adata.obs_names``.
    title
        Heatmap title.
    figsize
        Figure size passed to matplotlib.
    cmap
        Colormap name.
    show
        Whether to call ``plt.show()`` (passed through to the plot
        function).
    **plot_kwargs
        Extra keyword arguments forwarded to
        :func:`embpy.pl.plot_similarity_heatmap`.

    Returns
    -------
    Square similarity matrix of shape ``(n_obs, n_obs)``.
    """
    X = _get_embedding(adata, obsm_key)

    if metric == "cosine":
        sim = cosine_similarity(X)
    elif metric in ("pearson", "correlation"):
        sim = np.corrcoef(X)
    elif metric == "spearman":
        n = X.shape[0]
        sim = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = spearmanr(X[i], X[j])
                sim[i, j] = rho
                sim[j, i] = rho
    else:
        raise ValueError(f"Unknown similarity metric '{metric}'. Choose from: cosine, pearson, spearman.")

    if plot:
        from embpy.pl import plot_similarity_heatmap

        if labels is None:
            labels = list(adata.obs_names)
        plot_similarity_heatmap(
            sim,
            labels=labels,
            title=title,
            figsize=figsize,
            cmap=cmap,
            **plot_kwargs,
        )
        if show:
            import matplotlib.pyplot as _plt

            _plt.show()

    return sim


def compute_distance_matrix(
    adata: AnnData,
    obsm_key: str,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute pairwise distance matrix between embeddings.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    metric
        Distance metric: ``"euclidean"``, ``"cosine"``, ``"wasserstein"``.

    Returns
    -------
    Square distance matrix of shape ``(n_obs, n_obs)``.
    """
    X = _get_embedding(adata, obsm_key)

    if metric == "euclidean":
        return euclidean_distances(X)
    if metric == "cosine":
        return 1.0 - cosine_similarity(X)
    if metric == "wasserstein":
        n = X.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = wasserstein_distance(X[i], X[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    raise ValueError(f"Unknown distance metric '{metric}'. Choose from: euclidean, cosine, wasserstein.")


def compute_knn_overlap(
    adata: AnnData,
    obsm_key_a: str,
    obsm_key_b: str,
    k: int = 15,
) -> tuple[np.ndarray, float]:
    """Compute per-observation KNN Jaccard overlap between two embedding spaces.

    Parameters
    ----------
    adata
        AnnData containing both embeddings.
    obsm_key_a, obsm_key_b
        Embedding keys.
    k
        Number of nearest neighbors.

    Returns
    -------
    Tuple of ``(per_obs_jaccard, mean_jaccard)``.
    """
    X_a = _get_embedding(adata, obsm_key_a)
    X_b = _get_embedding(adata, obsm_key_b)

    if X_a.shape[0] != X_b.shape[0]:
        raise ValueError("Embedding matrices must have the same number of observations.")

    actual_k = min(k, X_a.shape[0] - 1)

    nn_a = NearestNeighbors(n_neighbors=actual_k + 1, metric="cosine").fit(X_a)
    nn_b = NearestNeighbors(n_neighbors=actual_k + 1, metric="cosine").fit(X_b)

    idx_a = nn_a.kneighbors(X_a, return_distance=False)[:, 1:]
    idx_b = nn_b.kneighbors(X_b, return_distance=False)[:, 1:]

    n = X_a.shape[0]
    jaccard = np.zeros(n, dtype=np.float64)
    for i in range(n):
        set_a = set(idx_a[i])
        set_b = set(idx_b[i])
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard[i] = intersection / union if union > 0 else 0.0

    col_name = f"knn_jaccard_{obsm_key_a}_{obsm_key_b}"
    adata.obs[col_name] = jaccard
    mean_j = float(jaccard.mean())
    logging.info(
        "KNN overlap (k=%d) between '%s' and '%s': mean Jaccard = %.4f",
        k, obsm_key_a, obsm_key_b, mean_j,
    )
    return jaccard, mean_j


def rank_perturbations(
    adata: AnnData,
    query: str | np.ndarray,
    obsm_key: str,
    top_k: int = 10,
    metric: str = "cosine",
) -> list[tuple[str, float]]:
    """Rank perturbations by similarity to a query embedding.

    Parameters
    ----------
    adata
        AnnData with perturbation embeddings in ``obsm[obsm_key]``.
    query
        Perturbation identifier (``str``) or raw embedding vector.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    top_k
        Number of top results.
    metric
        ``"cosine"`` or ``"pearson"``.

    Returns
    -------
    List of ``(perturbation_id, similarity_score)`` tuples, descending.
    """
    X = _get_embedding(adata, obsm_key)

    if isinstance(query, str):
        if query not in adata.obs_names:
            raise KeyError(f"Query '{query}' not found in adata.obs_names.")
        idx = list(adata.obs_names).index(query)
        q_vec = X[idx].reshape(1, -1)
    else:
        q_vec = np.asarray(query, dtype=np.float64).reshape(1, -1)

    if metric == "cosine":
        sims = cosine_similarity(q_vec, X).ravel()
    elif metric in ("pearson", "correlation"):
        sims = np.array([pearsonr(q_vec.ravel(), X[i])[0] for i in range(X.shape[0])])
    else:
        raise ValueError(f"Unknown ranking metric '{metric}'.")

    order = np.argsort(sims)[::-1]
    names = list(adata.obs_names)
    return [(names[i], float(sims[i])) for i in order[:top_k]]


def cross_modal_mantel(
    adata: AnnData,
    obsm_key_a: str,
    obsm_key_b: str,
    metric: str = "cosine",
) -> tuple[float, float]:
    """Mantel test: Spearman correlation between two pairwise distance matrices.

    A non-zero correlation indicates that observations that are far apart in
    space *A* also tend to be far apart in space *B*.  Used to quantify
    cross-modal geometric alignment (e.g. protein embedding vs. transcriptomic
    phenotype space) without requiring the two spaces to have the same
    dimensionality.

    Parameters
    ----------
    adata
        AnnData containing both embeddings.
    obsm_key_a, obsm_key_b
        Keys in ``.obsm`` for the two embedding spaces.
    metric
        Distance metric used to build both matrices.
        ``"cosine"`` (default) or ``"euclidean"``.

    Returns
    -------
    Tuple of ``(rho, p_value)`` — Spearman correlation between the upper
    triangles of the two distance matrices and the corresponding two-sided
    p-value.

    Examples
    --------
    >>> rho, p = tl.cross_modal_mantel(adata_pb, "X_esm2_650M", "X_scgpt")
    >>> print(f"Mantel ρ = {rho:.3f}  (p = {p:.2e})")
    """
    X_a = _get_embedding(adata, obsm_key_a)
    X_b = _get_embedding(adata, obsm_key_b)

    if X_a.shape[0] != X_b.shape[0]:
        raise ValueError(
            f"Embeddings must have the same number of observations; "
            f"got {X_a.shape[0]} for '{obsm_key_a}' and {X_b.shape[0]} for '{obsm_key_b}'."
        )
    n = X_a.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 observations for a Mantel test, got {n}.")

    if metric == "cosine":
        dist_a = 1.0 - cosine_similarity(X_a.astype(np.float64))
        dist_b = 1.0 - cosine_similarity(X_b.astype(np.float64))
    elif metric == "euclidean":
        dist_a = euclidean_distances(X_a.astype(np.float64))
        dist_b = euclidean_distances(X_b.astype(np.float64))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: cosine, euclidean.")

    triu = np.triu_indices(n, k=1)
    rho, p = spearmanr(dist_a[triu], dist_b[triu])
    rho, p = float(rho), float(p)

    logger.info(
        "Mantel test ('%s' vs '%s', metric=%s, n=%d): rho=%.4f, p=%.4e",
        obsm_key_a, obsm_key_b, metric, n, rho, p,
    )
    return rho, p


def pseudobulk_embeddings(
    adata: AnnData,
    group_col: str,
    obsm_key: str | None = None,
    label_col: str | None = None,
) -> AnnData:
    """Aggregate replicate observations into per-group mean embeddings.

    Wraps :func:`scanpy.get.aggregate` to compute per-group mean
    embeddings -- the standard "pseudobulk" operation used to go from
    well-level to perturbation-level profiles.

    Parameters
    ----------
    adata
        AnnData with embedding vectors.  Embeddings are read from
        ``obsm[obsm_key]`` when *obsm_key* is given, otherwise from ``.X``.
    group_col
        Column in ``adata.obs`` to group by (e.g. ``"gene"``,
        ``"Metadata_JCP2022"``).  The unique values become the
        ``obs_names`` of the returned object.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.  If ``None``
        (default), ``.X`` is used.
    label_col
        Optional column in ``adata.obs`` whose first-per-group value is
        carried into the returned ``.obs``.  Useful for keeping a
        human-readable label alongside an opaque group identifier
        (e.g. ``label_col="gene"`` when grouping by JCP2022 IDs).

    Returns
    -------
    AnnData
        One row per group.  Mean embeddings are stored in ``.X`` and,
        when *obsm_key* is not ``None``, also in ``.obsm[obsm_key]``.
        ``obs_names`` are the unique group values.

    Examples
    --------
    >>> import embpy.tl as tl
    >>> gene_adata = tl.pseudobulk_embeddings(
    ...     adata, group_col="gene", obsm_key="X_emb",
    ... )
    >>> tl.rank_perturbations(gene_adata, "TP53", obsm_key="X_emb")
    """
    import scanpy as sc

    if group_col not in adata.obs.columns:
        raise KeyError(
            f"'{group_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if label_col is not None and label_col not in adata.obs.columns:
        raise KeyError(
            f"label_col '{label_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if obsm_key is not None:
        X = _get_embedding(adata, obsm_key)
        work = AnnData(X=X, obs=adata.obs.copy())
    else:
        work = adata

    agg = sc.get.aggregate(work, by=group_col, func="mean")

    # scanpy >= 1.11 stores the result in layers["mean"] and sets X=None;
    # older versions put it directly in X.
    raw = agg.layers.get("mean") if agg.X is None else agg.X
    dense = np.asarray(
        raw.toarray() if hasattr(raw, "toarray") else raw,
        dtype=np.float32,
    )

    if group_col in agg.obs.columns:
        agg.obs_names = agg.obs[group_col].astype(str).values
    agg.obs.index.name = group_col

    if obsm_key is not None:
        agg.obsm[obsm_key] = dense.astype(np.float64)

    agg.X = dense

    if label_col is not None:
        first_labels = (
            adata.obs
            .groupby(group_col, sort=False)[label_col]
            .first()
        )
        first_labels.index = first_labels.index.astype(str)
        agg.obs[label_col] = agg.obs_names.map(first_labels).values

    logging.info(
        "Pseudobulk: %d observations -> %d groups (column '%s').",
        adata.n_obs, agg.n_obs, group_col,
    )
    return agg
