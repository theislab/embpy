"""Similarity, distance, perturbation ranking, and pseudobulk aggregation tools."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors


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

    sim = embedding_similarity_matrix(X, metric=metric)

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

    jaccard, mean_j = knn_jaccard(X_a, X_b, k=k)

    col_name = f"knn_jaccard_{obsm_key_a}_{obsm_key_b}"
    adata.obs[col_name] = jaccard
    logging.info(
        "KNN overlap (k=%d) between '%s' and '%s': mean Jaccard = %.4f",
        k,
        obsm_key_a,
        obsm_key_b,
        mean_j,
    )
    return jaccard, mean_j


def embedding_similarity_matrix(matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Compute pairwise row similarity for an embedding matrix."""
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embedding_similarity_matrix: matrix must be 2D, got shape {X.shape!r}.")
    if metric == "cosine":
        return cosine_similarity(X)
    if metric in ("pearson", "correlation"):
        return np.corrcoef(X)
    if metric == "spearman":
        n = X.shape[0]
        sim = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = spearmanr(X[i], X[j])
                sim[i, j] = rho
                sim[j, i] = rho
        return sim
    raise ValueError(f"Unknown similarity metric '{metric}'. Choose from: cosine, pearson, spearman.")


def aggregate_embedding_table(
    matrix: np.ndarray,
    labels: Sequence[str],
    reducer: str = "mean",
) -> pd.DataFrame:
    """Aggregate embedding rows into group centroids or summaries."""
    X = np.asarray(matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"aggregate_embedding_table: matrix must be 2D, got shape {X.shape!r}.")
    group_labels = [str(x) for x in labels]
    if len(group_labels) != X.shape[0]:
        raise ValueError(
            f"aggregate_embedding_table: labels length {len(group_labels)} does not match {X.shape[0]} rows."
        )
    frame = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(X.shape[1])])
    frame["_group"] = group_labels
    grouped = frame.groupby("_group", sort=True)
    if reducer == "mean":
        out = grouped.mean(numeric_only=True)
    elif reducer == "median":
        out = grouped.median(numeric_only=True)
    elif reducer == "sum":
        out = grouped.sum(numeric_only=True)
    else:
        raise ValueError(f"aggregate_embedding_table: reducer must be mean, median, or sum; got {reducer!r}.")
    out.index.name = "group"
    return out


def similarity_correlation(
    matrix_a: np.ndarray,
    *,
    matrix_b: np.ndarray | None = None,
    phenotype: Sequence[float] | np.ndarray | None = None,
    metric: str = "cosine",
    label_a: str = "embedding_a",
    target: str = "target",
) -> pd.DataFrame:
    """Correlate embedding pairwise similarity with another space or phenotype."""
    if matrix_b is None and phenotype is None:
        raise ValueError("similarity_correlation: pass matrix_b or phenotype.")
    sim_a = embedding_similarity_matrix(matrix_a, metric=metric)
    iu = np.triu_indices(sim_a.shape[0], k=1)
    x = sim_a[iu]
    if matrix_b is not None:
        B = np.asarray(matrix_b, dtype=np.float32)
        if B.shape[0] != sim_a.shape[0]:
            raise ValueError(
                f"similarity_correlation: matrix_b has {B.shape[0]} rows but matrix_a has {sim_a.shape[0]}."
            )
        y = embedding_similarity_matrix(B, metric=metric)[iu]
    else:
        values = np.asarray(phenotype, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != sim_a.shape[0]:
            raise ValueError(
                "similarity_correlation: phenotype must be 1D with one value per row "
                f"(got shape {values.shape!r}, expected {sim_a.shape[0]})."
            )
        if not np.isfinite(values).all():
            raise ValueError("similarity_correlation: phenotype contains NaN/Inf values.")
        y = -np.abs(values[:, None] - values[None, :])[iu]

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pd.DataFrame(
        [
            {
                "embedding": label_a,
                "target": target,
                "metric": metric,
                "pearson": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "n_pairs": int(len(x)),
            }
        ]
    )


def knn_jaccard(
    matrix_a: np.ndarray, matrix_b: np.ndarray, k: int = 15, metric: str = "cosine"
) -> tuple[np.ndarray, float]:
    """Compute per-row and mean KNN Jaccard overlap between two matrices."""
    X_a = np.asarray(matrix_a, dtype=np.float32)
    X_b = np.asarray(matrix_b, dtype=np.float32)
    if X_a.ndim != 2 or X_b.ndim != 2:
        raise ValueError(f"knn_jaccard: matrices must be 2D, got {X_a.shape!r} and {X_b.shape!r}.")
    if X_a.shape[0] != X_b.shape[0]:
        raise ValueError("knn_jaccard: matrices must have the same number of rows.")
    if X_a.shape[0] < 2:
        raise ValueError("knn_jaccard: at least two rows are required.")

    actual_k = min(int(k), X_a.shape[0] - 1)
    nn_a = NearestNeighbors(n_neighbors=actual_k + 1, metric=metric).fit(X_a)
    nn_b = NearestNeighbors(n_neighbors=actual_k + 1, metric=metric).fit(X_b)
    idx_a = nn_a.kneighbors(X_a, return_distance=False)[:, 1:]
    idx_b = nn_b.kneighbors(X_b, return_distance=False)[:, 1:]

    jaccard = np.zeros(X_a.shape[0], dtype=np.float64)
    for i, (a_row, b_row) in enumerate(zip(idx_a, idx_b, strict=True)):
        set_a = set(a_row)
        set_b = set(b_row)
        union = len(set_a | set_b)
        jaccard[i] = len(set_a & set_b) / union if union else 0.0
    return jaccard, float(jaccard.mean())


def compare_embedding_matrices(
    embeddings: Mapping[str, np.ndarray],
    *,
    metric: str = "cosine",
    k: int = 15,
) -> pd.DataFrame:
    """Compare multiple embedding spaces with similarity correlation and KNN overlap."""
    keys = list(embeddings)
    if len(keys) < 2:
        raise ValueError("compare_embedding_matrices: pass at least two embeddings.")
    rows: list[dict[str, float | str]] = []
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            corr = similarity_correlation(
                embeddings[key_a],
                matrix_b=embeddings[key_b],
                metric=metric,
                label_a=key_a,
                target=key_b,
            ).iloc[0]
            _per_row, mean_j = knn_jaccard(embeddings[key_a], embeddings[key_b], k=k)
            rows.append(
                {
                    "embedding_a": key_a,
                    "embedding_b": key_b,
                    "metric": metric,
                    "similarity_pearson": float(corr["pearson"]),
                    "similarity_spearman": float(corr["spearman"]),
                    "knn_jaccard": mean_j,
                }
            )
    return pd.DataFrame(rows)


def nearest_neighbors_table(
    matrix: np.ndarray,
    ids: Sequence[str],
    query: str | Sequence[float] | np.ndarray | None = None,
    k: int = 10,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Return nearest neighbors as a tidy table.

    This is the tabular counterpart to :func:`embpy.tl.find_nearest_neighbors`,
    which computes and stores a scanpy neighbor graph in AnnData. It accepts a
    raw matrix plus ids so higher-level semantic layers such as ``adata.embpy``
    can reuse the same implementation for AnnData-aligned embeddings and
    external store embeddings.
    """
    X = np.asarray(matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"nearest_neighbors_table: matrix must be 2D, got shape {X.shape!r}.")
    names = tuple(str(x) for x in ids)
    if len(names) != X.shape[0]:
        raise ValueError(f"nearest_neighbors_table: ids length {len(names)} does not match {X.shape[0]} rows.")
    if X.shape[0] < 2:
        raise ValueError("nearest_neighbors_table: at least two rows are required.")

    actual_k = min(int(k), X.shape[0] - 1)
    query_is_existing_row = isinstance(query, str)
    n_neighbors = actual_k + 1 if query is None or query_is_existing_row else actual_k
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X)

    if query is None:
        distances, indices = nn.kneighbors(X)
        rows = []
        for i, source_id in enumerate(names):
            rank = 0
            for dist, idx in zip(distances[i], indices[i], strict=True):
                if idx == i:
                    continue
                rank += 1
                rows.append(_neighbor_row(source_id, names[idx], rank, dist, metric))
        return pd.DataFrame(rows)

    query_id = "query"
    if isinstance(query, str):
        if query not in names:
            raise KeyError(f"nearest_neighbors_table: query {query!r} not found in ids.")
        query_id = query
        query_vec = X[names.index(query)].reshape(1, -1)
    else:
        query_vec = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query_vec.shape[1] != X.shape[1]:
            raise ValueError(
                f"nearest_neighbors_table: query has {query_vec.shape[1]} dims but matrix has {X.shape[1]}."
            )

    distances, indices = nn.kneighbors(query_vec)
    rows = []
    rank = 0
    for dist, idx in zip(distances[0], indices[0], strict=True):
        if query_is_existing_row and names[idx] == query_id:
            continue
        rank += 1
        rows.append(_neighbor_row(query_id, names[idx], rank, dist, metric))
    return pd.DataFrame(rows)


def nearest_neighbors(
    adata: AnnData,
    obsm_key: str,
    query: str | Sequence[float] | np.ndarray | None = None,
    k: int = 10,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Return nearest neighbors from an AnnData ``.obsm`` embedding as a table."""
    return nearest_neighbors_table(
        _get_embedding(adata, obsm_key), tuple(adata.obs_names), query=query, k=k, metric=metric
    )


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


def _neighbor_row(
    source_id: str, neighbor_id: str, rank: int, distance: float, metric: str
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "query_id": source_id,
        "neighbor_id": neighbor_id,
        "rank": int(rank),
        "distance": float(distance),
    }
    if metric == "cosine":
        row["similarity"] = float(1.0 - distance)
    return row


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
    ...     adata,
    ...     group_col="gene",
    ...     obsm_key="X_emb",
    ... )
    >>> tl.rank_perturbations(gene_adata, "TP53", obsm_key="X_emb")
    """
    import scanpy as sc

    if group_col not in adata.obs.columns:
        raise KeyError(f"'{group_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    if label_col is not None and label_col not in adata.obs.columns:
        raise KeyError(f"label_col '{label_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

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
        first_labels = adata.obs.groupby(group_col, sort=False)[label_col].first()
        first_labels.index = first_labels.index.astype(str)
        agg.obs[label_col] = agg.obs_names.map(first_labels).values

    logging.info(
        "Pseudobulk: %d observations -> %d groups (column '%s').",
        adata.n_obs,
        agg.n_obs,
        group_col,
    )
    return agg
