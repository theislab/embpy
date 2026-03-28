"""Similarity, distance, and perturbation ranking tools."""

from __future__ import annotations

import logging

import numpy as np
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

    Returns
    -------
    Square similarity matrix of shape ``(n_obs, n_obs)``.
    """
    X = _get_embedding(adata, obsm_key)

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
