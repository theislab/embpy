"""Analysis tools for perturbation embeddings.

Provides functions for computing similarity matrices, distance matrices,
clustering, dimensionality reduction (UMAP / t-SNE), KNN overlap, and
perturbation ranking.

Functions that depend on ``scanpy`` perform a lazy import so the package
only needs to be installed when those specific functions are called.
"""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_embedding(adata: AnnData, obsm_key: str) -> np.ndarray:
    """Extract an embedding matrix from *adata.obsm* with validation."""
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array in adata.obsm['{obsm_key}'], got shape {X.shape}.")
    return X


def _require_scanpy():
    """Lazily import scanpy, raising a clear error if missing."""
    try:
        import scanpy as sc

        return sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for this function but is not installed. Install it with:  pip install 'embpy[scanpy]'"
        ) from e


# ---------------------------------------------------------------------------
# Similarity & distance
# ---------------------------------------------------------------------------


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
        Similarity metric.  One of ``"cosine"``, ``"pearson"``,
        ``"spearman"``, ``"correlation"`` (alias for pearson).

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
        Distance metric.  One of ``"euclidean"``, ``"cosine"``,
        ``"wasserstein"``.

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


# ---------------------------------------------------------------------------
# Nearest neighbours & KNN overlap
# ---------------------------------------------------------------------------


def find_nearest_neighbors(
    adata: AnnData,
    obsm_key: str,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> AnnData:
    """Compute nearest-neighbor graph on perturbation embeddings.

    Uses scanpy to build the graph and stores the results in
    ``adata.obsp['distances']`` and ``adata.obsp['connectivities']``.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_neighbors
        Number of nearest neighbors.
    metric
        Distance metric for neighbor computation.

    Returns
    -------
    AnnData with neighbor graph stored in ``obsp``.
    """
    sc = _require_scanpy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=obsm_key, metric=metric)  # type: ignore[arg-type]
    logging.info(
        "Computed %d-nearest-neighbor graph on '%s' (metric=%s).",
        n_neighbors,
        obsm_key,
        metric,
    )
    return adata


def compute_knn_overlap(
    adata: AnnData,
    obsm_key_a: str,
    obsm_key_b: str,
    k: int = 15,
) -> tuple[np.ndarray, float]:
    """Compute per-observation KNN Jaccard overlap between two embedding spaces.

    For each observation, the *k* nearest neighbors are found independently
    in embedding A and embedding B.  The Jaccard index of these two
    neighbor sets is the overlap score for that observation.

    Parameters
    ----------
    adata
        AnnData containing both embeddings.
    obsm_key_a
        First embedding key.
    obsm_key_b
        Second embedding key.
    k
        Number of nearest neighbors.

    Returns
    -------
    Tuple of ``(per_obs_jaccard, mean_jaccard)`` where *per_obs_jaccard*
    is an array of shape ``(n_obs,)`` and *mean_jaccard* is the scalar
    average.
    """
    X_a = _get_embedding(adata, obsm_key_a)
    X_b = _get_embedding(adata, obsm_key_b)

    if X_a.shape[0] != X_b.shape[0]:
        raise ValueError("Embedding matrices must have the same number of observations.")

    actual_k = min(k, X_a.shape[0] - 1)

    nn_a = NearestNeighbors(n_neighbors=actual_k + 1, metric="cosine").fit(X_a)
    nn_b = NearestNeighbors(n_neighbors=actual_k + 1, metric="cosine").fit(X_b)

    idx_a = nn_a.kneighbors(X_a, return_distance=False)[:, 1:]  # exclude self
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
        k,
        obsm_key_a,
        obsm_key_b,
        mean_j,
    )
    return jaccard, mean_j


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def leiden(
    adata: AnnData,
    obsm_key: str,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    key_added: str = "leiden",
) -> AnnData:
    """Run Leiden community detection on an embedding space.

    Convenience wrapper that builds a neighbor graph (if needed) and
    runs Leiden clustering via scanpy.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    resolution
        Resolution parameter for Leiden (higher = more clusters).
    n_neighbors
        Number of neighbors for the graph.
    key_added
        Column name in ``adata.obs`` for the cluster labels.

    Returns
    -------
    AnnData with cluster labels in ``obs[key_added]``.
    """
    sc = _require_scanpy()
    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    n_clusters = adata.obs[key_added].nunique()
    logging.info(
        "Leiden clustering on '%s' (resolution=%.2f): %d clusters, stored in obs['%s'].",
        obsm_key,
        resolution,
        n_clusters,
        key_added,
    )
    return adata


def cluster_embeddings(
    adata: AnnData,
    obsm_key: str,
    method: str = "leiden",
    resolution: float = 1.0,
    n_clusters: int | None = None,
    n_neighbors: int = 15,
    key_added: str = "cluster",
) -> AnnData:
    """Cluster perturbation embeddings and annotate the AnnData object.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    method
        Clustering method: ``"leiden"``, ``"kmeans"``, ``"spectral"``.
    resolution
        Resolution parameter for Leiden clustering.
    n_clusters
        Number of clusters (for k-means / spectral).  Ignored for Leiden.
    n_neighbors
        Number of neighbors for Leiden graph construction.
    key_added
        Column in ``adata.obs`` to store cluster labels.

    Returns
    -------
    AnnData with cluster labels in ``obs[key_added]``.
    """
    X = _get_embedding(adata, obsm_key)

    if method == "leiden":
        return leiden(adata, obsm_key=obsm_key, resolution=resolution, n_neighbors=n_neighbors, key_added=key_added)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        k = n_clusters or 10
        labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
        adata.obs[key_added] = [str(l) for l in labels]
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        logging.info("KMeans clustering (k=%d) stored in obs['%s'].", k, key_added)
        return adata

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        k = n_clusters or 10
        labels = SpectralClustering(n_clusters=k, random_state=0, affinity="nearest_neighbors").fit_predict(X)
        adata.obs[key_added] = [str(l) for l in labels]
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        logging.info("Spectral clustering (k=%d) stored in obs['%s'].", k, key_added)
        return adata

    raise ValueError(f"Unknown clustering method '{method}'. Choose from: leiden, kmeans, spectral.")


# ---------------------------------------------------------------------------
# Dimensionality reduction (UMAP / t-SNE)
# ---------------------------------------------------------------------------


def compute_umap(
    adata: AnnData,
    obsm_key: str,
    n_neighbors: int = 15,
    n_components: int = 2,
    output_key: str | None = None,
) -> AnnData:
    """Compute UMAP coordinates from perturbation embeddings.

    Builds a neighbor graph (via scanpy) and runs UMAP.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_neighbors
        Number of neighbors for graph construction.
    n_components
        Number of UMAP dimensions.
    output_key
        Key for the UMAP coordinates in ``.obsm``.
        Defaults to ``"X_umap_{obsm_key}"``.

    Returns
    -------
    AnnData with UMAP coordinates in ``obsm[output_key]``.
    """
    sc = _require_scanpy()
    out = output_key or f"X_umap_{obsm_key}"
    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata, n_components=n_components)
    adata.obsm[out] = adata.obsm["X_umap"].copy()
    logging.info("UMAP (%d-D) on '%s' stored in obsm['%s'].", n_components, obsm_key, out)
    return adata


def compute_tsne(
    adata: AnnData,
    obsm_key: str,
    n_components: int = 2,
    perplexity: float = 30.0,
    output_key: str | None = None,
) -> AnnData:
    """Compute t-SNE coordinates from perturbation embeddings.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_components
        Number of t-SNE dimensions.
    perplexity
        Perplexity parameter for t-SNE.
    output_key
        Key for the t-SNE coordinates in ``.obsm``.
        Defaults to ``"X_tsne_{obsm_key}"``.

    Returns
    -------
    AnnData with t-SNE coordinates in ``obsm[output_key]``.
    """
    sc = _require_scanpy()
    out = output_key or f"X_tsne_{obsm_key}"
    actual_perp = min(perplexity, adata.n_obs - 1)
    X = np.asarray(adata.obsm[obsm_key])
    n_pcs = min(X.shape[1], X.shape[0] - 1) if X.shape[1] > 50 else 0
    sc.tl.tsne(adata, use_rep=obsm_key, n_pcs=n_pcs if n_pcs > 0 else None, perplexity=actual_perp)
    adata.obsm[out] = adata.obsm["X_tsne"].copy()
    logging.info("t-SNE (%d-D) on '%s' stored in obsm['%s'].", n_components, obsm_key, out)
    return adata


# ---------------------------------------------------------------------------
# Perturbation ranking
# ---------------------------------------------------------------------------


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
        Either a perturbation identifier (``str``) present in
        ``adata.obs_names``, or a raw embedding vector (``np.ndarray``).
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    top_k
        Number of top similar perturbations to return.
    metric
        Similarity metric (``"cosine"`` or ``"pearson"``).

    Returns
    -------
    List of ``(perturbation_id, similarity_score)`` tuples, sorted
    descending by score.
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
    results = [(names[i], float(sims[i])) for i in order[:top_k]]
    return results


# ---------------------------------------------------------------------------
# Placeholder (kept for backwards-compat)
# ---------------------------------------------------------------------------


def basic_tool(adata: AnnData) -> int:
    """Run a tool on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    return 0
