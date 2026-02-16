from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from anndata import AnnData


def compute_similarity(
    adata: AnnData,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise similarity between perturbation embeddings.

    Parameters
    ----------
    adata
        AnnData object where X contains embedding vectors.
    metric
        Distance/similarity metric: "cosine", "euclidean", "correlation".
        Defaults to "cosine".

    Returns
    -------
    Square similarity matrix of shape (n_obs, n_obs).
    """
    raise NotImplementedError("compute_similarity will be implemented in a future release.")


def cluster_embeddings(
    adata: AnnData,
    n_clusters: int | None = None,
    method: str = "leiden",
    resolution: float = 1.0,
) -> AnnData:
    """
    Cluster perturbation embeddings and annotate the AnnData object.

    Adds a 'cluster' column to adata.obs with cluster assignments.
    Optionally computes a UMAP for visualization.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in X.
    n_clusters
        Number of clusters (for k-means). Ignored for graph-based methods.
    method
        Clustering method: "leiden", "kmeans", "spectral".
    resolution
        Resolution parameter for Leiden clustering.

    Returns
    -------
    AnnData with cluster labels in obs['cluster'].
    """
    raise NotImplementedError("cluster_embeddings will be implemented in a future release.")


def rank_perturbations(
    adata: AnnData,
    query: str | np.ndarray | None = None,
    top_k: int = 10,
    metric: str = "cosine",
) -> list[tuple[str, float]]:
    """
    Rank perturbations by similarity to a query embedding.

    Parameters
    ----------
    adata
        AnnData with perturbation embeddings in X.
    query
        Either a perturbation identifier (str) present in adata.obs_names,
        or a raw embedding vector (np.ndarray).
    top_k
        Number of top similar perturbations to return.
    metric
        Similarity metric to use.

    Returns
    -------
    List of (perturbation_id, similarity_score) tuples, sorted descending.
    """
    raise NotImplementedError("rank_perturbations will be implemented in a future release.")


def find_nearest_neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> AnnData:
    """
    Compute nearest-neighbor graph on perturbation embeddings.

    Stores results in adata.obsp['distances'] and adata.obsp['connectivities'].

    Parameters
    ----------
    adata
        AnnData with embedding vectors in X.
    n_neighbors
        Number of nearest neighbors.
    metric
        Distance metric for neighbor computation.

    Returns
    -------
    AnnData with neighbor graph stored in obsp.
    """
    raise NotImplementedError("find_nearest_neighbors will be implemented in a future release.")


def compute_umap(
    adata: AnnData,
    n_components: int = 2,
) -> AnnData:
    """
    Compute UMAP coordinates from perturbation embeddings.

    Stores results in adata.obsm['X_umap'].

    Parameters
    ----------
    adata
        AnnData with embedding vectors in X.
    n_components
        Number of UMAP dimensions.

    Returns
    -------
    AnnData with UMAP coordinates in obsm['X_umap'].
    """
    raise NotImplementedError("compute_umap will be implemented in a future release.")


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
    raise NotImplementedError("basic_tool is a placeholder.")
