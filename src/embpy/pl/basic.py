from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData


def plot_embedding_space(
    adata: AnnData,
    color: str | None = None,
    basis: str = "X_umap",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    **kwargs: Any,
) -> Any:
    """
    Plot perturbation embeddings in 2D reduced space.

    Visualizes UMAP or other dimensionality reduction of perturbation
    embeddings, colored by metadata (e.g., perturbation type, cluster).

    Parameters
    ----------
    adata
        AnnData with reduced coordinates in obsm[basis].
    color
        Column in adata.obs to color points by (e.g., 'perturbation_type', 'cluster').
    basis
        Key in adata.obsm for 2D coordinates. Defaults to 'X_umap'.
    title
        Plot title.
    figsize
        Figure size as (width, height).
    **kwargs
        Additional arguments passed to matplotlib scatter.

    Returns
    -------
    matplotlib Figure or Axes object.
    """
    raise NotImplementedError("plot_embedding_space will be implemented in a future release.")


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Perturbation Similarity",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    **kwargs: Any,
) -> Any:
    """
    Plot a heatmap of pairwise perturbation similarities.

    Parameters
    ----------
    similarity_matrix
        Square matrix of pairwise similarity scores.
    labels
        Labels for rows/columns (perturbation names).
    title
        Plot title.
    figsize
        Figure size.
    cmap
        Colormap name.
    **kwargs
        Additional arguments passed to seaborn heatmap.

    Returns
    -------
    matplotlib Figure or Axes object.
    """
    raise NotImplementedError("plot_similarity_heatmap will be implemented in a future release.")


def plot_perturbation_ranking(
    rankings: list[tuple[str, float]],
    query_label: str = "Query",
    top_k: int = 20,
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Any:
    """
    Bar plot of top-ranked perturbations by similarity to a query.

    Parameters
    ----------
    rankings
        List of (perturbation_id, score) tuples from tl.rank_perturbations.
    query_label
        Label for the query perturbation.
    top_k
        Number of top results to show.
    figsize
        Figure size.
    **kwargs
        Additional arguments passed to matplotlib barh.

    Returns
    -------
    matplotlib Figure or Axes object.
    """
    raise NotImplementedError("plot_perturbation_ranking will be implemented in a future release.")


def plot_cluster_composition(
    adata: AnnData,
    cluster_key: str = "cluster",
    color_by: str = "perturbation_type",
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Any:
    """
    Stacked bar plot showing composition of clusters by perturbation type.

    Parameters
    ----------
    adata
        AnnData with cluster assignments and perturbation metadata.
    cluster_key
        Column in obs with cluster labels.
    color_by
        Column in obs to break down each cluster by.
    figsize
        Figure size.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib Figure or Axes object.
    """
    raise NotImplementedError("plot_cluster_composition will be implemented in a future release.")
