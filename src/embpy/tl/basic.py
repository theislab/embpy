"""Backward-compatibility shim.

All functions have been moved to dedicated modules:

- ``embpy.tl.similarity`` -- compute_similarity, compute_distance_matrix,
  compute_knn_overlap, rank_perturbations
- ``embpy.tl.dimred`` -- compute_umap, compute_tsne
- ``embpy.tl.clustering`` -- find_nearest_neighbors, leiden, cluster_embeddings

This module re-exports them so existing ``from embpy.tl.basic import ...``
imports continue to work.
"""

from __future__ import annotations

from anndata import AnnData

from .clustering import cluster_embeddings, find_nearest_neighbors, leiden
from .dimred import compute_tsne, compute_umap
from .similarity import (
    compute_distance_matrix,
    compute_knn_overlap,
    compute_similarity,
    rank_perturbations,
)


def basic_tool(adata: AnnData) -> int:
    """Placeholder kept for backwards-compat."""
    return 0


__all__ = [
    "basic_tool",
    "cluster_embeddings",
    "compute_distance_matrix",
    "compute_knn_overlap",
    "compute_similarity",
    "compute_tsne",
    "compute_umap",
    "find_nearest_neighbors",
    "leiden",
    "rank_perturbations",
]
