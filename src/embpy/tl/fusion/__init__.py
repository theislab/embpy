"""Stage-1 fusion of multiple representations of the same entities.

Feature operators expose sklearn-style fit/transform and return EmbeddingResult.
Late-fusion helpers operate on labelled score tables and return score tables.
No neural framework or pretrained model is required.
"""

from ._views import align_views, subset_views, view_from_store, views_from_anndata
from .concatenation import ConcatFusion
from .ensemble import fuse_ranks, fuse_scores
from .linear import PCAFusion

__all__ = [
    "ConcatFusion",
    "PCAFusion",
    "align_views",
    "fuse_ranks",
    "fuse_scores",
    "view_from_store",
    "views_from_anndata",
    "subset_views",
]
