"""Fusion of multiple representations of the same entities.

Feature operators expose sklearn-style fit/transform and return EmbeddingResult.
Late-fusion helpers operate on labelled score tables and return score tables.
No neural framework or pretrained model is required.
"""

from ._base import BaseFusion, FusionConfig
from ._views import align_views, subset_views, view_from_store, views_from_anndata
from .cca import CCAFusion
from .concatenation import ConcatFusion
from .ensemble import fuse_ranks, fuse_scores
from .gcca import GCCAFusion
from .linear import PCAFusion
from .procrustes import ProcrustesFusion

__all__ = [
    "BaseFusion",
    "FusionConfig",
    "CCAFusion",
    "GCCAFusion",
    "ProcrustesFusion",
    "ConcatFusion",
    "PCAFusion",
    "align_views",
    "fuse_ranks",
    "fuse_scores",
    "view_from_store",
    "views_from_anndata",
    "subset_views",
]
