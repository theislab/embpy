"""Fusion of multiple representations of the same entities.

Feature operators expose sklearn-style fit/transform and return EmbeddingResult.
Late-fusion helpers operate on labelled score tables and return score tables.
Kernel and linear methods require no neural framework. Neural methods import
the optional PyTorch dependency only when fitted.
"""

from ._base import BaseFusion, FusionConfig
from ._views import align_views, subset_views, view_from_store, views_from_anndata
from .cca import CCAFusion
from .concatenation import ConcatFusion
from .ensemble import fuse_ranks, fuse_scores
from .gcca import GCCAFusion
from .kernel import KernelFusion
from .linear import PCAFusion
from .neural import ConcatMLPFusion, DeepSetsFusion, GatedFusion
from .procrustes import ProcrustesFusion

__all__ = [
    "BaseFusion",
    "FusionConfig",
    "CCAFusion",
    "GCCAFusion",
    "ProcrustesFusion",
    "ConcatFusion",
    "PCAFusion",
    "KernelFusion",
    "DeepSetsFusion",
    "GatedFusion",
    "ConcatMLPFusion",
    "align_views",
    "fuse_ranks",
    "fuse_scores",
    "view_from_store",
    "views_from_anndata",
    "subset_views",
]
