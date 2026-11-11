"""Optional PyTorch fusion; importing these estimators does not import torch."""

from .concat import ConcatMLPFusion
from .deepsets import DeepSetsFusion
from .gated import GatedFusion

__all__ = ["ConcatMLPFusion", "DeepSetsFusion", "GatedFusion"]
