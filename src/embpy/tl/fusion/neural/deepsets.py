"""Permutation-invariant aggregation after named view projections."""

from ._base import NeuralFusion


class DeepSetsFusion(NeuralFusion):
    """Supervised DeepSets-style fusion: named adapters → shared phi → mean → rho.

    View widths may differ. Named adapters keep view identity, while pooling
    is invariant to their order. This does not support arbitrary unseen views
    or missing views. See NeuralFusion for training parameters and target use.
    """

    aggregation = "mean"
