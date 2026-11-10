"""Supervised concatenation control for nonlinear fusion benchmarks."""

from ._base import NeuralFusion


class ConcatMLPFusion(NeuralFusion):
    """Concatenate preprocessed views → MLP → supervised representation.

    Use alongside DeepSets/GatedFusion to control for nonlinear capacity and
    access to targets. A one-view configuration is a single-view neural control.
    Parameter counts are reported; equal widths do not imply equal capacity.
    """

    aggregation = "concat"
