"""Compound-dependent softmax weighting of projected views."""

import pandas as pd

from ._base import NeuralFusion


class GatedFusion(NeuralFusion):
    """Supervised view encoders with a shared scalar gate and softmax pooling.

    The gate sees projected input views only, never MoA labels or test targets.
    Its weights are model diagnostics, not causal feature importance. Training
    parameters and the fixed-epoch protocol are documented by NeuralFusion.
    """

    aggregation = "gated"

    def view_weights(self, X):
        """Return nonnegative per-entity view weights summing to one."""
        aligned, _ = self._align_transform(X)
        return pd.DataFrame(
            self._infer(aligned, "weights"), index=next(iter(aligned.values())).entity_ids, columns=self.view_names_
        )
