"""Global-mean baseline.

For every non-control cell, predict the *mean expression of all
perturbed cells in the train split*. Probes whether perturbations on
average move the cell state in a consistent direction; if a model
fails to beat this, it has not learned anything perturbation-specific.
"""

from __future__ import annotations

import numpy as np

from .base import Baseline, BaselineTrainData


class MeanBaseline(Baseline):
    """Predicts the global perturbed-cell mean for every perturbed cell."""

    name = "mean"

    def fit(self, data: BaselineTrainData) -> None:
        train = data.expression[data.train_indices]
        labels = data.perturbation_labels[data.train_indices]
        is_perturbed = labels != data.control_label
        if not is_perturbed.any():
            raise ValueError("No perturbed cells in train split for MeanBaseline.")
        self.perturbed_mean_: np.ndarray = train[is_perturbed].mean(axis=0).astype(np.float32)
        self._control_label = data.control_label
        self._fitted = True

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._require_fit()
        out = np.broadcast_to(self.perturbed_mean_, state.shape).astype(np.float32, copy=True)
        is_ctrl = action == self._control_label
        if is_ctrl.any():
            out[is_ctrl] = state[is_ctrl]
        return out


__all__ = ["MeanBaseline"]
