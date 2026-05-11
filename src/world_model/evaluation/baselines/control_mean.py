"""Control-mean baseline.

Returns the train-split control mean for every non-control cell.
Distinct from :class:`IdentityBaseline` only in that it ignores the
caller's ``state`` and uses the *training* control mean instead --
matching the convention used in CPA / cell-eval baseline tables.

Cells whose action label equals the control label are returned
unchanged (predicting "no perturbation" for a control cell would be a
contradiction).
"""

from __future__ import annotations

import numpy as np

from .base import Baseline, BaselineTrainData


class ControlMeanBaseline(Baseline):
    """Predicts the train-split control mean for every perturbed cell."""

    name = "control_mean"

    def fit(self, data: BaselineTrainData) -> None:
        train = data.expression[data.train_indices]
        labels = data.perturbation_labels[data.train_indices]
        is_control = labels == data.control_label
        if not is_control.any():
            raise ValueError("No control cells in train split for ControlMeanBaseline.")
        self.control_mean_: np.ndarray = train[is_control].mean(axis=0).astype(np.float32)
        self._control_label = data.control_label
        self._fitted = True

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._require_fit()
        out = np.broadcast_to(self.control_mean_, state.shape).astype(np.float32, copy=True)
        is_ctrl = action == self._control_label
        if is_ctrl.any():
            out[is_ctrl] = state[is_ctrl]
        return out


__all__ = ["ControlMeanBaseline"]
