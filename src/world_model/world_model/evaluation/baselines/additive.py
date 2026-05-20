"""Additive baseline: ``state + delta(perturbation)``.

For every perturbation seen in train, ``delta(p) = mean(perturbed_p) -
mean(controls)``. At predict time the prediction is
``state + delta(action)`` per cell.

When it is meaningful
---------------------
This baseline is the obvious linear-in-perturbation model. It works
well exactly when:

* the perturbation has been observed in training (i.e. the test split
  is *cell-level*, not perturbation-level), or
* the test perturbations come with a known "average effect" direction
  from a related dataset.

When it is NOT meaningful
-------------------------
For the standard *perturbation-level* split (the default in this
code), test perturbations have no observed delta, so this baseline
predicts ``state`` for them (it falls back to the identity baseline).
The class still computes and stores per-perturbation deltas for the
*train* perturbations, so it can be re-used in a held-out-cell setting
without modification.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Baseline, BaselineTrainData

logger = logging.getLogger(__name__)


class AdditiveBaseline(Baseline):
    """``predict(s, a) = s + delta_a`` (or ``s`` if ``delta_a`` unknown)."""

    name = "additive"

    def fit(self, data: BaselineTrainData) -> None:
        train = data.expression[data.train_indices]
        labels = data.perturbation_labels[data.train_indices]
        is_control = labels == data.control_label
        if not is_control.any():
            raise ValueError("No control cells in train split for AdditiveBaseline.")
        control_mean = train[is_control].mean(axis=0).astype(np.float32)

        deltas: dict[str, np.ndarray] = {}
        for label in np.unique(labels):
            if label == data.control_label:
                continue
            mask = labels == label
            if not mask.any():
                continue
            deltas[str(label)] = (train[mask].mean(axis=0) - control_mean).astype(np.float32)
        if not deltas:
            logger.warning("AdditiveBaseline: no non-control perturbations in train.")

        self.deltas_ = deltas
        self.control_mean_train_: np.ndarray = control_mean
        self._control_label = data.control_label
        self._fitted = True

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._require_fit()
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"state ({state.shape[0]}) and action ({action.shape[0]}) row count must match"
            )
        out = state.astype(np.float32, copy=True)
        n_unknown = 0
        n_perturbed = 0
        for i, lbl in enumerate(action):
            if lbl == self._control_label:
                continue
            n_perturbed += 1
            delta = self.deltas_.get(str(lbl))
            if delta is None:
                n_unknown += 1
                continue
            out[i] = state[i] + delta
        if n_perturbed > 0 and n_unknown == n_perturbed:
            logger.warning(
                "AdditiveBaseline: every perturbed cell has an unseen action; "
                "predictions reduce to identity. This is the expected behaviour for "
                "split_by='perturbation' and is documented in the module docstring.",
            )
        return out


__all__ = ["AdditiveBaseline"]
