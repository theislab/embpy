"""Identity baseline -- "the perturbation has no effect"."""

from __future__ import annotations

import numpy as np

from .base import Baseline, BaselineTrainData


class IdentityBaseline(Baseline):
    """Returns ``state`` unchanged for every cell.

    This is the trivial floor: any baseline (or model) worth keeping
    must beat this on test perturbations where the response is
    non-zero.
    """

    name = "identity"

    def fit(self, data: BaselineTrainData) -> None:
        self._n_genes = int(data.expression.shape[1])
        self._fitted = True

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._require_fit()
        if state.ndim != 2 or state.shape[1] != self._n_genes:
            raise ValueError(
                f"state must be (n_cells, {self._n_genes}), got {state.shape}"
            )
        if action.shape[0] != state.shape[0]:
            raise ValueError(
                f"action ({action.shape[0]}) and state ({state.shape[0]}) row count must match"
            )
        return state.astype(np.float32, copy=True)


__all__ = ["IdentityBaseline"]
