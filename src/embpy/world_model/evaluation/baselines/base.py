"""Baseline ABC and the data container they consume.

A baseline operates in gene-expression space:

* ``fit(BaselineTrainData)`` -- learns whatever per-perturbation
  statistics it needs from the *train* split.
* ``predict(state, action) -> next_state`` -- per-cell prediction.
  ``state`` is ``(n_cells, n_genes)``, ``action`` is ``(n_cells,)``
  string array of perturbation labels (each label may be the control
  label, in which case the baseline should return ``state``
  unchanged).

This per-cell signature is what the world model exposes too, so the
evaluation harness can call model and baselines with identical inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineTrainData:
    """Container for whatever a baseline might need at fit time."""

    expression: np.ndarray
    """``(n_cells, n_genes)`` log-normalised expression matrix."""

    perturbation_labels: np.ndarray
    """``(n_cells,)`` string array of perturbation labels."""

    train_indices: np.ndarray
    """Cell indices into ``expression`` belonging to the train split."""

    control_label: str
    gene_symbols: list[str]

    perturbation_to_action: dict[str, np.ndarray] | None = None
    """Optional: pretrained gene-embedding row per perturbation. The
    linear baseline uses this to map perturbation labels to action
    embeddings; mean / control-mean / additive baselines ignore it."""


class Baseline(ABC):
    """Common interface for all baselines.

    Subclasses must implement :meth:`fit` and :meth:`predict`. The
    :attr:`name` attribute (class-level) is used by the comparison
    pipeline to label CSV rows and plot bars.
    """

    name: str = "baseline"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, data: BaselineTrainData) -> None:
        """Learn statistics from the training split."""

    @abstractmethod
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return per-cell predictions.

        Parameters
        ----------
        state
            ``(n_cells, n_genes)`` per-cell input expression.
        action
            ``(n_cells,)`` string array of perturbation labels.

        Returns
        -------
        np.ndarray
            ``(n_cells, n_genes)`` predicted post-perturbation expression.
        """

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} called before fit().")


__all__ = ["Baseline", "BaselineTrainData"]
