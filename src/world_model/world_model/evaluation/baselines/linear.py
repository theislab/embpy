"""Ridge-regression baseline.

Two ways to parameterise the linear map are supported via the
``target`` argument to the constructor:

* ``"delta"`` (default) -- regress action embedding -> per-gene
  perturbation delta. Final prediction is ``state + W * action_emb``.
  Recommended: it only has to learn the perturbation effect, not the
  basal state.
* ``"next_state"`` -- regress (state, action embedding) -> next state
  directly. More parameters, no inductive bias.

Implementation notes
--------------------
* Default backend: :class:`sklearn.linear_model.Ridge` with ``alpha=1.0``.
  Set ``alpha=0`` for unregularised least-squares. Pass
  ``solver="lstsq"`` to bypass sklearn entirely (numpy lstsq fallback).
* Action embeddings are looked up by perturbation label via the
  mapping the user passes in :class:`BaselineTrainData.perturbation_to_action`.
  Labels that are missing from the mapping fall back to identity.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Baseline, BaselineTrainData

logger = logging.getLogger(__name__)


class LinearRegressionBaseline(Baseline):
    """Ridge regression in either delta- or next-state space."""

    name = "linear"

    def __init__(self, alpha: float = 1.0, target: str = "delta", solver: str = "auto") -> None:
        super().__init__()
        if target not in {"delta", "next_state"}:
            raise ValueError(f"target must be 'delta' or 'next_state', got {target!r}")
        self.alpha = float(alpha)
        self.target = target
        self.solver = solver

    def fit(self, data: BaselineTrainData) -> None:
        if data.perturbation_to_action is None:
            raise ValueError(
                "LinearRegressionBaseline.fit requires perturbation_to_action embeddings."
            )
        train = data.expression[data.train_indices]
        labels = data.perturbation_labels[data.train_indices]
        is_control = labels == data.control_label
        if not is_control.any():
            raise ValueError("No control cells in train split for LinearRegressionBaseline.")
        control_mean = train[is_control].mean(axis=0).astype(np.float32)
        self.control_mean_train_: np.ndarray = control_mean
        self._control_label = data.control_label
        # Some embedding catalogs carry NaN by design (e.g. DepMap
        # crispr_gene_effect: cell lines never profiled for a given gene).
        # Ridge / lstsq both reject NaN. Compute a per-feature imputation
        # vector (column mean over all rows that have finite values for
        # that column; 0 when the entire column is NaN) and substitute it
        # everywhere a NaN appears, both at fit and predict time.
        raw_actions = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in data.perturbation_to_action.items()
        }
        self._action_dim = next(iter(raw_actions.values())).size
        stacked = np.stack(list(raw_actions.values()), axis=0)
        if not np.isfinite(stacked).all():
            n_nan = int((~np.isfinite(stacked)).sum())
            col_mean = np.nanmean(stacked, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
            self._action_impute: np.ndarray = col_mean
            for k, v in raw_actions.items():
                mask = ~np.isfinite(v)
                if mask.any():
                    raw_actions[k] = np.where(mask, col_mean, v).astype(np.float32)
            logger.warning(
                "LinearRegressionBaseline: imputed %d NaN values across %d "
                "perturbations (column-mean) before fitting Ridge. This is "
                "expected for embeddings derived from sparse cell-line panels "
                "(e.g. crispr_gene_effect).",
                n_nan, len(raw_actions),
            )
        else:
            self._action_impute = np.zeros(self._action_dim, dtype=np.float32)
        self._actions = raw_actions

        rows_x: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        for p, emb in self._actions.items():
            mask = labels == p
            if not mask.any():
                continue
            mean_p = train[mask].mean(axis=0).astype(np.float32)
            if self.target == "delta":
                target = mean_p - control_mean
                feature = emb
            else:
                target = mean_p
                feature = np.concatenate([control_mean, emb], axis=0)
            rows_x.append(feature)
            rows_y.append(target)

        if not rows_x:
            raise ValueError(
                "LinearRegressionBaseline: no overlap between train perturbations "
                "and perturbation_to_action keys."
            )

        X = np.stack(rows_x, axis=0).astype(np.float32)
        Y = np.stack(rows_y, axis=0).astype(np.float32)

        use_sklearn = self.solver != "lstsq"
        if use_sklearn:
            try:
                from sklearn.linear_model import Ridge  # noqa: PLC0415

                model = Ridge(alpha=self.alpha)
                model.fit(X, Y)
                self.coef_: np.ndarray = model.coef_.astype(np.float32)
                self.intercept_: np.ndarray = np.asarray(model.intercept_, dtype=np.float32)
            except ImportError:
                logger.warning("scikit-learn missing; falling back to numpy lstsq.")
                use_sklearn = False

        if not use_sklearn:
            X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
            sol, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
            self.coef_ = sol[:-1].T.astype(np.float32)
            self.intercept_ = sol[-1].astype(np.float32)

        self._fitted = True
        logger.info(
            "Fitted LinearRegressionBaseline target=%s on %d perturbations (alpha=%.3f)",
            self.target, X.shape[0], self.alpha,
        )

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._require_fit()
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"state ({state.shape[0]}) and action ({action.shape[0]}) row count must match"
            )
        out = state.astype(np.float32, copy=True)
        n_unknown = 0
        for i, lbl in enumerate(action):
            if lbl == self._control_label:
                continue
            emb = self._actions.get(str(lbl))
            if emb is None:
                n_unknown += 1
                continue
            if self.target == "delta":
                feature = emb
                delta = feature @ self.coef_.T + self.intercept_
                out[i] = state[i] + delta
            else:
                feature = np.concatenate([state[i], emb], axis=0)
                out[i] = feature @ self.coef_.T + self.intercept_
        if n_unknown:
            logger.warning(
                "LinearRegressionBaseline: missing action embeddings for %d cells.", n_unknown,
            )
        return out

    def set_action_embeddings(self, mapping: dict[str, np.ndarray]) -> None:
        """Extend / override the perturbation-to-embedding lookup at predict time.

        Useful when the test split contains perturbations not present in
        :class:`BaselineTrainData.perturbation_to_action` -- pass the
        full mapping here so every test cell has an embedding to look up.
        """
        impute = getattr(self, "_action_impute", None)
        for k, v in mapping.items():
            emb = np.asarray(v, dtype=np.float32).reshape(-1)
            if impute is not None and not np.isfinite(emb).all():
                mask = ~np.isfinite(emb)
                emb = np.where(mask, impute, emb).astype(np.float32)
            self._actions[str(k)] = emb


__all__ = ["LinearRegressionBaseline"]
