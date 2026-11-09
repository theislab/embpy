"""Fitted, block-aware concatenation of identifier-aligned embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.validation import check_is_fitted

from ._base import fused_result, id_digest, validate_weights
from ._views import Views, align_views


class ConcatFusion(TransformerMixin, BaseEstimator):
    """Concatenate representations with reusable, train-only preprocessing.

    Parameters
    ----------
    normalization
        ``'none'``, ``'zscore'`` (per feature), or ``'l2'`` (per entity),
        either globally or as an exact view-name mapping. Zero L2 rows stay zero.
    weights
        Fixed view weights. Each block is multiplied by ``sqrt(weight)``.
        With L2-normalized nonzero blocks, cosine similarity of the concatenation
        is the weighted average of the original cosine similarities. None means
        all weights are one; weights are not fitted or silently normalized.
    balance_blocks
        If true, divide each preprocessed block by its training RMS row norm.
        This prevents wider z-scored blocks dominating solely by dimension.
        All-zero training blocks use scale one.
    join
        ``'strict'`` requires equal entity sets; ``'inner'`` intersects them.
        After fitting, transform requires the same named views and dimensions,
        but accepts new entities and a different mapping insertion order.

    Notes
    -----
    Fit/transform never mutate input arrays. Standard sklearn cloning and
    joblib serialization preserve the fitted scales and source-view order.
    """

    def __init__(self, normalization="none", weights=None, balance_blocks=False, join="strict"):
        self.normalization = normalization
        self.weights = weights
        self.balance_blocks = balance_blocks
        self.join = join

    def fit(self, X: Views, y=None):
        """Learn preprocessing statistics from the training entities only."""
        aligned, coverage = align_views(X, join=self.join)
        names = tuple(aligned)
        weights = validate_weights(names, self.weights)
        if isinstance(self.normalization, Mapping):
            if set(self.normalization) != set(names):
                raise ValueError("normalization must name exactly the supplied views.")
            norms = dict(self.normalization)
        else:
            norms = dict.fromkeys(names, self.normalization)
        if any(value not in {"none", "zscore", "l2"} for value in norms.values()):
            raise ValueError("normalization must be 'none', 'zscore', or 'l2'.")
        scalers, scales = {}, {}
        for name, view in aligned.items():
            values = np.asarray(view.matrix, dtype=np.float64)
            scaler = StandardScaler().fit(values) if norms[name] == "zscore" else None
            values = self._normalize(values, norms[name], scaler)
            rms = float(np.sqrt(np.mean(np.sum(values * values, axis=1)))) if self.balance_blocks else 1.0
            scalers[name], scales[name] = scaler, rms if rms > 0 else 1.0
        first = next(iter(aligned.values()))
        self.view_names_ = names
        self.weights_ = weights
        self.normalizations_ = norms
        self.scalers_, self.block_scales_ = scalers, scales
        self.input_dimensions_ = {name: view.n_dims for name, view in aligned.items()}
        self.entity_schema_ = (first.entity_type, first.id_scheme)
        self.training_ids_sha256_ = id_digest(first.entity_ids)
        self.n_fit_entities_, self.coverage_ = first.n_entities, coverage
        self.n_features_out_ = sum(self.input_dimensions_.values())
        return self

    @staticmethod
    def _normalize(values, method, scaler):
        if method == "zscore":
            return scaler.transform(values)
        if method == "l2":
            return normalize(values, axis=1)
        return values

    def _align_transform(self, X):
        check_is_fitted(self, "view_names_")
        if set(X) != set(self.view_names_):
            raise ValueError("transform requires exactly the view names used during fit.")
        aligned, coverage = align_views({name: X[name] for name in self.view_names_}, join=self.join)
        for name, view in aligned.items():
            if (view.entity_type, view.id_scheme) != self.entity_schema_:
                raise ValueError("Entity schema differs from the fitted fusion schema.")
            if view.n_dims != self.input_dimensions_[name]:
                raise ValueError(f"View {name!r} has a different dimension than during fit.")
        return aligned, coverage

    def _matrix(self, aligned):
        blocks = []
        for name, weight in zip(self.view_names_, self.weights_, strict=True):
            block = self._normalize(
                np.asarray(aligned[name].matrix, dtype=np.float64),
                self.normalizations_[name],
                self.scalers_[name],
            )
            blocks.append(block * (np.sqrt(weight) / self.block_scales_[name]))
        return np.concatenate(blocks, axis=1)

    def transform(self, X: Views):
        """Apply the fitted transformation and return an EmbeddingResult."""
        aligned, coverage = self._align_transform(X)
        return fused_result(self, aligned, self._matrix(aligned), coverage)
