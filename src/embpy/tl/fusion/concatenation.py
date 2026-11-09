"""Fitted, block-aware concatenation of identifier-aligned embeddings."""

from __future__ import annotations

import numpy as np

from ._base import BaseFusion, fused_result, validate_weights
from ._preprocessing import fit_preprocessing, transform_preprocessing
from ._views import Views


class ConcatFusion(BaseFusion):
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
        aligned = self._fit_views(X)
        self.weights_ = validate_weights(self.view_names_, self.weights)
        fit_preprocessing(self, aligned, balance_blocks=self.balance_blocks)
        self.n_features_out_ = sum(self.input_dimensions_.values())
        self.n_projection_coefficients_ = 0
        self.is_fitted_ = True
        return self

    def _matrix(self, aligned):
        prepared = transform_preprocessing(self, aligned)
        blocks = [
            prepared[name] * np.sqrt(weight) for name, weight in zip(self.view_names_, self.weights_, strict=True)
        ]
        return np.concatenate(blocks, axis=1)

    def transform(self, X: Views):
        """Apply the fitted transformation and return an EmbeddingResult."""
        aligned, coverage = self._align_transform(X)
        return fused_result(self, aligned, self._matrix(aligned), coverage)
