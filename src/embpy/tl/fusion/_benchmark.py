"""Array adapter for fitting fusion inside sklearn cross-validation pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from embpy.io.result import EmbeddingProvenance, EmbeddingResult


class MatrixFusion(TransformerMixin, BaseEstimator):
    """Adapt already ID-aligned, packed views to sklearn's row-slicing protocol.

    Used only after the benchmark has aligned identifiers and recorded splits.
    Synthetic row IDs here are local to the pipeline; they are not exported as
    biological identifiers. The full fitted fusion remains inside the pipeline.
    """

    def __init__(self, fusion, names, widths):
        self.fusion = fusion
        self.names = names
        self.widths = widths

    def _views(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != sum(self.widths):
            raise ValueError("Packed view dimensions do not match the benchmark schema.")
        ids = tuple(str(i) for i in range(len(X)))
        offset, views = 0, {}
        for name, width in zip(self.names, self.widths, strict=True):
            views[name] = EmbeddingResult(
                X[:, offset : offset + width],
                ids,
                "sample",
                "benchmark_row",
                EmbeddingProvenance(model=name),
            )
            offset += width
        return views

    def fit(self, X, y=None):
        """Clone and fit the fusion using only rows supplied by sklearn."""
        self.fusion_ = clone(self.fusion).fit(self._views(X))
        return self

    def transform(self, X):
        """Apply the fitted fusion to validation or test rows."""
        check_is_fitted(self, "fusion_")
        # Raw descriptors can be large while still valid float32 embeddings.
        # Readout Gram matrices must not overflow when squaring those values.
        return self.fusion_.transform(self._views(X)).matrix.astype(np.float64)
