"""Numerical helpers for inductive, linear multi-view alignment."""

from collections.abc import Mapping

import numpy as np
from sklearn.utils.extmath import randomized_svd

from embpy._pca import fit_pca

from ._base import BaseFusion, fused_result, validate_weights
from ._preprocessing import fit_preprocessing, transform_preprocessing


class AlignmentFusion(BaseFusion):
    """Shared implementation; public constructors live in the concrete classes."""

    def _prepare(self, X, *, exact_views=None):
        aligned = self._fit_views(X, min_views=2, exact_views=exact_views)
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, (int, np.integer))
            or self.n_components < 1
        ):
            raise ValueError("n_components must be a positive integer.")
        if not np.isfinite(self.rank_tol) or not 0 < self.rank_tol < 1:
            raise ValueError("rank_tol must be finite and between zero and one.")
        if self.n_fit_entities_ < 2:
            raise ValueError("Alignment needs at least two training entities.")
        weights = validate_weights(self.view_names_, self.weights)
        weights = weights / weights.max()
        self.weights_ = weights / weights.sum()
        blocks = fit_preprocessing(self, aligned)
        self.means_ = {name: block.mean(axis=0) for name, block in blocks.items()}
        return {name: block - self.means_[name] for name, block in blocks.items()}

    def _finish(self):
        self.n_features_out_ = self.n_components
        self.n_projection_coefficients_ = sum(matrix.size for matrix in self.projections_.values())
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Project unseen entities, then average their aligned views with fixed weights."""
        aligned, coverage = self._align_transform(X)
        prepared = transform_preprocessing(self, aligned)
        matrix = sum(
            weight * ((prepared[name] - self.means_[name]) @ self.projections_[name])
            for name, weight in zip(self.view_names_, self.weights_, strict=True)
        )
        return fused_result(self, aligned, matrix, coverage)

    def _metadata(self):
        return {
            "aggregation": "weighted_mean",
            "normalized_weights": dict(zip(self.view_names_, self.weights_.tolist(), strict=True)),
            "retained_ranks": self.ranks_,
        }


def compact_svd(operator, name, matrix, pre_components=None):
    """Return U, s, V with null directions removed; absorb any PCA centering."""
    basis = None
    if not np.any(matrix):
        raise ValueError(f"View {name!r} has zero numerical rank after preprocessing.")
    if pre_components is not None:
        limit = min(matrix.shape[0] - 1, matrix.shape[1])
        if (
            isinstance(pre_components, bool)
            or not isinstance(pre_components, (int, np.integer))
            or not 1 <= pre_components <= limit
        ):
            raise ValueError(f"pre_components must be an integer between 1 and {limit}.")
        pca = fit_pca(matrix, pre_components, strict=True, random_state=operator.random_state)
        operator.means_[name] += pca.mean_
        basis = pca.components_.T
        matrix = (matrix - pca.mean_) @ basis
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    keep = s > operator.rank_tol * s[0] if len(s) and s[0] > 0 else np.zeros(len(s), dtype=bool)
    if not keep.any():
        raise ValueError(f"View {name!r} has zero numerical rank after preprocessing.")
    V = Vt[keep].T
    return U[:, keep], s[keep], V if basis is None else basis @ V


def ridge_factors(operator, blocks):
    values = operator.regularization
    if isinstance(values, Mapping):
        if set(values) != set(blocks):
            raise ValueError("regularization must name exactly the supplied views.")
        values = dict(values)
    else:
        values = dict.fromkeys(blocks, values)
    if any(not np.isscalar(x) or not np.isfinite(x) or x < 0 for x in values.values()):
        raise ValueError("regularization must be finite and nonnegative.")
    factors, operator.ranks_, operator.ridge_ = {}, {}, {}
    for name, matrix in blocks.items():
        U, s, V = compact_svd(operator, name, matrix, operator.pre_components)
        # Relative ridge: multiplier times the mean retained covariance eigenvalue.
        ridge = float(values[name] * np.mean(s * s) / (len(matrix) - 1))
        operator.ridge_[name], operator.ranks_[name] = ridge, len(s)
        factors[name] = (U, s, V, s * s + (len(matrix) - 1) * ridge)
    return factors


def leading_svd(matrix, count, random_state):
    if count > min(matrix.shape):
        raise ValueError("n_components exceeds the available shared rank.")
    if min(matrix.shape) > 256 and count < min(matrix.shape) // 2:
        return randomized_svd(matrix, n_components=count, n_iter=7, random_state=random_state)
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :count], s[:count], Vt[:count]
