"""Regularized two-view canonical correlation fusion."""

import numpy as np

from ._alignment import AlignmentFusion, ridge_factors


class CCAFusion(AlignmentFusion):
    """Align exactly two views by regularized CCA and average their scores.

    Each view is normalized and centered using training data. ``regularization``
    is a nonnegative scalar or exact view-name mapping: the covariance ridge is
    this multiplier times the mean retained covariance eigenvalue. Zero gives
    ordinary CCA on the numerical support. Optional ``pre_components`` applies
    train-fitted PCA before CCA. ``n_components`` cannot exceed either retained
    rank. Canonical scores are scaled to unit training standard deviation before
    weighted averaging. ``weights`` are fixed, nonnegative and normalized to sum
    one. Labels are ignored. New entities require both fitted views.

    ``canonical_correlations_`` are training Pearson correlations;
    ``regularized_correlations_`` are the singular values of the regularized
    whitened cross-covariance. These differ when regularization is nonzero.
    """

    def __init__(
        self,
        n_components=64,
        regularization=0.1,
        normalization="l2",
        weights=None,
        pre_components=None,
        join="strict",
        rank_tol=1e-7,
        random_state=0,
    ):
        self.n_components = n_components
        self.regularization = regularization
        self.normalization = normalization
        self.weights = weights
        self.pre_components = pre_components
        self.join = join
        self.rank_tol = rank_tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit canonical projections using paired training entities only."""
        blocks = self._prepare(X, exact_views=2)
        factors = ridge_factors(self, blocks)
        if self.n_components > min(self.ranks_.values()):
            raise ValueError("n_components exceeds a view's numerical rank.")
        first, second = self.view_names_
        U, s, V, denominator = factors[first]
        Q, t, W, other_denominator = factors[second]
        cross = (s / np.sqrt(denominator))[:, None] * (U.T @ Q) * (t / np.sqrt(other_denominator))[None, :]
        left, correlations, right = np.linalg.svd(cross, full_matrices=False)
        k = self.n_components
        directions = (left[:, :k], right.T[:, :k])
        self.projections_, scores = {}, []
        for name, direction in zip(self.view_names_, directions, strict=True):
            U, s, V, denominator = factors[name]
            projection = (V / np.sqrt(denominator)) @ direction
            score = (U * s) @ (V.T @ projection)
            scale = score.std(axis=0, ddof=1)
            if np.any(scale <= 0):
                raise ValueError("Canonical scores have zero variance.")
            self.projections_[name] = projection / scale
            scores.append(score / scale)
        self.regularized_correlations_ = correlations[:k]
        self.canonical_correlations_ = np.sum(scores[0] * scores[1], axis=0) / (self.n_fit_entities_ - 1)
        return self._finish()

    def _metadata(self):
        return {
            **super()._metadata(),
            "covariance_ridge": self.ridge_,
            "canonical_correlations": self.canonical_correlations_.tolist(),
            "regularized_correlations": self.regularized_correlations_.tolist(),
        }
