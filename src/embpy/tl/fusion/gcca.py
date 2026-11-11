"""Regularized MAXVAR generalized canonical correlation fusion."""

import numpy as np

from ._alignment import AlignmentFusion, leading_svd, ridge_factors


class GCCAFusion(AlignmentFusion):
    """Find a common space for two or more views using ridge MAXVAR GCCA.

    Minimize weighted ridge reconstruction of a common score matrix whose
    columns are orthogonal. Solve through a concatenation of compact ridge
    projector factors, without constructing feature-by-feature cross-covariance
    blocks. Large low-rank solves use reproducible randomized SVD (7 iterations).
    This is MAXVAR, not SUMCOR MCCA.

    Parameters match CCAFusion. Individual views may have rank below
    ``n_components`` provided their combined support has sufficient rank.
    Fitted ridge regressions map each view into the common space; their weighted
    mean is scaled to unit training standard deviation. Training and new entities
    use exactly the same projection-and-average formula. No labels are used.
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
        """Fit a common MAXVAR space and reusable per-view projections."""
        blocks = self._prepare(X)
        factors = ridge_factors(self, blocks)
        factor = np.concatenate(
            [
                np.sqrt(weight) * U * (s / np.sqrt(denominator))
                for weight, (U, s, V, denominator) in zip(self.weights_, factors.values(), strict=True)
            ],
            axis=1,
        )
        common, singular, _ = leading_svd(factor, self.n_components, self.random_state)
        if singular[-1] <= self.rank_tol * singular[0]:
            raise ValueError("n_components exceeds the combined numerical rank.")
        common *= np.sqrt(self.n_fit_entities_ - 1)
        self.projections_ = {}
        fitted_scores = np.zeros_like(common)
        for weight, (name, (U, s, V, denominator)) in zip(self.weights_, factors.items(), strict=True):
            coefficient = (s / denominator)[:, None] * (U.T @ common)
            self.projections_[name] = V @ coefficient
            fitted_scores += weight * ((U * s) @ coefficient)
        scale = fitted_scores.std(axis=0, ddof=1)
        if np.any(scale <= 0):
            raise ValueError("Common scores have zero variance.")
        self.projections_ = {name: matrix / scale for name, matrix in self.projections_.items()}
        self.consensus_eigenvalues_ = singular * singular
        return self._finish()

    def _metadata(self):
        return {
            **super()._metadata(),
            "objective": "ridge_MAXVAR",
            "covariance_ridge": self.ridge_,
            "consensus_eigenvalues": self.consensus_eigenvalues_.tolist(),
        }
