"""Inductive fusion through a weighted sum of positive semidefinite kernels."""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from ._base import BaseFusion, fused_result, validate_weights
from ._preprocessing import fit_preprocessing, transform_preprocessing


class KernelFusion(BaseFusion):
    """Truncated Nyström features of a weighted multi-view kernel.

    ``kernels`` is ``'rbf'``, ``'linear'``, ``'tanimoto'`` or an exact view-name
    mapping. Tanimoto accepts nonnegative features; use normalization='none'
    for binary fingerprints. Its zero/zero similarity is one. Fixed weights
    are normalized to sum to one; this is not supervised multiple-kernel
    learning. ``gamma='median'`` uses the median nonzero squared distance in
    at most 512 training rows per RBF view (gamma = 1 / median).

    Landmarks are sampled from training entities without replacement. The
    leading ``n_components`` positive landmark-kernel eigenvectors define
    the out-of-sample mapping; width is never silently reduced. Computation
    uses batches of rows by landmarks, never an all-entity kernel matrix.
    ``center=True`` subtracts the training feature mean after projection.
    """

    def __init__(
        self,
        n_components=64,
        n_landmarks=256,
        kernels="rbf",
        gamma="median",
        weights=None,
        normalization="l2",
        center=True,
        batch_size=1024,
        rank_tol=1e-10,
        join="strict",
        random_state=0,
    ):
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.kernels = kernels
        self.gamma = gamma
        self.weights = weights
        self.normalization = normalization
        self.center = center
        self.batch_size = batch_size
        self.rank_tol = rank_tol
        self.join = join
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit preprocessing, landmarks and their eigensystem on training views."""
        aligned = self._fit_views(X)
        for key in ("n_components", "n_landmarks", "batch_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if not 0 < self.rank_tol < 1:
            raise ValueError("rank_tol must be in (0, 1).")
        n_landmarks = min(self.n_landmarks, self.n_fit_entities_)
        if self.n_components > n_landmarks:
            raise ValueError("n_components exceeds the number of training landmarks.")
        self.kernels_ = self._per_view(self.kernels, "kernels")
        if any(k not in {"rbf", "linear", "tanimoto"} for k in self.kernels_.values()):
            raise ValueError("kernels must be 'rbf', 'linear' or 'tanimoto'.")
        self.weights_ = validate_weights(self.view_names_, self.weights)
        self.weights_ /= self.weights_.sum()
        blocks = fit_preprocessing(self, aligned)
        for name, values in blocks.items():
            if self.kernels_[name] == "tanimoto" and (values < 0).any():
                raise ValueError("Tanimoto requires nonnegative features; do not z-score fingerprints.")
        rng = np.random.default_rng(self.random_state)
        rows = rng.choice(self.n_fit_entities_, n_landmarks, replace=False)
        self.landmark_ids_ = tuple(next(iter(aligned.values())).entity_ids[i] for i in rows)
        self.landmarks_ = {name: values[rows].copy() for name, values in blocks.items()}
        self.gammas_ = self._per_view(self.gamma, "gamma")
        for name, values in blocks.items():
            if self.kernels_[name] != "rbf":
                continue
            gamma = self.gammas_[name]
            if isinstance(gamma, str) and gamma == "median":
                sample = values[rng.choice(len(values), min(512, len(values)), replace=False)]
                distances = euclidean_distances(sample, squared=True)
                distances = distances[np.triu_indices(len(sample), k=1)]
                positive = distances[distances > 0]
                gamma = 1 / float(np.median(positive)) if len(positive) else 1.0
            if isinstance(gamma, (str, bool)) or not np.isfinite(gamma) or gamma <= 0:
                raise ValueError("RBF gamma must be 'median' or a positive finite number.")
            self.gammas_[name] = float(gamma)
        kernel = self._kernel(self.landmarks_)
        values, vectors = np.linalg.eigh((kernel + kernel.T) / 2)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        rank = int(np.count_nonzero(values > self.rank_tol * max(values[0], 1.0)))
        if self.n_components > rank:
            raise ValueError(f"n_components exceeds positive landmark kernel rank ({rank}).")
        self.eigenvalues_ = values[: self.n_components]
        self.projection_ = vectors[:, : self.n_components] / np.sqrt(self.eigenvalues_)
        self.feature_mean_ = np.zeros(self.n_components)
        if self.center:
            total = np.zeros(self.n_components)
            for start in range(0, self.n_fit_entities_, self.batch_size):
                batch = {k: v[start : start + self.batch_size] for k, v in blocks.items()}
                total += (self._kernel(batch) @ self.projection_).sum(axis=0)
            self.feature_mean_ = total / self.n_fit_entities_
        self.n_features_out_ = self.n_components
        self.n_projection_coefficients_ = self.projection_.size
        self.is_fitted_ = True
        return self

    def _per_view(self, value, label):
        if isinstance(value, Mapping):
            if set(value) != set(self.view_names_):
                raise ValueError(f"{label} must name exactly the supplied views.")
            return dict(value)
        return dict.fromkeys(self.view_names_, value)

    def _kernel(self, blocks):
        result = None
        for name, weight in zip(self.view_names_, self.weights_, strict=True):
            X, Y = blocks[name], self.landmarks_[name]
            kind = self.kernels_[name]
            if kind == "rbf":
                K = np.exp(-self.gammas_[name] * euclidean_distances(X, Y, squared=True))
            else:
                K = X @ Y.T
                if kind == "tanimoto":
                    if (X < 0).any() or (Y < 0).any():
                        raise ValueError("Tanimoto requires nonnegative features; do not z-score fingerprints.")
                    denominator = (X * X).sum(axis=1)[:, None] + (Y * Y).sum(axis=1)[None, :] - K
                    K = np.divide(K, denominator, out=np.ones_like(K), where=denominator > 0)
            result = weight * K if result is None else result + weight * K
        return result

    def transform(self, X):
        """Apply the stored landmark mapping to unseen entities."""
        aligned, coverage = self._align_transform(X)
        blocks = transform_preprocessing(self, aligned)
        n = next(iter(aligned.values())).n_entities
        output = np.empty((n, self.n_features_out_))
        for start in range(0, n, self.batch_size):
            batch = {k: v[start : start + self.batch_size] for k, v in blocks.items()}
            output[start : start + self.batch_size] = self._kernel(batch) @ self.projection_ - self.feature_mean_
        return fused_result(self, aligned, output, coverage)

    def _metadata(self):
        return {
            "kernels": self.kernels_,
            "gamma_fitted": self.gammas_,
            "n_landmarks": len(self.landmark_ids_),
            "supervised": False,
            "approximation": "truncated_nystrom_weighted_kernel_sum",
        }
