"""Linear fusion operators fitted on training entities."""

from __future__ import annotations

from sklearn.utils.validation import check_is_fitted

from embpy._pca import fit_pca

from ._base import fused_result
from .concatenation import ConcatFusion


class PCAFusion(ConcatFusion):
    """Normalize and concatenate views, then fit a single shared PCA.

    Unlike independently projecting each view, this learns a joint basis.
    ``n_components`` must fit within the training matrix dimensions; it is
    never silently reduced, so benchmarks can enforce a fixed output width.
    All other parameters have the meanings documented by ConcatFusion.
    """

    def __init__(
        self,
        n_components=128,
        normalization="l2",
        weights=None,
        balance_blocks=False,
        join="strict",
        random_state=0,
    ):
        super().__init__(normalization, weights, balance_blocks, join)
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit block preprocessing and joint PCA on training entities."""
        if hasattr(self, "pca_"):
            del self.pca_
        super().fit(X, y)
        aligned, _ = self._align_transform(X)
        self.pca_ = fit_pca(self._matrix(aligned), self.n_components, random_state=self.random_state, strict=True)
        self.n_features_out_ = int(self.pca_.n_components_)
        return self

    def transform(self, X):
        """Project new entities into the stored training PCA basis."""
        check_is_fitted(self, "pca_")
        aligned, coverage = self._align_transform(X)
        return fused_result(self, aligned, self.pca_.transform(self._matrix(aligned)), coverage)
