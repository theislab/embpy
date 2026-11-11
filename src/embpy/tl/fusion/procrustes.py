"""PCA followed by orthogonal Procrustes alignment and averaging."""

import numpy as np
from scipy.linalg import orthogonal_procrustes

from ._alignment import AlignmentFusion, compact_svd


class ProcrustesFusion(AlignmentFusion):
    """Rotate per-view PCA scores into a reference view, then average.

    PCA and the orthogonal maps are fitted on training entities only. Rotations
    may include reflections; no post-alignment scaling or whitening is applied.
    All views must have at least ``n_components`` non-null directions.
    ``reference=None`` uses the first fitted view; specifying its name makes the
    choice explicit. ``weights`` are fixed and normalized to sum one. New
    entities require every fitted view. This aligns geometry, not responses.
    """

    def __init__(
        self,
        n_components=64,
        normalization="l2",
        reference=None,
        weights=None,
        join="strict",
        rank_tol=1e-7,
        random_state=0,
    ):
        self.n_components = n_components
        self.normalization = normalization
        self.reference = reference
        self.weights = weights
        self.join = join
        self.rank_tol = rank_tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit PCA bases and rotations toward the chosen reference."""
        blocks = self._prepare(X)
        self.reference_view_ = self.view_names_[0] if self.reference is None else self.reference
        if self.reference_view_ not in blocks:
            raise ValueError("reference must name a supplied view.")
        scores, bases, self.ranks_ = {}, {}, {}
        for name, block in blocks.items():
            U, s, V = compact_svd(self, name, block, self.n_components)
            if len(s) < self.n_components:
                raise ValueError(f"n_components exceeds the numerical rank of {name!r}.")
            scores[name], bases[name], self.ranks_[name] = U * s, V, len(s)
        reference = scores[self.reference_view_]
        self.projections_, self.rotations_, self.alignment_residuals_ = {}, {}, {}
        for name in blocks:
            rotation = (
                np.eye(self.n_components)
                if name == self.reference_view_
                else orthogonal_procrustes(scores[name], reference)[0]
            )
            self.rotations_[name] = rotation
            self.projections_[name] = bases[name] @ rotation
            self.alignment_residuals_[name] = float(
                np.linalg.norm(scores[name] @ rotation - reference) / np.linalg.norm(reference)
            )
        return self._finish()

    def _metadata(self):
        return {
            **super()._metadata(),
            "reference": self.reference_view_,
            "relative_alignment_residuals": self.alignment_residuals_,
        }
