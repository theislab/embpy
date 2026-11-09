"""The single PCA projection used across embpy.

Both :func:`embpy.tl.dimred.compute_pca` (AnnData-oriented) and
:func:`embpy.io.harmonize.harmonize` (EmbeddingResult-oriented) delegate
here, so there is exactly one PCA implementation in the package.

Kept as a dependency-light leaf -- numpy + scikit-learn only, no torch /
transformers / anndata -- so the io output layer can harmonize
embeddings without importing the heavy analysis / model stack.
"""

from __future__ import annotations

import numpy as np


def fit_pca(matrix, n_components, *, random_state=0, strict=False):
    """Return a fitted PCA so new entities can reuse the training basis.

    Existing projection callers retain clamping; strict fusion callers reject
    impossible output dimensions instead of silently changing model capacity.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("matrix must be a nonempty, finite 2D array.")
    limit = min(X.shape)
    if strict and (
        isinstance(n_components, bool)
        or not isinstance(n_components, (int, np.integer))
        or not 1 <= n_components <= limit
    ):
        raise ValueError(f"n_components must be an integer between 1 and {limit} for the training matrix.")
    count = max(1, min(int(n_components), limit))
    return PCA(n_components=count, random_state=random_state).fit(X)


def pca_project(
    matrix: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the rows of ``matrix`` onto its top principal components.

    Parameters
    ----------
    matrix
        ``(n_rows, n_features)`` array.
    n_components
        Desired number of components. Clamped to
        ``min(n_rows, n_features, n_components)`` (and at least 1).
    random_state
        Seed for the SVD solver (determinism).

    Returns
    -------
    coords : np.ndarray
        ``(n_rows, n_kept)`` projected coordinates (float64).
    explained_variance_ratio : np.ndarray
        ``(n_kept,)`` fraction of variance per kept component.
    """
    pca = fit_pca(matrix, n_components, random_state=random_state)
    coords = pca.transform(np.asarray(matrix, dtype=np.float64))
    return coords, np.asarray(pca.explained_variance_ratio_)


__all__ = ["fit_pca", "pca_project"]
