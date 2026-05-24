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
    from sklearn.decomposition import PCA

    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"matrix must be 2D (n_rows, n_features), got shape {X.shape!r}.")
    n_comp = max(1, min(int(n_components), X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, np.asarray(pca.explained_variance_ratio_)


__all__ = ["pca_project"]
