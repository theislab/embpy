"""Dimensionality reduction tools (PCA, UMAP, t-SNE).

UMAP supports both CPU (scanpy) and GPU (rapids_singlecell) backends.
PCA and t-SNE are CPU-only (scikit-learn / scanpy).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from anndata import AnnData


def _require_scanpy():
    try:
        import scanpy as sc
        return sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for this function. "
            "Install with: pip install scanpy"
        ) from e


def _require_rapids():
    try:
        import rapids_singlecell as rsc
        return rsc
    except ImportError as e:
        raise ImportError(
            "rapids_singlecell is required for GPU backend. "
            "Install with: pip install rapids-singlecell"
        ) from e


def compute_pca(
    adata: AnnData,
    obsm_key: str,
    n_components: int = 2,
    output_key: str | None = None,
    random_state: int = 0,
) -> AnnData:
    """Compute PCA coordinates from an embedding matrix.

    Mirrors :func:`compute_umap` / :func:`compute_tsne` but uses
    :class:`sklearn.decomposition.PCA`. Stores the projected coordinates in
    ``obsm[output_key]`` and the explained-variance ratio in
    ``uns[f"{output_key}_variance_ratio"]`` so plotting helpers can label
    axes with the percentage of variance explained.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_components
        Number of principal components to keep. Clamped to
        ``min(n_obs, n_features, n_components)``.
    output_key
        Key for the PCA coordinates in ``.obsm``.
        Defaults to ``"X_pca_{obsm_key}"``.
    random_state
        Random seed for the SVD solver.

    Returns
    -------
    AnnData with PCA coordinates in ``obsm[output_key]`` and the
    variance-ratio array in ``uns[output_key + "_variance_ratio"]``.
    """
    from .._pca import pca_project

    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not in adata.obsm. Available: {list(adata.obsm.keys())}"
        )

    out = output_key or f"X_pca_{obsm_key}"
    coords, variance_ratio = pca_project(
        adata.obsm[obsm_key], n_components, random_state=random_state,
    )
    adata.obsm[out] = coords
    adata.uns[f"{out}_variance_ratio"] = variance_ratio
    logging.info(
        "PCA (%d-D) on '%s' stored in obsm['%s'] (variance ratio: %s).",
        coords.shape[1], obsm_key, out,
        np.round(variance_ratio, 3).tolist(),
    )
    return adata


def compute_umap(
    adata: AnnData,
    obsm_key: str,
    n_neighbors: int = 15,
    n_components: int = 2,
    output_key: str | None = None,
    backend: Literal["cpu", "gpu"] = "cpu",
) -> AnnData:
    """Compute UMAP coordinates from embeddings.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_neighbors
        Number of neighbors for graph construction.
    n_components
        Number of UMAP dimensions.
    output_key
        Key for the UMAP coordinates in ``.obsm``.
        Defaults to ``"X_umap_{obsm_key}"``.
    backend
        ``"cpu"`` uses scanpy, ``"gpu"`` uses rapids_singlecell.

    Returns
    -------
    AnnData with UMAP coordinates in ``obsm[output_key]``.
    """
    out = output_key or f"X_umap_{obsm_key}"

    # UMAP's default spectral initialisation solves for n_components + 1
    # eigenvectors of the neighbour graph, which needs strictly more
    # observations than that. Below the threshold scipy raises "Cannot use
    # scipy.linalg.eigh for sparse A with k >= N" from deep inside the solver.
    # A random init is the standard fallback and is what UMAP itself suggests
    # for tiny inputs; the layout is less stable, which is inherent at this size.
    init_pos = "spectral" if adata.n_obs > n_components + 1 else "random"
    if init_pos == "random":
        logging.warning(
            "Only %d observations for a %d-D UMAP; using a random initialisation "
            "instead of spectral, and the layout should be read as indicative.",
            adata.n_obs, n_components,
        )
    n_neighbors = max(2, min(n_neighbors, adata.n_obs - 1))

    if backend == "gpu":
        rsc = _require_rapids()
        rsc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        rsc.tl.umap(adata, n_components=n_components, init_pos=init_pos)
    else:
        sc = _require_scanpy()
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata, n_components=n_components, init_pos=init_pos)

    adata.obsm[out] = adata.obsm["X_umap"].copy()
    logging.info(
        "UMAP (%d-D, backend=%s) on '%s' stored in obsm['%s'].",
        n_components, backend, obsm_key, out,
    )
    return adata


def compute_tsne(
    adata: AnnData,
    obsm_key: str,
    n_components: int = 2,
    perplexity: float = 30.0,
    output_key: str | None = None,
) -> AnnData:
    """Compute t-SNE coordinates from embeddings.

    .. note::
       t-SNE is CPU-only. rapids_singlecell does not provide a GPU
       t-SNE implementation.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_components
        Number of t-SNE dimensions.
    perplexity
        Perplexity parameter.
    output_key
        Key for t-SNE coordinates in ``.obsm``.
        Defaults to ``"X_tsne_{obsm_key}"``.

    Returns
    -------
    AnnData with t-SNE coordinates in ``obsm[output_key]``.
    """
    sc = _require_scanpy()
    out = output_key or f"X_tsne_{obsm_key}"
    actual_perp = min(perplexity, adata.n_obs - 1)
    X = np.asarray(adata.obsm[obsm_key])
    n_pcs = min(X.shape[1], X.shape[0] - 1) if X.shape[1] > 50 else 0
    sc.tl.tsne(
        adata, use_rep=obsm_key,
        n_pcs=n_pcs if n_pcs > 0 else None,
        perplexity=actual_perp,
    )
    adata.obsm[out] = adata.obsm["X_tsne"].copy()
    logging.info(
        "t-SNE (%d-D) on '%s' stored in obsm['%s'].",
        n_components, obsm_key, out,
    )
    return adata
