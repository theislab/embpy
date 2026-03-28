"""Dimensionality reduction tools (UMAP, t-SNE).

Supports both CPU (scanpy) and GPU (rapids_singlecell) backends.
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

    if backend == "gpu":
        rsc = _require_rapids()
        rsc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        rsc.tl.umap(adata, n_components=n_components)
    else:
        sc = _require_scanpy()
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata, n_components=n_components)

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
