"""Shared helpers for the pl module."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

_SKIP_PREFIXES = ("X_umap", "X_tsne", "X_pca", "X_diffmap", "X_draw_graph")


def _get_embedding_keys(
    adata: AnnData,
    obsm_keys: list[str] | None = None,
) -> list[str]:
    """Discover embedding keys in *adata.obsm*.

    Selects 2-D numeric arrays and excludes common reduced-coordinate keys
    (``X_umap``, ``X_tsne``, ...) unless they are explicitly requested.
    """
    if obsm_keys is not None:
        for k in obsm_keys:
            if k not in adata.obsm:
                raise KeyError(f"'{k}' not in adata.obsm. Available: {list(adata.obsm.keys())}")
        return list(obsm_keys)

    keys: list[str] = []
    for k in adata.obsm:
        arr = adata.obsm[k]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2:
            continue
        if any(k.startswith(p) for p in _SKIP_PREFIXES):
            continue
        keys.append(k)
    return keys


def _labels_from_adata(adata: AnnData) -> list[str] | None:
    """Try to extract short observation labels from common columns."""
    for col in ("identifier", "drug_id", "gene_symbol"):
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).tolist()
    return list(adata.obs_names)
