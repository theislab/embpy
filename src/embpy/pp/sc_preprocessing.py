"""Single-cell RNA-seq preprocessing pipelines.

Provides two pipelines for preparing AnnData objects before embedding:

* **raw** -- minimal QC filtering, keeps raw counts in ``.X``.
* **standard** -- scanpy-based workflow: filter, normalize, log1p,
  HVG selection, optional scaling.  Raw counts are preserved in
  ``.layers["counts"]`` and processed data in ``.layers["log_normalized"]``,
  while ``.X`` is restored to the original raw counts.

Set ``backend="gpu"`` to use ``rapids_singlecell`` for GPU-accelerated
preprocessing (normalize, log1p, HVG, scale).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _require_scanpy():  # type: ignore[no-untyped-def]
    try:
        import scanpy as sc  # type: ignore[import-untyped]
        return sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for single-cell preprocessing. "
            "Install with: pip install scanpy"
        ) from exc


def _require_rapids():  # type: ignore[no-untyped-def]
    try:
        import rapids_singlecell as rsc  # type: ignore[import-untyped]
        return rsc
    except ImportError as exc:
        raise ImportError(
            "rapids_singlecell is required for GPU preprocessing. "
            "Install with: pip install rapids-singlecell"
        ) from exc


def preprocess_counts(
    adata,  # anndata.AnnData
    pipeline: Literal["raw", "standard"] = "standard",
    *,
    # QC filtering
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float | None = None,
    # Normalization (standard pipeline)
    target_sum: float | None = 1e4,
    log_transform: bool = True,
    # HVG selection
    n_top_genes: int = 2000,
    select_hvg: bool = True,
    # Scaling
    scale: bool = False,
    max_value: float | None = 10.0,
    copy: bool = True,
    # Backend
    backend: Literal["cpu", "gpu"] = "cpu",
):
    """Preprocess an AnnData for downstream cell embedding.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``.X``.
    pipeline
        ``"raw"`` applies only basic QC filtering.
        ``"standard"`` runs the full workflow (normalize, log1p,
        HVG, optional scale).
    min_genes
        Minimum number of genes expressed per cell.
    min_cells
        Minimum number of cells expressing a gene.
    max_pct_mito
        If set, cells with mitochondrial fraction above this threshold
        are removed.
    target_sum
        Target total counts per cell for normalization.
    log_transform
        Whether to log1p-transform after normalization.
    n_top_genes
        Number of highly variable genes to select.
    select_hvg
        Whether to filter to HVGs (only marks them if ``False``).
    scale
        Whether to scale to unit variance.
    max_value
        Clip values after scaling (ignored if ``scale=False``).
    copy
        If ``True``, operate on a copy; otherwise modify in place.
    backend
        ``"cpu"`` uses scanpy (default). ``"gpu"`` uses
        ``rapids_singlecell`` for GPU-accelerated normalize, log1p,
        HVG, and scale.

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData with:

        - ``.X`` = original raw counts
        - ``.layers["counts"]`` = raw counts (copy)
        - ``.layers["log_normalized"]`` = processed expression
          (standard pipeline only)
        - ``.var["highly_variable"]`` = HVG flag (standard pipeline)
    """
    sc = _require_scanpy()
    if backend == "gpu":
        rsc = _require_rapids()
        pp = rsc.pp
        logger.info("Using rapids_singlecell GPU backend for preprocessing")
    else:
        pp = sc.pp

    if copy:
        adata = adata.copy()

    # Preserve raw counts
    if sp.issparse(adata.X):
        adata.layers["counts"] = adata.X.copy()
    else:
        adata.layers["counts"] = np.array(adata.X, copy=True)

    # ---- QC filtering (always scanpy -- fast on CPU) ---------------------
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if max_pct_mito is not None:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True,
        )
        before = adata.n_obs
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
        logger.info(
            "Mito filter: %d -> %d cells (threshold=%.1f%%)",
            before, adata.n_obs, max_pct_mito,
        )
        if "counts" not in adata.layers:
            if sp.issparse(adata.X):
                adata.layers["counts"] = adata.X.copy()
            else:
                adata.layers["counts"] = np.array(adata.X, copy=True)

    logger.info("After QC: %d cells x %d genes", adata.n_obs, adata.n_vars)

    if pipeline == "raw":
        logger.info("Pipeline 'raw': returning filtered raw counts.")
        return adata

    # ---- Standard pipeline (uses pp = scanpy or rapids_singlecell) -------
    pp.normalize_total(adata, target_sum=target_sum)

    if log_transform:
        pp.log1p(adata)

    # Store the log-normalized state
    if sp.issparse(adata.X):
        adata.layers["log_normalized"] = adata.X.copy()
    else:
        adata.layers["log_normalized"] = np.array(adata.X, copy=True)

    # HVG
    if select_hvg or n_top_genes:
        pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes,
            flavor="seurat_v3" if not log_transform else "seurat",
        )
        n_hvg = adata.var["highly_variable"].sum()
        logger.info("Identified %d highly variable genes", n_hvg)

    # Scale
    if scale:
        pp.scale(adata, max_value=max_value)
        logger.info("Scaled to unit variance (max_value=%s)", max_value)

    # Restore .X to raw counts
    adata.X = adata.layers["counts"].copy()

    logger.info(
        "Pipeline 'standard' (backend=%s): %d cells x %d genes, "
        "log_normalized in .layers, raw counts in .X",
        backend, adata.n_obs, adata.n_vars,
    )
    return adata
