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
import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp

from embpy._counts import NotRawCountsWarning
from embpy._counts import counts_source as _counts_source

logger = logging.getLogger(__name__)


class EmptyAfterFilteringError(ValueError):
    """QC filtering in :func:`preprocess_counts` left no cell or no gene.

    ``thresholds`` holds the filters that were applied; ``n_obs`` / ``n_vars`` the input size.
    """

    def __init__(self, emptied: str, thresholds: dict[str, float], n_obs: int, n_vars: int) -> None:
        self.emptied = emptied
        self.thresholds = thresholds
        self.n_obs = n_obs
        self.n_vars = n_vars
        applied = " and ".join(f"{name}={value}" for name, value in thresholds.items())
        super().__init__(
            f"Filtering with {applied} removed every {emptied} (input: {n_obs} cells x {n_vars} genes), "
            "so there is nothing to preprocess. Lower the thresholds."
        )


def _raw_counts_source(adata) -> tuple[bool, str | None]:  # type: ignore[no-untyped-def]
    """``(use_counts_layer, warning)`` for :func:`preprocess_counts`; see :func:`embpy._counts.counts_source`."""
    choice = _counts_source(adata)
    return choice["source"] == "counts", choice["message"]


def _require_scanpy():  # type: ignore[no-untyped-def]
    try:
        import scanpy as sc  # type: ignore[import-untyped]

        return sc
    except ImportError as exc:
        raise ImportError("scanpy is required for single-cell preprocessing. Install with: pip install scanpy") from exc


def _require_rapids():  # type: ignore[no-untyped-def]
    try:
        import rapids_singlecell as rsc  # type: ignore[import-untyped]

        return rsc
    except ImportError as exc:
        raise ImportError(
            "rapids_singlecell is required for GPU preprocessing. Install with: pip install rapids-singlecell"
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
    hvg_flavor: Literal["auto", "seurat", "seurat_v3", "cell_ranger"] = "auto",
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
        AnnData with counts in ``.X``. Non-integer counts with count-like magnitudes
        (ambient-corrected or EM counts) are used as they are. Only when ``.X`` is
        clearly not counts (log-transformed, e.g. marked by ``.uns["log1p"]``,
        normalized or scaled) and ``.layers["counts"]`` holds counts are those used
        instead; if neither holds counts, ``.X`` is treated as counts. Every case but
        integer counts in ``.X`` gives a :class:`NotRawCountsWarning` (a
        ``UserWarning``) saying what was found and which matrix was used. When ``.X``
        is used and the input's ``.layers["counts"]`` holds other values (e.g. the
        original counts next to corrected ones), that layer is kept as
        ``.layers["counts_original"]`` instead of being overwritten, with the same
        warning.
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
        Whether to compute and mark highly variable genes.
    hvg_flavor
        Highly-variable-gene method passed to scanpy/rapids. ``"auto"``
        uses ``"seurat"`` after log transformation and ``"seurat_v3"``
        when operating on unlogged normalized counts.
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
        - no ``.uns["log1p"]``: scanpy reads that entry as "``.X`` is
          log-transformed", and ``.X`` holds counts

    Raises
    ------
    EmptyAfterFilteringError
        A ``ValueError`` raised when QC filtering leaves no cell or no gene; it
        carries the ``thresholds`` that were applied.
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

    choice = _counts_source(adata)
    if choice["message"]:
        warnings.warn(choice["message"], NotRawCountsWarning, stacklevel=2)
    if choice["source"] == "counts":
        # .X is clearly not counts, but .layers["counts"] holds them: start from those.
        # .uns['log1p'] described the replaced .X, which now holds counts.
        adata.X = adata.layers["counts"].copy()
        adata.uns.pop("log1p", None)
    elif choice["keep_original_as"]:
        # .X is the source; keep the input's different .layers["counts"] instead of overwriting it.
        adata.layers[choice["keep_original_as"]] = adata.layers["counts"]

    # Preserve raw counts
    if sp.issparse(adata.X):
        adata.layers["counts"] = adata.X.copy()
    else:
        adata.layers["counts"] = np.array(adata.X, copy=True)

    # ---- QC filtering (always scanpy -- fast on CPU) ---------------------
    n_obs_input, n_vars_input = int(adata.n_obs), int(adata.n_vars)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if max_pct_mito is not None:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        before = adata.n_obs
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
        logger.info(
            "Mito filter: %d -> %d cells (threshold=%.1f%%)",
            before,
            adata.n_obs,
            max_pct_mito,
        )
        if "counts" not in adata.layers:
            if sp.issparse(adata.X):
                adata.layers["counts"] = adata.X.copy()
            else:
                adata.layers["counts"] = np.array(adata.X, copy=True)

    if adata.n_obs == 0 or adata.n_vars == 0:
        # Otherwise normalization and HVG selection fail later with an unclear error.
        thresholds: dict[str, float] = {"min_genes": min_genes, "min_cells": min_cells}
        if max_pct_mito is not None:
            thresholds["max_pct_mito"] = max_pct_mito
        raise EmptyAfterFilteringError("cell" if adata.n_obs == 0 else "gene", thresholds, n_obs_input, n_vars_input)

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
    if select_hvg and n_top_genes:
        resolved_hvg_flavor = ("seurat_v3" if not log_transform else "seurat") if hvg_flavor == "auto" else hvg_flavor
        pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=resolved_hvg_flavor,
        )
        n_hvg = adata.var["highly_variable"].sum()
        logger.info(
            "Identified %d highly variable genes (flavor=%s)",
            n_hvg,
            resolved_hvg_flavor,
        )

    # Scale
    if scale:
        pp.scale(adata, max_value=max_value)
        logger.info("Scaled to unit variance (max_value=%s)", max_value)

    # Restore .X to raw counts. uns['log1p'] tells scanpy (and cell-eval) that .X is
    # log-transformed, which it no longer is.
    adata.X = adata.layers["counts"].copy()
    adata.uns.pop("log1p", None)

    logger.info(
        "Pipeline 'standard' (backend=%s): %d cells x %d genes, log_normalized in .layers, raw counts in .X",
        backend,
        adata.n_obs,
        adata.n_vars,
    )
    return adata
