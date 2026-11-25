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

logger = logging.getLogger(__name__)


class NotRawCountsWarning(UserWarning):
    """:func:`preprocess_counts` was given values that do not look like raw counts."""


# Kinds of values that are certainly not raw counts.
_NOT_RAW_COUNTS = ("log_normalized", "continuous_normalized", "scaled_or_centered")


def _expression_kind(matrix) -> str:  # type: ignore[no-untyped-def]
    """What a count matrix holds, judged from up to 1000 values of its first 50 rows.

    Returns ``"raw_counts"`` (non-negative integers), ``"log_normalized"`` (non-integers
    up to 25), ``"continuous_normalized"`` (larger non-integers), ``"scaled_or_centered"``
    (negative values) or ``"unknown"`` (nothing to sample); the same rules as the MCP
    tool ``inspect_data``.
    """
    sample = matrix[: min(50, matrix.shape[0])]
    values = sample.data[:1000] if sp.issparse(sample) else np.ravel(np.asarray(sample))[:1000]
    if values.size == 0:
        return "unknown"
    if float(np.min(values)) < -1e-5:
        return "scaled_or_centered"
    if np.all(np.isclose(values, np.round(values), atol=1e-4)):
        return "raw_counts"
    return "log_normalized" if float(np.max(values)) <= 25.0 else "continuous_normalized"


def _raw_counts_source(adata) -> tuple[bool, str | None]:  # type: ignore[no-untyped-def]
    """Where :func:`preprocess_counts` should take raw counts from.

    Returns ``(use_counts_layer, problem)``: ``use_counts_layer`` is True when ``.X`` is
    not raw counts but ``.layers["counts"]`` is; ``problem`` describes the input when
    neither holds raw counts (``.X`` is then treated as counts, as before).
    """
    x_kind = _expression_kind(adata.X)
    if x_kind not in _NOT_RAW_COUNTS:
        return False, None
    if "counts" in adata.layers:
        layer_kind = _expression_kind(adata.layers["counts"])
        if layer_kind not in _NOT_RAW_COUNTS:
            return True, None
        found = f".X holds {x_kind} values and .layers['counts'] holds {layer_kind} values"
    else:
        found = f".X holds {x_kind} values and there is no .layers['counts']"
    return False, (
        f"preprocess_counts expects raw counts, but {found}. They are treated as raw counts, so "
        "normalizing and log-transforming them again distorts the data. Pass raw counts in .X or "
        ".layers['counts'], or skip preprocessing for data that is already normalized."
    )


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
        AnnData with raw counts in ``.X``. If ``.X`` holds other values (normalized,
        log-transformed or scaled) and ``.layers["counts"]`` holds counts, those
        counts are used instead. If neither holds counts, ``.X`` is treated as counts
        and a :class:`NotRawCountsWarning` (a ``UserWarning``) says what was found.
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

    use_counts_layer, problem = _raw_counts_source(adata)
    if use_counts_layer:
        # .X is already normalized, but .layers["counts"] holds the counts: start from those.
        logger.info("preprocess_counts: .X is not raw counts; starting from .layers['counts'].")
        adata.X = adata.layers["counts"].copy()
    elif problem:
        warnings.warn(problem, NotRawCountsWarning, stacklevel=2)

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
