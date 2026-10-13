"""Core bioinformatics skills for the embpy Model Context Protocol (MCP) server.

These tools expose single-cell AnnData inspection, foundation-model embedding
generation via BioEmbedder, and embedding evaluation using scIB metrics.
All operations redirect non-protocol outputs to stderr to ensure that the
stdio communication stream remains clean and uncorrupted JSON-RPC.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_val(val: Any) -> Any:
    """Recursively convert numpy types and non-finite floats to JSON-safe primitives."""
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_sanitize_val(x) for x in val.tolist()]
    if isinstance(val, dict):
        return {str(k): _sanitize_val(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_val(x) for x in val]
    return val


def inspect_anndata(path: str, max_categories: int = 10) -> dict[str, Any]:
    """Inspect metadata, dimensions, and slots of an AnnData (.h5ad) dataset.

    Reads dataset metadata without loading full matrices into memory when
    possible, reporting cell and gene counts, observation columns (.obs),
    feature variables (.var), expression layers, embeddings (.obsm),
    and unstructured metadata (.uns).

    Parameters
    ----------
    path : str
        Filesystem path to the .h5ad AnnData file.
    max_categories : int, default=10
        Maximum unique category values to summarize for categorical observation columns.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting dataset status and metadata:
        - status: "success" or "error"
        - path: Normalized filesystem path
        - n_obs: Total cell count
        - n_vars: Total gene / feature count
        - obs_columns: Column names present in adata.obs
        - obs_summary: High-level summary of observation annotations
        - var_columns: Column names present in adata.var
        - layers: List of expression matrix layer names
        - obsm_keys: List of multidimensional array keys (e.g., 'X_pca', 'X_scvi')
        - obsp_keys: List of pairwise graph / distance keys
        - uns_keys: List of unstructured metadata dictionary keys
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at path: {path} (resolved: {resolved_path})",
                "path": str(path),
            }

        import anndata

        # Read in backed mode first for lightweight inspection
        try:
            adata = anndata.read_h5ad(resolved_path, backed="r")
            is_backed = True
        except Exception:
            adata = anndata.read_h5ad(resolved_path)
            is_backed = False

        obs_summary: dict[str, Any] = {}
        for col in adata.obs.columns:
            series = adata.obs[col]
            n_unique = int(series.nunique())
            summary_entry: dict[str, Any] = {"dtype": str(series.dtype), "n_unique": n_unique}
            if n_unique <= max_categories or series.dtype.name in ("category", "object", "string", "bool"):
                top_vals = series.value_counts(dropna=False).head(max_categories).to_dict()
                summary_entry["top_categories"] = _sanitize_val(top_vals)
            obs_summary[str(col)] = summary_entry

        result = {
            "status": "success",
            "path": str(resolved_path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_columns": [str(c) for c in adata.obs.columns],
            "obs_summary": obs_summary,
            "var_columns": [str(c) for c in adata.var.columns],
            "layers": [str(k) for k in adata.layers.keys()],
            "obsm_keys": [str(k) for k in adata.obsm.keys()],
            "obsp_keys": [str(k) for k in adata.obsp.keys()],
            "uns_keys": [str(k) for k in adata.uns.keys()],
        }

        if is_backed and hasattr(adata, "file") and hasattr(adata.file, "close"):
            adata.file.close()

        return result
    except Exception as exc:
        logger.error("inspect_anndata failed on %s: %s", path, exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }


def embed_cells(
    path: str,
    models: list[str] | str = "pca",
    batch_size: int = 32,
    output_key: str | None = None,
    output_path: str | None = None,
    preprocessing: str = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    """Generate biological cell embeddings across one or more models using BioEmbedder.

    Loads a single-cell AnnData (.h5ad) dataset, prepares counts, and runs model
    inference (e.g., PCA, scVI, scGPT, Geneformer) to populate adata.obsm.

    Parameters
    ----------
    path : str
        Filesystem path to the input .h5ad AnnData file.
    models : list[str] or str, default="pca"
        Model roster to execute (e.g., ["pca"], ["pca", "scvi"], or "pca").
    batch_size : int, default=32
        Batch size for inference.
    output_key : str, optional
        Custom .obsm key for the computed embedding (if a single model is run).
    output_path : str, optional
        Filesystem path where the updated AnnData file should be saved.
        If None, the embeddings are computed and shapes returned without saving.
    preprocessing : str, default="auto"
        Preprocessing mode: 'auto', 'standard', 'raw', or 'none'.
    device : str, default="auto"
        Computation device: 'auto', 'cpu', or 'cuda'.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting execution status and embedding metadata:
        - status: "success" or "error"
        - input_path: Input dataset path
        - output_path: Saved output dataset path (or None)
        - n_cells: Number of embedded cells
        - models_computed: List of models executed
        - embedding_keys: Available .obsm embedding keys
        - embedding_shapes: Shapes of the computed embedding matrices
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"Input file not found at path: {path} (resolved: {resolved_path})",
                "path": str(path),
            }

        model_list = [models] if isinstance(models, str) else list(models)
        if not model_list:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "The models list cannot be empty. Specify at least one model, e.g. ['pca'].",
                "path": str(path),
            }

        import anndata
        from embpy import BioEmbedder

        adata = anndata.read_h5ad(resolved_path)

        # Divert any third-party prints or warnings away from stdout
        with contextlib.redirect_stdout(sys.stderr):
            embedder = BioEmbedder(device=device)

            extra_kwargs: dict[str, Any] = {}
            # Small test datasets (< 2000 genes) should not trigger HVG bin errors
            if adata.n_vars < 2000 and preprocessing == "auto":
                extra_kwargs["n_top_genes"] = max(10, adata.n_vars - 1)
                extra_kwargs["pca_use_hvg"] = False

            adata_out = embedder.embed_cells(
                adata,
                models=model_list,
                batch_size=batch_size,
                preprocessing=preprocessing,
                obsm_prefix="X_",
                **extra_kwargs,
            )

        if output_key and len(model_list) == 1:
            canonical_key = f"X_{model_list[0]}"
            if canonical_key in adata_out.obsm:
                adata_out.obsm[output_key] = adata_out.obsm[canonical_key]

        saved_path: str | None = None
        if output_path:
            saved_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
            adata_out.write_h5ad(saved_path)

        shapes = {str(k): [int(dim) for dim in adata_out.obsm[k].shape] for k in adata_out.obsm.keys()}

        return {
            "status": "success",
            "input_path": str(resolved_path),
            "output_path": saved_path,
            "n_cells": int(adata_out.n_obs),
            "models_computed": [str(m) for m in model_list],
            "embedding_keys": [str(k) for k in adata_out.obsm.keys()],
            "embedding_shapes": shapes,
        }
    except Exception as exc:
        msg = str(exc)
        is_oom = "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower()
        if is_oom:
            return {
                "status": "error",
                "error_type": "ModelOOMError",
                "error": (
                    f"Out of memory during model inference: {msg}. "
                    f"Actionable resolution: reduce batch_size (current: {batch_size}) or use device='cpu'."
                ),
                "path": str(path),
                "models": [str(m) for m in (model_list if "model_list" in locals() else [models])],
            }
        logger.error("embed_cells failed on %s: %s", path, exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": msg,
            "path": str(path),
            "models": [str(m) for m in (model_list if "model_list" in locals() else [models])],
        }


def evaluate_scib(
    path: str,
    embedding_keys: list[str] | str,
    label_key: str,
    batch_key: str | None = None,
    output_csv: str | None = None,
) -> dict[str, Any]:
    """Score one or more embeddings using scIB metrics and return a comparison table.

    Evaluates representations on bio-conservation (NMI, ARI, ASW label)
    and batch-correction (ASW batch, graph connectivity, iLISI, kBET) against
    ground-truth annotations.

    Parameters
    ----------
    path : str
        Filesystem path to the .h5ad AnnData file containing embeddings in .obsm.
    embedding_keys : list[str] or str
        The .obsm keys to evaluate (e.g., ["X_pca"] or ["X_pca", "X_scvi"]).
    label_key : str
        Column name in adata.obs holding ground-truth cell type / cluster labels.
    batch_key : str, optional
        Column name in adata.obs holding technical batch annotations.
    output_csv : str, optional
        Filesystem path to export the resulting metrics comparison table as CSV.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting evaluation status, summary scores, and metrics:
        - status: "success" or "error"
        - path: Dataset path
        - embedding_keys: Evaluated .obsm keys
        - label_key: Ground-truth label column
        - batch_key: Batch column (or None)
        - output_csv: Path to saved CSV (or None)
        - summary: List of per-model metric records sorted by total score descending
        - metrics: Dictionary mapping each embedding key to its metric results
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at path: {path} (resolved: {resolved_path})",
                "path": str(path),
            }

        keys = [embedding_keys] if isinstance(embedding_keys, str) else list(embedding_keys)
        if not keys:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "The embedding_keys parameter cannot be empty.",
                "path": str(path),
            }

        import anndata
        from embpy.tl import compute_scib_metrics

        adata = anndata.read_h5ad(resolved_path)

        if label_key not in adata.obs:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"label_key '{label_key}' not found in adata.obs. Available columns: {list(adata.obs.columns)}",
                "path": str(path),
            }

        if batch_key and batch_key not in adata.obs:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"batch_key '{batch_key}' not found in adata.obs. Available columns: {list(adata.obs.columns)}",
                "path": str(path),
            }

        missing_keys = [k for k in keys if k not in adata.obsm]
        if missing_keys:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"Embedding keys {missing_keys} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}",
                "path": str(path),
            }

        with contextlib.redirect_stdout(sys.stderr):
            metrics_df = compute_scib_metrics(
                adata,
                embedding_keys=keys,
                label_key=label_key,
                batch_key=batch_key,
                verbose=False,
            )

        saved_csv: str | None = None
        if output_csv:
            saved_csv = os.path.abspath(output_csv)
            os.makedirs(os.path.dirname(saved_csv) or ".", exist_ok=True)
            metrics_df.to_csv(saved_csv)

        summary_records = _sanitize_val(metrics_df.reset_index().to_dict(orient="records"))
        metrics_dict = _sanitize_val(metrics_df.to_dict(orient="index"))

        return {
            "status": "success",
            "path": str(resolved_path),
            "embedding_keys": [str(k) for k in keys],
            "label_key": str(label_key),
            "batch_key": str(batch_key) if batch_key else None,
            "output_csv": saved_csv,
            "summary": summary_records,
            "metrics": metrics_dict,
        }
    except Exception as exc:
        logger.error("evaluate_scib failed on %s: %s", path, exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
