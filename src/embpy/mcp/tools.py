"""Core bioinformatics skills for the embpy Model Context Protocol (MCP) server.

Provides 7 comprehensive tools for single-cell genomics and foundation-model analysis:
1. inspect_anndata: Profile dataset dimensions, metadata, layers, and count vs normalized expression nature.
2. list_available_models: Query single-cell foundation models, compute devices, and optional dependency extras.
3. map_gene_identifiers: Map and validate gene IDs between Ensembl and HGNC symbols using embpy GeneResolver.
4. embed_cells: Run multi-model single-cell embeddings via BioEmbedder (PCA, scVI, foundation models).
5. evaluate_scib: Benchmark embeddings on biological conservation and batch integration via scIB.
6. evaluate_perturbations: Score pairwise perturbation predictions against real data using cell-eval.
7. compute_umap: Compute kNN graph and 2D UMAP coordinates on chosen embedding representation.

All operations enforce stdio isolation by redirecting non-protocol outputs to stderr
to guarantee pure, uncorrupted JSON-RPC transmission over stdout.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import sys
from typing import Any, Literal

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


def _detect_expression_nature(adata: Any) -> dict[str, Any]:
    """Inspect AnnData to determine if .X holds raw integer counts or normalized expression."""
    has_counts_layer = "counts" in adata.layers
    has_norm_layer = any(k in adata.layers for k in ("log_normalized", "normalized", "log1p"))

    nature = "unknown"
    is_integer = False
    has_negative = False
    min_val: float | None = None
    max_val: float | None = None

    try:
        import scipy.sparse as sp

        # Sample up to 50 cells
        n_sample = min(50, adata.n_obs)
        X_sample = adata.X[:n_sample]
        if sp.issparse(X_sample):
            data_arr = X_sample.data[:1000] if len(X_sample.data) > 0 else np.array([])
        else:
            data_arr = np.ravel(X_sample)[:1000]

        if len(data_arr) > 0:
            min_val = float(np.min(data_arr))
            max_val = float(np.max(data_arr))
            has_negative = bool(min_val < -1e-5)
            is_integer = bool(np.all(np.isclose(data_arr, np.round(data_arr), atol=1e-4)))

            if has_negative:
                nature = "scaled_or_centered"
            elif is_integer and min_val >= 0:
                nature = "raw_counts"
            elif max_val <= 25.0 and not is_integer:
                nature = "log_normalized"
            else:
                nature = "continuous_normalized"
    except Exception as exc:
        logger.debug("Could not sample .X for expression detection: %s", exc)
        nature = "could_not_sample"

    return {
        "detected_nature": nature,
        "is_discrete_integers": is_integer,
        "has_negative_values": has_negative,
        "sample_min": min_val,
        "sample_max": max_val,
        "has_counts_layer": has_counts_layer,
        "has_normalized_layer": has_norm_layer,
    }


def inspect_anndata(path: str, max_categories: int = 10) -> dict[str, Any]:
    """Inspect metadata, dimensions, expression nature, and slots of an AnnData (.h5ad) dataset.

    Reads dataset metadata without loading full matrices into memory when
    possible, reporting cell and gene counts, observation columns (.obs),
    feature variables (.var), expression layers, embeddings (.obsm),
    and whether .X represents raw counts or normalized expression.

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
        - expression_nature: Assessment of raw count vs normalized data
        - obs_columns: Column names present in adata.obs
        - obs_summary: High-level summary of observation annotations
        - var_columns: Column names present in adata.var
        - var_sample: Sample of gene identifiers in adata.var_names
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

        expression_info = _detect_expression_nature(adata)

        var_sample = [str(x) for x in list(adata.var_names[:10])]

        result = {
            "status": "success",
            "path": str(resolved_path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "expression_nature": expression_info,
            "obs_columns": [str(c) for c in adata.obs.columns],
            "obs_summary": obs_summary,
            "var_columns": [str(c) for c in adata.var.columns],
            "var_sample": var_sample,
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


def list_available_models() -> dict[str, Any]:
    """Query embpy registry to report installed models, hardware devices, and optional extras.

    Inspects available single-cell foundation models (e.g. PCA, scVI, scGPT, Geneformer,
    STATE), detects the optimal computing hardware (CUDA GPU, Apple Silicon MPS, or CPU),
    and audits which optional dependency extras are installed or missing.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting model and environment status:
        - status: "success" or "error"
        - compute_device: Detected PyTorch device ('cuda', 'mps', or 'cpu')
        - device_details: Hardware device name or CPU specification
        - runnable_models: Models ready to run in the current environment
        - all_singlecell_models: Full roster of registered single-cell foundation models
        - optional_extras_status: Audit of optional dependency bundles and install commands
    """
    try:
        from embpy.embedder import get_device
        from embpy.models.singlecell_models import list_singlecell_models

        device = get_device()
        device_str = str(device.type)
        device_name = "CPU"
        try:
            import torch

            if device.type == "cuda" and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
            elif device.type == "mps":
                device_name = "Apple Silicon MPS"
        except Exception:
            pass

        all_models = list_singlecell_models()

        # Audit optional extras
        extras = {
            "mcp": {
                "installed": importlib.util.find_spec("mcp") is not None,
                "pip": "pip install 'embpy[mcp]'",
                "feature": "Model Context Protocol stdio server",
            },
            "scib": {
                "installed": importlib.util.find_spec("scib") is not None,
                "pip": "pip install 'embpy[scib]'",
                "feature": "scIB single-cell embedding quality metrics",
            },
            "eval": {
                "installed": importlib.util.find_spec("cell_eval") is not None,
                "pip": "pip install 'embpy[eval]'",
                "feature": "ArcInstitute cell-eval perturbation metrics",
            },
            "genome": {
                "installed": importlib.util.find_spec("pyensembl") is not None,
                "pip": "pip install 'embpy[genome]'",
                "feature": "Local Ensembl offline genome resolving",
            },
            "seqmodels": {
                "installed": (
                    importlib.util.find_spec("borzoi_pytorch") is not None
                    and importlib.util.find_spec("enformer_pytorch") is not None
                ),
                "pip": "pip install 'embpy[seqmodels]'",
                "feature": "Borzoi & Enformer DNA regulatory models",
            },
            "pertpy": {
                "installed": importlib.util.find_spec("pertpy") is not None,
                "pip": "pip install 'embpy[pertpy]'",
                "feature": "Perturbation biology annotators & drug lookups",
            },
        }

        # Determine which models can run given dependencies
        runnable = ["pca"]
        if importlib.util.find_spec("scvi") is not None:
            runnable.extend(["scvi", "scanvi", "totalvi"])
        if importlib.util.find_spec("scgpt") is not None:
            runnable.append("scgpt")
        if importlib.util.find_spec("geneformer") is not None:
            runnable.extend([m for m in all_models if "geneformer" in m])
        if importlib.util.find_spec("arc_state") is not None:
            runnable.append("state")
        if importlib.util.find_spec("arc_stack") is not None:
            runnable.append("stack")

        return {
            "status": "success",
            "compute_device": device_str,
            "device_details": device_name,
            "runnable_models": runnable,
            "all_singlecell_models": all_models,
            "optional_extras_status": extras,
        }
    except Exception as exc:
        logger.error("list_available_models failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def map_gene_identifiers(
    genes: list[str] | None = None,
    path: str | None = None,
    target_type: Literal["ensembl", "symbol"] = "ensembl",
    organism: str = "human",
    var_column: str | None = None,
    update_h5ad_path: str | None = None,
) -> dict[str, Any]:
    """Validate and convert gene identifiers between Ensembl IDs and HGNC symbols.

    Harmonizes gene vocabularies between Ensembl gene IDs (e.g. 'ENSG00000141510')
    and HGNC symbols (e.g. 'TP53') using embpy GeneResolver. Resolves either an
    explicit list of genes or the var_names of an AnnData (.h5ad) dataset.

    Parameters
    ----------
    genes : list[str], optional
        List of gene identifier strings to convert.
    path : str, optional
        Filesystem path to an AnnData (.h5ad) file whose variables to map.
    target_type : {'ensembl', 'symbol'}, default='ensembl'
        Target identifier format: 'ensembl' (converts symbols to Ensembl) or
        'symbol' (converts Ensembl IDs to symbols).
    organism : str, default='human'
        Target species ('human', 'mouse', etc.).
    var_column : str, optional
        Column in adata.var containing gene identifiers (defaults to adata.var_names).
    update_h5ad_path : str, optional
        If provided, writes the updated AnnData file with mapped IDs added into adata.var.

    Returns
    -------
    dict[str, Any]
        Conversion status and summary dictionary:
        - status: "success" or "error"
        - target_type: Target ID format
        - organism: Selected species
        - total_genes: Count of input genes
        - mapped_count: Number of successfully mapped genes
        - unmapped_count: Number of genes that could not be mapped
        - mapping_rate: Percentage of genes resolved
        - mapping_sample: Preview dictionary of input to output mappings
        - unmapped_sample: List of up to 10 unmapped gene identifiers
        - output_path: Saved updated AnnData path (or None)
    """
    try:
        input_genes: list[str] = []
        adata = None

        if path:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"AnnData file not found: {path}",
                }
            import anndata

            adata = anndata.read_h5ad(resolved_path)
            if var_column and var_column in adata.var:
                input_genes = [str(x) for x in adata.var[var_column].dropna()]
            else:
                input_genes = [str(x) for x in adata.var_names]
        elif genes:
            input_genes = [str(g) for g in genes]
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Either 'genes' (list of gene strings) or 'path' (.h5ad file) must be provided.",
            }

        from embpy.resources.gene.resolver import GeneResolver

        with contextlib.redirect_stdout(sys.stderr):
            resolver = GeneResolver(species=organism)
            # Batch mapping
            if target_type == "ensembl":
                mapping_result = resolver.symbols_to_ensembl_batch(input_genes, organism=organism)
            else:
                mapping_result = resolver.ensembl_to_symbols_batch(input_genes, organism=organism)

        mapped_dict = {k: v for k, v in mapping_result.items() if v is not None}
        unmapped = [k for k, v in mapping_result.items() if v is None]
        total = len(input_genes)
        mapped_count = len(mapped_dict)
        mapping_rate = (mapped_count / total * 100.0) if total > 0 else 0.0

        saved_path: str | None = None
        if update_h5ad_path and adata is not None:
            saved_path = os.path.abspath(update_h5ad_path)
            col_name = f"gene_{target_type}"
            if var_column:
                adata.var[col_name] = adata.var[var_column].map(mapped_dict)
            else:
                adata.var[col_name] = [mapped_dict.get(name) for name in adata.var_names]
            os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
            adata.write_h5ad(saved_path)

        preview = dict(list(mapped_dict.items())[:15])

        return {
            "status": "success",
            "target_type": target_type,
            "organism": organism,
            "total_genes": total,
            "mapped_count": mapped_count,
            "unmapped_count": len(unmapped),
            "mapping_rate": round(mapping_rate, 2),
            "mapping_sample": preview,
            "unmapped_sample": unmapped[:10],
            "output_path": saved_path,
        }
    except Exception as exc:
        logger.error("map_gene_identifiers failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
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

        with contextlib.redirect_stdout(sys.stderr):
            embedder = BioEmbedder(device=device)

            extra_kwargs: dict[str, Any] = {}
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
    """Score one or more embeddings using scIB metrics and return a formatted markdown comparison table.

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
        Dictionary reporting evaluation status, summary scores, and markdown table:
        - status: "success" or "error"
        - path: Dataset path
        - embedding_keys: Evaluated .obsm keys
        - label_key: Ground-truth label column
        - batch_key: Batch column (or None)
        - output_csv: Path to saved CSV (or None)
        - markdown_table: Formatted Markdown string of ranking metrics for agent display
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

        # Build clean markdown representation
        try:
            markdown_table = metrics_df.round(4).to_markdown()
        except Exception:
            # Fallback simple markdown table builder if tabulate is unavailable
            cols = list(metrics_df.columns)
            hdr = "| Embedding | " + " | ".join(cols) + " |\n"
            sep = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |\n"
            rows = []
            for idx, row in metrics_df.iterrows():
                row_str = f"| **{idx}** | " + " | ".join([f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v) for v in row]) + " |"
                rows.append(row_str)
            markdown_table = hdr + sep + "\n".join(rows)

        return {
            "status": "success",
            "path": str(resolved_path),
            "embedding_keys": [str(k) for k in keys],
            "label_key": str(label_key),
            "batch_key": str(batch_key) if batch_key else None,
            "output_csv": saved_csv,
            "markdown_table": markdown_table,
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


def evaluate_perturbations(
    path: str,
    path_pred: str | None = None,
    control_pert: str = "control",
    pert_col: str = "perturbation",
    embedding_keys: list[str] | str | None = None,
    profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "minimal",
    split_col: str | None = None,
    output_csv: str | None = None,
) -> dict[str, Any]:
    """Score predicted vs real single-cell perturbation effects using ArcInstitute cell-eval.

    Evaluates differential expression, mean absolute error (MAE), and pairwise
    discrimination metrics between real perturbation datasets and model predictions.
    Can accept two separate AnnData files (real and predicted) or a single AnnData
    with a train/test or real/pred split column.

    Parameters
    ----------
    path : str
        Filesystem path to real AnnData (or combined dataset if split_col is specified).
    path_pred : str, optional
        Filesystem path to predicted AnnData file.
    control_pert : str, default='control'
        Identifier for control condition in adata.obs[pert_col].
    pert_col : str, default='perturbation'
        Column name in adata.obs identifying perturbation conditions.
    embedding_keys : list[str] or str, optional
        Target .obsm embedding keys to evaluate for discrimination scores.
    profile : {'full', 'vcc', 'minimal', 'de', 'anndata'}, default='minimal'
        Evaluation profile to run.
    split_col : str, optional
        If path_pred is None, column in adata.obs used to split into pred and real
        (e.g., 'split' where values are 'pred'/'test' and 'real'/'train').
    output_csv : str, optional
        Filesystem path to save the evaluation results as a CSV file.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting perturbation evaluation scores:
        - status: "success" or "error"
        - control_pert: Control perturbation identifier
        - pert_col: Perturbation column
        - n_perturbations: Number of perturbations evaluated
        - markdown_table: Formatted Markdown string of perturbation metric scores
        - results: List of per-perturbation score dictionaries
        - output_csv: Path to saved CSV (or None)
    """
    try:
        resolved_real = os.path.abspath(path)
        if not os.path.exists(resolved_real):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"Dataset file not found at: {path}",
            }

        import anndata
        from embpy.tl import run_cell_eval

        if path_pred:
            resolved_pred = os.path.abspath(path_pred)
            if not os.path.exists(resolved_pred):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"Predicted dataset file not found at: {path_pred}",
                }
            adata_real = anndata.read_h5ad(resolved_real)
            adata_pred = anndata.read_h5ad(resolved_pred)
        elif split_col:
            adata_full = anndata.read_h5ad(resolved_real)
            if split_col not in adata_full.obs:
                return {
                    "status": "error",
                    "error_type": "KeyError",
                    "error": f"split_col '{split_col}' not found in adata.obs.",
                }
            unique_splits = list(adata_full.obs[split_col].unique())
            if len(unique_splits) < 2:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": f"split_col '{split_col}' must contain at least 2 categories.",
                }
            pred_mask = adata_full.obs[split_col].isin(["pred", "test", "predicted"])
            if not pred_mask.any():
                pred_mask = adata_full.obs[split_col] == unique_splits[0]
            adata_pred = adata_full[pred_mask].copy()
            adata_real = adata_full[~pred_mask].copy()
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Either 'path_pred' (path to predicted AnnData) or 'split_col' must be specified.",
            }

        if pert_col not in adata_real.obs or pert_col not in adata_pred.obs:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"pert_col '{pert_col}' must exist in both real and predicted observations.",
            }

        extra_configs: dict[str, Any] = {}
        if embedding_keys:
            key_list = [embedding_keys] if isinstance(embedding_keys, str) else list(embedding_keys)
            extra_configs["metric_configs"] = {
                "discrimination_score_cosine": {"embed_key": key_list[0]}
            }

        with contextlib.redirect_stdout(sys.stderr):
            df_results = run_cell_eval(
                adata_pred=adata_pred,
                adata_real=adata_real,
                control_pert=control_pert,
                pert_col=pert_col,
                profile=profile,
                **extra_configs,
            )

        saved_csv: str | None = None
        if output_csv:
            saved_csv = os.path.abspath(output_csv)
            os.makedirs(os.path.dirname(saved_csv) or ".", exist_ok=True)
            df_results.to_csv(saved_csv)

        summary_records = _sanitize_val(df_results.reset_index().to_dict(orient="records"))

        try:
            markdown_table = df_results.round(4).to_markdown()
        except Exception:
            cols = list(df_results.columns)
            hdr = "| " + " | ".join(cols) + " |\n"
            sep = "| " + " | ".join([":---:" for _ in cols]) + " |\n"
            rows = []
            for _, row in df_results.iterrows():
                row_str = "| " + " | ".join([f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v) for v in row]) + " |"
                rows.append(row_str)
            markdown_table = hdr + sep + "\n".join(rows)

        return {
            "status": "success",
            "control_pert": control_pert,
            "pert_col": pert_col,
            "n_perturbations": len(df_results),
            "markdown_table": markdown_table,
            "results": summary_records,
            "output_csv": saved_csv,
        }
    except Exception as exc:
        logger.error("evaluate_perturbations failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }


def compute_umap(
    path: str,
    embedding_key: str = "X_pca",
    output_key: str | None = None,
    output_path: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute kNN neighbor graph and 2D UMAP coordinates on a chosen embedding representation.

    Constructs a kNN neighborhood graph over an existing .obsm representation
    (e.g., 'X_pca', 'X_scvi') and runs UMAP to produce 2D coordinates for visual
    inspection, storing results under .obsm['X_umap_<model>'] or a custom key.

    Parameters
    ----------
    path : str
        Filesystem path to the .h5ad AnnData file containing embeddings.
    embedding_key : str, default='X_pca'
        The .obsm matrix to compute the neighbor graph and UMAP on.
    output_key : str, optional
        Custom .obsm key for the computed coordinates. Defaults to 'X_umap_<model>'.
    output_path : str, optional
        Filesystem path to save the updated AnnData file. If None, computes and
        reports dimensions without saving.
    n_neighbors : int, default=15
        Number of neighbors for kNN graph construction.
    min_dist : float, default=0.5
        Effective minimum distance between embedded points for UMAP.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary reporting UMAP calculation status:
        - status: "success" or "error"
        - path: Dataset path
        - embedding_key: Base representation used
        - umap_key: Final key written in adata.obsm
        - n_cells: Total cell count
        - umap_shape: Dimensions of generated 2D coordinate array
        - output_path: Path to saved file (or None)
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at: {path}",
            }

        import anndata
        import scanpy as sc

        adata = anndata.read_h5ad(resolved_path)

        if embedding_key not in adata.obsm:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"embedding_key '{embedding_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}",
            }

        target_key = output_key or f"X_umap_{embedding_key.replace('X_', '')}"

        # Adjust n_neighbors for small datasets
        effective_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))

        with contextlib.redirect_stdout(sys.stderr):
            sc.pp.neighbors(
                adata,
                use_rep=embedding_key,
                n_neighbors=effective_neighbors,
                random_state=random_state,
            )
            sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
            adata.obsm[target_key] = adata.obsm["X_umap"].copy()

        saved_path: str | None = None
        if output_path:
            saved_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
            adata.write_h5ad(saved_path)

        return {
            "status": "success",
            "path": str(resolved_path),
            "embedding_key": str(embedding_key),
            "umap_key": str(target_key),
            "n_cells": int(adata.n_obs),
            "umap_shape": [int(d) for d in adata.obsm[target_key].shape],
            "output_path": saved_path,
        }
    except Exception as exc:
        logger.error("compute_umap failed on %s: %s", path, exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
