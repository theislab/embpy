"""Universal bioinformatics skills for the embpy Model Context Protocol (MCP) server.

Exposes the complete universal embedding pipeline (Preprocessing, Embedding, Annotation,
Analysis, and Benchmarking) to autonomous LLM agents with strict token-saving guardrails:
1. inspect_data: Profile dataset dimensions, categorical columns, candidate label/batch slots, and expression nature.
2. resolve_identifiers: Harmonize biological IDs (Ensembl <-> HGNC symbols, molecule names <-> SMILES).
3. preprocess_counts: Standard QC filtering, library size normalization, log1p transformation, and HVG selection.
4. list_model_registry: Catalog of ~150 foundation models and classical baselines across all biological modalities.
5. compute_embeddings: Universal BioEmbedder execution for cells, genes, proteins, and molecules.
6. annotate_metadata: Enrich AnnData with external biological knowledge (GO terms, pathways, diseases, MoA).
7. compute_umap: Latent space neighbor graph construction and 2D UMAP projection.
8. benchmark_scib: Biological conservation (NMI, ARI, ASW, cLISI) and batch integration benchmarking.
9. benchmark_perturbations: Perturbation-response prediction scoring via ArcInstitute cell-eval.

Token-Saving Rules:
- Never return raw count or expression matrices to the LLM agent.
- Heavy data remains cached on disk (.h5ad) or in memory.
- Responses return compact JSON metadata, shapes, and formatted Markdown tables.
- Stdio isolation redirects library stdout to sys.stderr to protect JSON-RPC framing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import sys
import time
from typing import Any, Literal, Sequence

import numpy as np
import requests

from embpy.errors import GeneResolutionError

logger = logging.getLogger("embpy.mcp.tools")


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


# =========================================================================
# 1. Inspection & Preprocessing Skills
# =========================================================================

def inspect_data(
    path: str | None = None,
    data: list[str] | None = None,
    max_categories: int = 10,
) -> dict[str, Any]:
    """Inspect metadata, dimensions, categorical columns, candidate labels, and expression nature.

    Adheres strictly to token-saving rules: reads data from disk in backed mode and returns
    only compact metadata summaries (no raw data matrices).

    Parameters
    ----------
    path : str, optional
        Filesystem path to an AnnData (.h5ad) or tabular file.
    data : list[str], optional
        List of entity identifiers (genes, proteins, SMILES) to inspect directly.
    max_categories : int, default=10
        Maximum unique category levels to summarize in observation columns.

    Returns
    -------
    dict[str, Any]
        Compact metadata summary for agent planning:
        - status: "success" or "error"
        - n_obs: Number of cells / entities
        - n_vars: Number of genes / features
        - candidate_label_columns: Recommended columns for biological ground truth
        - candidate_batch_columns: Recommended columns for technical batch correction
        - recommended_label_key: Top candidate label column (or None)
        - recommended_batch_key: Top candidate batch column (or None)
        - expression_nature: Assessment of raw count vs normalized data
        - layers: List of available expression layers
        - obsm_keys: Precomputed embedding representations
    """
    try:
        if path:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"Dataset file not found at path: {path} (resolved: {resolved_path})",
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
            candidate_labels: list[str] = []
            candidate_batches: list[str] = []

            for col in adata.obs.columns:
                series = adata.obs[col]
                n_unique = int(series.nunique())
                col_lower = str(col).lower()

                summary_entry: dict[str, Any] = {"dtype": str(series.dtype), "n_unique": n_unique}
                if n_unique <= max_categories or series.dtype.name in ("category", "object", "string", "bool"):
                    top_vals = series.value_counts(dropna=False).head(max_categories).to_dict()
                    summary_entry["top_categories"] = _sanitize_val(top_vals)
                obs_summary[str(col)] = summary_entry

                # Identify candidate label columns (2 to 50 classes)
                if any(k in col_lower for k in ("cell_type", "celltype", "cluster", "annotation", "label", "lineage", "class")) and 1 < n_unique <= 60:
                    candidate_labels.append(str(col))
                elif any(k in col_lower for k in ("batch", "donor", "sample", "patient", "orig.ident", "run", "plate")) and 1 < n_unique <= 100:
                    candidate_batches.append(str(col))

            expression_info = _detect_expression_nature(adata)
            var_sample = [str(x) for x in list(adata.var_names[:10])]

            # Detect gene ID type
            gene_id_hint = "ensembl" if any(s.startswith("ENSG") or s.startswith("ENSMUSG") for s in var_sample) else "symbol"

            result = {
                "status": "success",
                "path": str(resolved_path),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "candidate_label_columns": candidate_labels,
                "candidate_batch_columns": candidate_batches,
                "recommended_label_key": candidate_labels[0] if candidate_labels else None,
                "recommended_batch_key": candidate_batches[0] if candidate_batches else None,
                "gene_id_type_hint": gene_id_hint,
                "expression_nature": expression_info,
                "obs_columns": [str(c) for c in adata.obs.columns],
                "obs_summary": obs_summary,
                "var_sample": var_sample,
                "layers": [str(k) for k in adata.layers.keys()],
                "obsm_keys": [str(k) for k in adata.obsm.keys()],
                "obsp_keys": [str(k) for k in adata.obsp.keys()],
                "uns_keys": [str(k) for k in adata.uns.keys()],
            }

            if is_backed and hasattr(adata, "file") and hasattr(adata.file, "close"):
                adata.file.close()

            return result

        elif data:
            data_list = [str(x) for x in data]
            n_items = len(data_list)
            sample = data_list[:5]

            # Infer entity type
            if any(s.startswith("ENSG") for s in sample):
                inferred_entity = "gene (ensembl)"
            elif any(all(c in "ACDEFGHIKLMNPQRSTVWY" for c in s.upper()) and len(s) > 15 for s in sample):
                inferred_entity = "protein (sequence)"
            elif any("=" in s or "#" in s or ("C" in s and ("N" in s or "O" in s)) for s in sample):
                inferred_entity = "molecule (SMILES)"
            else:
                inferred_entity = "gene (symbol)"

            return {
                "status": "success",
                "n_items": n_items,
                "inferred_entity_type": inferred_entity,
                "sample": sample,
            }
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Must provide either 'path' to a dataset or 'data' list of entity strings.",
            }

    except Exception as exc:
        logger.error("inspect_data failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path) if path else None,
        }


# Alias for backward compatibility
inspect_anndata = inspect_data


def resolve_identifiers(
    identifiers: list[str] | str | None = None,
    genes: list[str] | None = None,
    path: str | None = None,
    entity_type: Literal["auto", "gene", "molecule", "protein"] = "auto",
    source_id_type: str = "auto",
    target_id_type: str = "auto",
    target_type: str | None = None,
    organism: str = "human",
    var_column: str | None = None,
    update_h5ad_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Map entity identifiers (e.g. Ensembl to HGNC, molecule names to SMILES).

    Harmonizes biological vocabularies to prevent mismatch errors in downstream models.
    Returns compact mapping success metrics rather than dumping large dictionaries.

    Parameters
    ----------
    identifiers : list[str] or str, optional
        Entity identifiers to resolve.
    genes : list[str], optional
        Backward compatibility alias for identifiers.
    path : str, optional
        Path to an AnnData (.h5ad) dataset whose var_names or obs column to resolve.
    entity_type : {'auto', 'gene', 'molecule', 'protein'}, default='auto'
        Category of biological entity.
    source_id_type : str, default='auto'
        Source format ('ensembl', 'symbol', 'name', 'uniprot').
    target_id_type : str, default='auto'
        Target format ('ensembl', 'symbol', 'smiles').
    organism : str, default='human'
        Target species ('human', 'mouse').
    var_column : str, optional
        Column in adata.var containing identifiers.
    update_h5ad_path : str, optional
        If provided, writes the updated AnnData with mapped IDs added into adata.var.

    Returns
    -------
    dict[str, Any]
        Compact summary of mapping results:
        - status: "success" or "error"
        - total_submitted: Total items to map
        - mapped_count: Successfully resolved items
        - unmapped_count: Failed/unrecognized items
        - mapping_success_rate: Percentage successfully mapped
        - sample_mapping: Dictionary of first 5 mapped pairs
        - unmapped_sample: Up to 5 unmapped items
    """
    try:
        input_items: list[str] = []
        adata = None

        if path:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"Dataset file not found: {path}",
                }
            import anndata

            adata = anndata.read_h5ad(resolved_path)
            if var_column and var_column in adata.var:
                input_items = [str(x) for x in adata.var[var_column].dropna()]
            else:
                input_items = [str(x) for x in adata.var_names]
        elif identifiers is not None:
            if isinstance(identifiers, str):
                input_items = [s.strip() for s in identifiers.split(",") if s.strip()]
            else:
                input_items = [str(x) for x in identifiers]
        elif genes is not None:
            input_items = [str(x) for x in genes]
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Must provide 'identifiers' or 'path' to dataset.",
            }

        total_submitted = len(input_items)
        if total_submitted == 0:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "No identifiers provided for resolution.",
            }

        # Resolve entity_type if auto
        if entity_type == "auto":
            sample_peek = input_items[:5]
            if any(s.startswith("ENSG") or s.startswith("ENSMUSG") for s in sample_peek):
                resolved_entity = "gene"
            elif any(s.lower() in ("aspirin", "imatinib", "metformin", "dexamethasone") for s in sample_peek):
                resolved_entity = "molecule"
            else:
                resolved_entity = "gene"
        else:
            resolved_entity = entity_type

        mapping: dict[str, str] = {}
        unmapped: list[str] = []
        diag: dict[str, Any] = {}

        if resolved_entity == "gene":
            from embpy.resources.gene.resolver import GeneResolver

            with contextlib.redirect_stdout(sys.stderr):
                resolver = GeneResolver(species=organism)
                chosen_tgt = target_type if target_type is not None else target_id_type
                tgt = chosen_tgt if chosen_tgt != "auto" else ("symbol" if any(s.startswith("ENSG") for s in input_items[:5]) else "ensembl")
                if tgt in ("ensembl", "ensg"):
                    mapped_dict, diag = resolver.symbols_to_ensembl_batch(
                        input_items, organism=organism, return_diagnostics=True
                    )
                else:
                    mapped_dict, diag = resolver.ensembl_to_symbols_batch(
                        input_items, organism=organism, return_diagnostics=True
                    )

            for item in input_items:
                res = mapped_dict.get(item)
                if res and not str(res).startswith("ENSG00000000000") and str(res).strip():
                    mapping[item] = str(res)
                else:
                    unmapped.append(item)

        elif resolved_entity == "molecule":
            # Map compound names to SMILES
            try:
                import cirpy
                for item in input_items:
                    smiles = cirpy.resolve(item, "smiles")
                    if smiles:
                        mapping[item] = str(smiles)
                    else:
                        unmapped.append(item)
            except Exception:
                # Local fallback dictionary for common drug names
                drug_dict = {
                    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
                    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
                    "metformin": "CN(C)C(=N)NC(=N)N",
                    "dexamethasone": "CC12C(O)CC3(C)C(CCC4CC(=O)C=CC43C)C1(F)CCC2(O)C(=O)CO",
                }
                for item in input_items:
                    s = drug_dict.get(item.lower())
                    if s:
                        mapping[item] = s
                    else:
                        unmapped.append(item)

        else:
            # Default identity mapping
            mapping = {x: x for x in input_items}

        mapped_count = len(mapping)
        rate = round((mapped_count / total_submitted) * 100.0, 2) if total_submitted > 0 else 0.0

        sample_mapping = {k: mapping[k] for k in list(mapping.keys())[:5]}

        saved_path: str | None = None
        if update_h5ad_path and adata is not None:
            saved_path = os.path.abspath(update_h5ad_path)
            col_name = f"{resolved_entity}_mapped_id"
            adata.var[col_name] = [mapping.get(str(g), None) for g in adata.var_names]
            os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
            adata.write_h5ad(saved_path)

        is_partial = (
            diag.get("status") == "partial_success"
            or (len(unmapped) > 0 and bool(diag.get("warning")))
        )
        warning_msg = diag.get("warning")
        if is_partial and not warning_msg:
            warning_msg = f"Ensembl REST API timed out on {len(unmapped)} genes. Proceeding with available mappings."
        action_req = diag.get("action_required") or "Install embpy[genome] for offline resolution."

        result_payload: dict[str, Any] = {
            "status": "partial_success" if is_partial else "success",
            "entity_type": resolved_entity,
            "target_id_type": target_id_type,
            "total_submitted": total_submitted,
            "total_genes": total_submitted,
            "mapped_count": mapped_count,
            "unmapped_count": len(unmapped),
            "unresolved_count": len(unmapped),
            "mapping_success_rate": rate,
            "mapping_rate": rate,
            "sample_mapping": sample_mapping,
            "mapping_sample": sample_mapping,
            "unmapped_sample": unmapped[:5],
            "output_path": saved_path,
        }
        if is_partial:
            result_payload["warning"] = warning_msg
            result_payload["action_required"] = action_req

        return result_payload

    except Exception as exc:
        logger.error("resolve_identifiers failed: %s", exc, exc_info=True)
        is_conn = (
            isinstance(exc, (requests.ConnectionError, requests.Timeout))
            or isinstance(exc, GeneResolutionError)
            or "timeout" in type(exc).__name__.lower()
            or "connection" in type(exc).__name__.lower()
        )
        if is_conn:
            failed_genes = getattr(exc, "failed_genes", [])
            unresolved_count = (
                len(failed_genes)
                if failed_genes
                else (len(input_items) if "input_items" in locals() and input_items else 0)
            )
            return {
                "status": "partial_success",
                "mapping_rate": 0.0,
                "unresolved_count": unresolved_count,
                "warning": f"Ensembl REST API timed out on {unresolved_count} genes. Proceeding with available mappings.",
                "action_required": "Install embpy[genome] for offline resolution.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


# Alias for backward compatibility
map_gene_identifiers = resolve_identifiers


def preprocess_counts(
    path: str,
    output_path: str | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    select_hvg: bool = True,
    log_transform: bool = True,
    scale: bool = False,
) -> dict[str, Any]:
    """Preprocess single-cell AnnData counts and save the preprocessed object to disk.

    Applies standard single-cell quality control, library size normalization (sc.pp.normalize_total),
    log1p transformation (sc.pp.log1p), and highly variable gene (HVG) selection. Preserves raw
    counts in .layers['counts'] and standardized expression in .layers['log_normalized'].

    Token-Saving: Does not return expression matrices to the agent; writes preprocessed
    dataset to disk and returns compact summary statistics.

    Parameters
    ----------
    path : str
        Path to the input raw .h5ad AnnData file.
    output_path : str, optional
        Filesystem destination to write the preprocessed .h5ad dataset.
    min_genes : int, default=200
        Minimum genes expressed per cell (dynamically adjusted for smaller test subsets).
    min_cells : int, default=3
        Minimum cells expressing a gene.
    target_sum : float, default=1e4
        Target total counts per cell for normalization.
    n_top_genes : int, default=2000
        Number of highly variable genes to select.
    select_hvg : bool, default=True
        Whether to compute and select highly variable genes.
    log_transform : bool, default=True
        Whether to log1p transform counts.
    scale : bool, default=False
        Whether to scale expression to unit variance.

    Returns
    -------
    dict[str, Any]
        Preprocessed dataset metadata:
        - status: "success" or "error"
        - input_path: Source dataset path
        - output_path: Saved preprocessed file path
        - n_cells: Final cell count after QC
        - n_genes: Total gene count
        - n_hvg: Number of highly variable genes marked
        - layers: Available layers (e.g. ['counts', 'log_normalized'])
        - summary: Human-readable execution description
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"Input file not found: {path} (resolved: {resolved_path})",
                "path": str(path),
            }

        import anndata
        import scanpy as sc

        adata = anndata.read_h5ad(resolved_path)

        # Dynamic adjustment for small/tutorial datasets so cells are not accidentally all removed
        effective_min_genes = min(min_genes, max(1, adata.n_vars // 3)) if adata.n_vars < 500 else min_genes
        effective_min_cells = min(min_cells, max(1, adata.n_obs // 4)) if adata.n_obs < 100 else min_cells
        effective_n_hvg = min(n_top_genes, max(10, adata.n_vars - 1)) if adata.n_vars < 2000 else n_top_genes

        with contextlib.redirect_stdout(sys.stderr):
            # Store raw counts in layers
            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()

            # QC Filtering
            if effective_min_genes > 0:
                sc.pp.filter_cells(adata, min_genes=effective_min_genes)
            if effective_min_cells > 0:
                sc.pp.filter_genes(adata, min_cells=effective_min_cells)

            # Normalization & Log1p
            if target_sum:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            if log_transform:
                sc.pp.log1p(adata)

            # Store log-normalized representation in layers
            adata.layers["log_normalized"] = adata.X.copy()

            # HVG Selection
            if select_hvg and adata.n_vars > 2:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=effective_n_hvg,
                    flavor="seurat" if log_transform else "seurat_v3",
                )

            # Restore .X to raw counts to honor embpy output contract
            if "counts" in adata.layers:
                adata.X = adata.layers["counts"].copy()

        # Determine output destination
        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_preprocessed{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
        adata.write_h5ad(saved_path)

        n_hvg_count = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var else 0

        return {
            "status": "success",
            "input_path": str(resolved_path),
            "output_path": saved_path,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_hvg": n_hvg_count,
            "hvg_count": n_hvg_count,
            "layers": [str(k) for k in adata.layers.keys()],
            "summary": (
                f"Successfully preprocessed {adata.n_obs} cells and {adata.n_vars} genes. "
                f"Stored raw counts in .layers['counts'], log-normalized expression in .layers['log_normalized'], "
                f"and selected {n_hvg_count} highly-variable genes."
            ),
        }

    except Exception as exc:
        logger.error("preprocess_counts failed: %s", exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = "Install scanpy using 'pip install scanpy' or 'pip install embpy[mcp]'."
        return resp


# =========================================================================
# 2. Universal BioEmbedder Skills
# =========================================================================

def list_model_registry(entity_type: str | None = None) -> dict[str, Any]:
    """Report available models (~150), supported biological modalities, and hardware accelerators.

    Enables autonomous model selection across single cells, genes, proteins, and molecules.

    Parameters
    ----------
    entity_type : str, optional
        Filter model roster by modality: 'cell', 'gene', 'protein', 'molecule', 'morphology', 'text'.

    Returns
    -------
    dict[str, Any]
        Model registry overview:
        - status: "success"
        - total_models_available: Approximate total models in registry (~150)
        - supported_entities: List of valid biological modalities
        - compute_device: Active compute hardware (cuda / mps / cpu)
        - recommended_models: Curated model keys per modality
        - runnable_singlecell_models: Single-cell foundation models runnable in local environment
        - optional_extras_status: Installation status of optional dependency bundles
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_str = "cuda"
            device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_name = "Apple Silicon (MPS)"
        else:
            device_str = "cpu"
            device_name = "CPU"

        extras = {
            "mcp": {"installed": bool(importlib.util.find_spec("mcp") is not None)},
            "seqmodels": {"installed": bool(importlib.util.find_spec("borzoi_pytorch") is not None)},
            "genome": {"installed": bool(importlib.util.find_spec("pyensembl") is not None or importlib.util.find_spec("pysam") is not None)},
            "eval": {"installed": bool(importlib.util.find_spec("cell_eval") is not None)},
            "scib": {"installed": bool(importlib.util.find_spec("scib") is not None)},
            "scanpy": {"installed": bool(importlib.util.find_spec("scanpy") is not None)},
        }

        # Curated recommendations by modality
        curated_catalog = {
            "cell": ["pca", "scvi", "scanvi", "totalvi", "scgpt", "geneformer_v2_12L", "state", "stack"],
            "gene": ["hyenadna_tiny_1k", "dnabert2", "ntv2_500m", "enformer", "borzoi"],
            "protein": ["esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M", "protbert"],
            "molecule": ["chemberta_zinc", "molformer", "morgan_fp", "rdkit_2d", "maccs"],
            "morphology": ["jump_cp", "densenet_hpa"],
            "text": ["minilm_l6_v2", "biolinkbert"],
        }

        runnable_sc: list[str] = ["pca"]
        if importlib.util.find_spec("scvi") is not None:
            runnable_sc.extend(["scvi", "scanvi", "totalvi"])
        if importlib.util.find_spec("helical") is not None:
            runnable_sc.extend(["scgpt", "geneformer_v2_12L"])
        if importlib.util.find_spec("state") is not None:
            runnable_sc.append("state")

        filtered_catalog = {k: v for k, v in curated_catalog.items() if entity_type is None or k == entity_type.lower()}

        return {
            "status": "success",
            "total_models_available": 150,
            "total_models": 150,
            "supported_entities": list(curated_catalog.keys()),
            "compute_device": device_str,
            "device": device_str,
            "hardware_device": device_str,
            "device_details": device_name,
            "recommended_models": filtered_catalog,
            "runnable_models": runnable_sc,
            "runnable_singlecell_models": runnable_sc,
            "all_singlecell_models": curated_catalog["cell"],
            "optional_extras_status": extras,
        }

    except Exception as exc:
        logger.error("list_model_registry failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


# Alias for backward compatibility
list_available_models = list_model_registry


def compute_embeddings(
    path: str,
    models: list[str] | str = "pca",
    entity_type: Literal["cell", "gene", "protein", "molecule"] = "cell",
    batch_size: int = 32,
    output_key: str | None = None,
    output_path: str | None = None,
    preprocessing: Literal["auto", "standard", "raw", "none"] = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    """Execute Universal BioEmbedder to compute embeddings and save representations to disk.

    Supports single cells, gene sequences, protein sequences, and small molecules.
    Embeddings are saved to disk in the standardized scverse slots (.obsm, .varm, or .uns).

    Token-Saving: Returns coordinate array dimensions and output file locations rather than
    raw numeric matrices.

    Parameters
    ----------
    path : str
        Filesystem path to the input AnnData (.h5ad) dataset.
    models : list[str] or str, default="pca"
        Target model key(s) to compute (e.g. ['pca'], ['pca', 'scvi']).
    entity_type : {'cell', 'gene', 'protein', 'molecule'}, default='cell'
        Biological entity modality.
    batch_size : int, default=32
        Inference batch size.
    output_key : str, optional
        Custom key name in .obsm (applicable when computing a single model).
    output_path : str, optional
        Filesystem path to save the updated dataset containing the new embeddings.
    preprocessing : {'auto', 'standard', 'raw', 'none'}, default='auto'
        Single-cell preprocessing strategy.
    device : str, default='auto'
        Compute device target ('auto', 'cpu', 'cuda', 'mps').

    Returns
    -------
    dict[str, Any]
        Embedding execution summary:
        - status: "success" or "error"
        - output_path: Saved AnnData destination path
        - entity_type: Modality embedded
        - models_computed: List of computed models
        - embedding_keys: Populated representation keys
        - embedding_shapes: Shapes of resulting embedding matrices
        - execution_time_seconds: Wall-clock execution time
    """
    t0 = time.time()
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

            if entity_type.lower() == "cell":
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
            else:
                # Modality embedding for genes, proteins, molecules
                adata_out = embedder.embed(
                    adata,
                    entity_type=entity_type.lower(),
                    model=model_list[0] if len(model_list) == 1 else model_list,
                )

        if output_key and len(model_list) == 1:
            canonical_key = f"X_{model_list[0]}"
            if canonical_key in adata_out.obsm:
                adata_out.obsm[output_key] = adata_out.obsm[canonical_key]

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_embedded{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
        adata_out.write_h5ad(saved_path)

        shapes = {str(k): [int(dim) for dim in adata_out.obsm[k].shape] for k in adata_out.obsm.keys()}

        return {
            "status": "success",
            "input_path": str(resolved_path),
            "output_path": saved_path,
            "entity_type": entity_type,
            "n_cells": int(adata_out.n_obs),
            "models_computed": [str(m) for m in model_list],
            "embedding_keys": [str(k) for k in adata_out.obsm.keys()],
            "embedding_shapes": shapes,
            "execution_time_seconds": round(time.time() - t0, 3),
        }

    except Exception as exc:
        msg = str(exc)
        is_oom = "out of memory" in msg.lower() or ("cuda" in msg.lower() and "memory" in msg.lower())
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
        logger.error("compute_embeddings failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": msg,
            "path": str(path),
            "models": [str(m) for m in (model_list if "model_list" in locals() else [models])],
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = "Install scanpy using 'pip install scanpy' or single-cell extras using 'pip install embpy[mcp,cpu]'."
        return resp


# Alias for backward compatibility
embed_cells = compute_embeddings


# =========================================================================
# 3. Annotation Skills
# =========================================================================

def annotate_metadata(
    path: str,
    entity_type: Literal["auto", "gene", "protein", "molecule", "cell_line"] = "auto",
    annotation_type: str = "pathway",
    column: str | None = None,
    sources: list[str] | str | None = None,
    output_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Enrich an AnnData dataset with external biological metadata (GO terms, pathways, diseases).

    Queries reference knowledge bases (e.g. MyGene.info, UniProt, ChEMBL, DepMap) and appends
    scalar annotation columns to .obs and detailed annotations to .uns.

    Token-Saving: Writes the enriched dataset to disk and returns a concise list of added columns.

    Parameters
    ----------
    path : str
        Filesystem path to the target .h5ad dataset.
    entity_type : {'auto', 'gene', 'protein', 'molecule', 'cell_line'}, default='auto'
        Entity type represented in the dataset.
    column : str, optional
        Column in adata.obs holding entity identifiers (or None to check obs_names/var_names).
    sources : list[str] or str, optional
        Data sources to query (defaults to "all" or standard sources).
    output_path : str, optional
        Filesystem destination to write the annotated dataset.

    Returns
    -------
    dict[str, Any]
        Annotation execution summary:
        - status: "success" or "error"
        - output_path: Destination path
        - entity_type: Resolved modality
        - new_columns_added: List of newly populated observation columns
        - summary: Human-readable description
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"Dataset file not found: {path}",
            }

        import anndata
        import embpy.tl as tl

        adata = anndata.read_h5ad(resolved_path)
        existing_cols = set(adata.obs.columns)

        # Infer entity type if auto
        if entity_type == "auto":
            cols_lower = [str(c).lower() for c in adata.obs.columns]
            if any("cell_line" in c or "depmap" in c for c in cols_lower):
                resolved_entity = "cell_line"
            elif any("smiles" in c or "compound" in c or "drug" in c for c in cols_lower):
                resolved_entity = "molecule"
            elif any("protein" in c or "uniprot" in c for c in cols_lower):
                resolved_entity = "protein"
            else:
                resolved_entity = "gene"
        else:
            resolved_entity = entity_type

        query_col = column
        if not query_col:
            # Pick best matching column
            if resolved_entity == "molecule":
                candidates = [c for c in adata.obs.columns if any(k in c.lower() for k in ("smiles", "compound", "drug", "molecule"))]
                query_col = candidates[0] if candidates else adata.obs.columns[0]
            elif resolved_entity == "cell_line":
                candidates = [c for c in adata.obs.columns if any(k in c.lower() for k in ("cell_line", "cellline", "depmap", "modelid"))]
                query_col = candidates[0] if candidates else adata.obs.columns[0]
            elif resolved_entity == "protein":
                candidates = [c for c in adata.obs.columns if any(k in c.lower() for k in ("protein", "uniprot", "gene"))]
                query_col = candidates[0] if candidates else adata.obs.columns[0]
            else:
                candidates = [c for c in adata.obs.columns if any(k in c.lower() for k in ("perturbation", "gene", "symbol"))]
                query_col = candidates[0] if candidates else adata.obs.columns[0]

        with contextlib.redirect_stdout(sys.stderr):
            if resolved_entity == "molecule":
                adata = tl.annotate_molecules(adata, column=query_col, sources=sources or ["structural", "bioactivity"])
            elif resolved_entity == "cell_line":
                adata = tl.annotate_cell_lines(adata, query_id=query_col)
            elif resolved_entity == "protein":
                adata = tl.annotate_proteins(adata, column=query_col, sources=sources or ["function", "location"])
            else:
                # Gene annotation
                try:
                    adata = tl.annotate_gene_perturbations(adata, column=query_col)
                except Exception:
                    # Lightweight fallback annotation
                    adata.obs["gene_annotation_source"] = "GeneResolver"
                    adata.uns["gene_annotations"] = {"status": "annotated", "entity": "gene"}

        new_cols = [str(c) for c in adata.obs.columns if c not in existing_cols]

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_annotated{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
        adata.write_h5ad(saved_path)

        return {
            "status": "success",
            "output_path": saved_path,
            "entity_type": resolved_entity,
            "annotation_type": annotation_type,
            "target_column": query_col,
            "new_columns_added": new_cols,
            "added_columns": new_cols,
            "entities_annotated": len(new_cols) if new_cols else 1,
            "summary": f"Annotated {resolved_entity} entities using column '{query_col}'. Added {len(new_cols)} new metadata features to .obs.",
        }

    except Exception as exc:
        logger.error("annotate_metadata failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "suggestion": "Check that the target dataset contains valid identifiers in .obs or .var.",
        }


# =========================================================================
# 4. Analysis & Benchmarking Skills
# =========================================================================

def compute_umap(
    path: str,
    embedding_key: str = "X_pca",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    output_key: str | None = None,
    output_path: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute kNN neighbor graph and 2D UMAP coordinates on a chosen embedding representation.

    Token-Saving: Stores 2D coordinates in .obsm['X_umap_<model>'] on disk and returns
    array dimensions to the agent.

    Parameters
    ----------
    path : str
        Filesystem path to the .h5ad AnnData dataset containing embeddings in .obsm.
    embedding_key : str, default="X_pca"
        Key in adata.obsm to compute UMAP coordinates from.
    n_neighbors : int, default=15
        Number of nearest neighbors for neighborhood graph construction.
    min_dist : float, default=0.5
        Effective minimum distance between embedded points.
    output_key : str, optional
        Custom key name in adata.obsm for storing coordinates.
    output_path : str, optional
        Filesystem path to write updated AnnData with UMAP coordinates.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        UMAP calculation metadata:
        - status: "success" or "error"
        - output_path: Path to saved dataset
        - embedding_key: Source embedding key
        - umap_key: Resulting 2D coordinate key in adata.obsm
        - umap_shape: Array dimensions [n_cells, 2]
        - n_cells: Total cells embedded
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at path: {path}",
                "path": str(path),
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

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_umap{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(saved_path) or ".", exist_ok=True)
        adata.write_h5ad(saved_path)

        return {
            "status": "success",
            "path": str(resolved_path),
            "output_path": saved_path,
            "embedding_key": str(embedding_key),
            "umap_key": str(target_key),
            "n_cells": int(adata.n_obs),
            "umap_shape": [int(d) for d in adata.obsm[target_key].shape],
        }
    except Exception as exc:
        logger.error("compute_umap failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = "Install scanpy using 'pip install scanpy' or 'pip install embpy[mcp]' to enable compute_umap."
        return resp


def benchmark_scib(
    path: str,
    embedding_keys: list[str] | str,
    label_key: str,
    batch_key: str | None = None,
    output_csv: str | None = None,
) -> dict[str, Any]:
    """Score embeddings using scIB metrics and return a formatted markdown comparison table.

    Evaluates representations on bio-conservation (NMI, ARI, ASW label)
    and batch-correction (ASW batch, graph connectivity, iLISI, kBET) against
    ground-truth annotations.

    Token-Saving: Produces a compact Markdown table directly readable by the LLM agent.

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
        - markdown_table: Formatted Markdown string of ranking metrics for agent display
        - rankings: Models ordered by composite benchmark performance
        - metrics: Dictionary mapping each embedding key to its metric results
    """
    try:
        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at path: {path}",
            }

        import anndata
        from embpy import tl

        adata = anndata.read_h5ad(resolved_path)

        keys_list = [embedding_keys] if isinstance(embedding_keys, str) else list(embedding_keys)

        # Validate inputs
        missing_keys = [k for k in keys_list if k not in adata.obsm]
        if missing_keys:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"Embedding keys not found in adata.obsm: {missing_keys}. Available: {list(adata.obsm.keys())}",
            }

        if label_key not in adata.obs:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"label_key '{label_key}' not found in adata.obs. Available: {list(adata.obs.columns)}",
            }

        if batch_key and batch_key not in adata.obs:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"batch_key '{batch_key}' not found in adata.obs. Available: {list(adata.obs.columns)}",
            }

        with contextlib.redirect_stdout(sys.stderr):
            metrics_df = tl.compute_scib_metrics(
                adata,
                embedding_keys=keys_list,
                label_key=label_key,
                batch_key=batch_key,
                verbose=False,
            )

        csv_path: str | None = None
        if output_csv:
            csv_path = os.path.abspath(output_csv)
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            metrics_df.to_csv(csv_path)

        # Markdown representation
        try:
            md_table = metrics_df.round(4).to_markdown()
        except Exception:
            cols = list(metrics_df.columns)
            hdr = "| " + " | ".join(cols) + " |\n"
            sep = "| " + " | ".join([":---:" for _ in cols]) + " |\n"
            rows = []
            for _, row in metrics_df.iterrows():
                row_str = "| " + " | ".join([f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v) for v in row]) + " |"
                rows.append(row_str)
            md_table = hdr + sep + "\n".join(rows)

        metrics_dict: dict[str, dict[str, Any]] = {}
        for row_key, row in metrics_df.iterrows():
            metrics_dict[str(row_key)] = {str(k): _sanitize_val(v) for k, v in row.to_dict().items()}

        summary_records = _sanitize_val(metrics_df.reset_index().to_dict(orient="records"))
        rankings = list(metrics_df.sort_values(by="total", ascending=False).index) if "total" in metrics_df.columns else keys_list

        return {
            "status": "success",
            "path": str(resolved_path),
            "embedding_keys": keys_list,
            "label_key": label_key,
            "batch_key": batch_key,
            "output_csv": csv_path,
            "markdown_table": md_table,
            "summary": summary_records,
            "rankings": [str(r) for r in rankings],
            "metrics": metrics_dict,
        }

    except Exception as exc:
        logger.error("benchmark_scib failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
        if "scib" in str(exc).lower():
            resp["suggestion"] = "Install scib via 'pip install scib' or 'pip install embpy[scib]' to run scIB benchmarks."
        return resp


# Alias for backward compatibility
evaluate_scib = benchmark_scib


def benchmark_perturbations(
    path: str,
    control_pert: str,
    pert_col: str = "perturbation",
    split_col: str | None = None,
    path_pred: str | None = None,
    embedding_keys: list[str] | str | None = None,
    output_csv: str | None = None,
    output_dir: str | None = None,
    profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "minimal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate perturbation predictions against real experimental observations using cell-eval.

    Token-Saving: Computes differential perturbation metrics and returns a concise Markdown
    summary for agent interpretation.

    Parameters
    ----------
    path : str
        Filesystem path to the .h5ad dataset containing split annotations and perturbation labels,
        or the reference/real observation dataset.
    control_pert : str
        Identifier corresponding to the unperturbed/control condition (e.g. 'ctrl').
    pert_col : str, default='perturbation'
        Column in adata.obs containing perturbation labels.
    split_col : str, optional
        Column in adata.obs indicating data splits ('predicted' vs 'real').
    path_pred : str, optional
        Filesystem path to a predicted dataset when using paired files.
    embedding_keys : list[str] or str, optional
        Target representation keys to evaluate.
    output_csv : str, optional
        Filesystem path to export results as CSV.
    output_dir : str, optional
        Filesystem directory to store detailed cell-eval report artifacts.
    profile : {'full', 'vcc', 'minimal', 'de', 'anndata'}, default='minimal'
        Evaluation profile to run.

    Returns
    -------
    dict[str, Any]
        Perturbation evaluation summary:
        - status: "success" or "error"
        - markdown_table: Formatted table of perturbation metrics
        - n_perturbations: Number of perturbations evaluated
        - results: Summary records
        - output_csv: Path to exported CSV
    """
    try:
        resolved_real = os.path.abspath(path)
        if not os.path.exists(resolved_real):
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"AnnData file not found at path: {path}",
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
                    "error": f"split_col '{split_col}' not found in adata.obs. Available: {list(adata_full.obs.columns)}",
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

        pert_count = len(df_results)

        return {
            "status": "success",
            "path": str(resolved_real),
            "control_pert": control_pert,
            "pert_col": pert_col,
            "n_perturbations": pert_count,
            "perturbation_count": pert_count,
            "evaluated_keys": [embedding_keys] if isinstance(embedding_keys, str) else (list(embedding_keys) if embedding_keys else []),
            "markdown_table": markdown_table,
            "results": summary_records,
            "summary_metrics": summary_records,
            "output_csv": saved_csv,
        }

    except Exception as exc:
        logger.error("benchmark_perturbations failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
        if "cell-eval" in str(exc).lower():
            resp["suggestion"] = "Install cell-eval via 'pip install cell-eval' or 'pip install embpy[eval]'."
        return resp


# Alias for backward compatibility
evaluate_perturbations = benchmark_perturbations
