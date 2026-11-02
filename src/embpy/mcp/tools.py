"""Universal bioinformatics skills for the embpy Model Context Protocol (MCP) server.

Exposes the complete universal embedding pipeline (Preprocessing, Embedding, Annotation,
Analysis, and Benchmarking) to autonomous LLM agents with strict token-saving guardrails:
1. inspect_data: Profile dataset dimensions, categorical columns, candidate label/batch slots, and expression nature.
2. resolve_identifiers: Gene symbols <-> Ensembl ids in the requested species; compound names/SMILES -> canonical SMILES.
3. preprocess_counts: Standard QC filtering, library size normalization, log1p transformation, and HVG selection.
4. list_model_registry: The models each embedding modality can run, plus the compute device.
5. compute_embeddings: BioEmbedder for cells, genes, DNA, proteins, molecules, text and morphology, reading inputs from a chosen column.
6. annotate_metadata: GO terms, pathways, domains, diseases, interactions; success counted from the retrieved records.
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
import json
import logging
import math
import os
import re
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


def _sanitize_for_h5ad(val: Any) -> Any:
    """Recursively sanitize metadata containers so H5AD/h5py can serialize them safely."""
    if isinstance(val, dict):
        return {str(k): _sanitize_for_h5ad(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        if any(isinstance(x, (dict, list, tuple)) for x in val):
            return json.dumps(val)
        return list(val)
    return val


def _save_anndata_safely(adata: Any, path: str) -> None:
    """Safely write an AnnData object to an .h5ad file, ensuring directories and HDF5 data types are valid."""
    target_dir = os.path.dirname(os.path.abspath(path))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    if getattr(adata, "uns", None):
        for k in list(adata.uns.keys()):
            adata.uns[k] = _sanitize_for_h5ad(adata.uns[k])
    adata.write_h5ad(path)


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


_GENE_ID_TYPES: dict[str, str] = {
    "ensembl": "ensembl",
    "ensembl_id": "ensembl",
    "ensg": "ensembl",
    "symbol": "symbol",
    "hgnc": "symbol",
    "gene_symbol": "symbol",
}
_MOLECULE_SOURCE_TYPES: dict[str, str] = {"name": "name", "smiles": "smiles"}
_MOLECULE_TARGET_TYPES: dict[str, str] = {"smiles": "smiles"}
_ENSEMBL_ID_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$", re.IGNORECASE)


def _parses_as_smiles(s: str, *, require_smiles_syntax: bool = True) -> bool:
    """True if RDKit parses ``s``.

    With ``require_smiles_syntax`` (used to tell molecules from genes) the string
    must also contain SMILES syntax: gene symbols such as 'CS' or 'NO' parse too.
    """
    from embpy.resources.gene.resolver import _looks_like_smiles

    if require_smiles_syntax and not _looks_like_smiles(s):
        return False
    try:
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(s) is not None
    except ImportError:
        return require_smiles_syntax


def _species_from_ensembl_id(eid: str) -> str | None:
    e = eid.strip().upper()
    if e.startswith("ENSMUSG"):
        return "mouse"
    if re.match(r"^ENSG\d", e):
        return "human"
    return None


def _species_from_symbol_case(sym: str) -> str | None:
    """HGNC symbols are upper case (TP53), MGI symbols title case (Trp53)."""
    letters = [c for c in sym if c.isalpha()]
    if not letters:
        return None
    if all(c.isupper() for c in letters):
        return "human"
    if sym[0].isupper() and any(c.islower() for c in letters[1:]):
        return "mouse"
    return None


def _resolve_gene_group(
    items: list[str],
    *,
    organism: str | None,
    source: str,
    target: str,
) -> dict[str, Any]:
    """Resolve gene symbols / Ensembl ids, never letting capitalization pick the species.

    ``organism`` is ``"human"``/``"mouse"`` (explicit: identifiers from another
    species are an error), ``None`` (human, except that an Ensembl id's prefix
    names its species) or ``"auto"`` (opt-in: species from Ensembl prefixes and
    from symbol capitalization, reported back per identifier).
    """
    from embpy.resources.gene.resolver import GeneResolver

    mapping: dict[str, str] = {}
    invalid: dict[str, str] = {}
    by_species: dict[tuple[str, str], list[str]] = {}
    species_of: dict[str, str] = {}
    conflicts: dict[str, str] = {}

    for item in items:
        kind = "ensembl" if _ENSEMBL_ID_RE.match(item) else "symbol"
        if source != "auto" and kind != source:
            invalid[item] = f"does not look like a {source} identifier"
            continue
        direction = target if target != "auto" else ("symbol" if kind == "ensembl" else "ensembl")
        if direction == kind:
            invalid[item] = f"is already a {kind} identifier (target_id_type={target!r})"
            continue

        id_species = _species_from_ensembl_id(item) if kind == "ensembl" else None
        if organism in ("human", "mouse"):
            if kind == "ensembl" and id_species not in (None, organism):
                conflicts[item] = id_species
                continue
            species = organism
        elif organism == "auto":
            species = id_species if kind == "ensembl" else _species_from_symbol_case(item)
            if species is None:
                invalid[item] = "species could not be determined (pass organism='human' or 'mouse')"
                continue
        else:
            species = id_species or "human"
        if kind == "ensembl" and id_species is None:
            invalid[item] = "is an Ensembl id of a species other than human or mouse"
            continue
        species_of[item] = species
        by_species.setdefault((species, direction), []).append(item)

    if conflicts:
        return {"conflicts": conflicts}

    warnings: list[str] = []
    actions: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    for (species, direction), batch in by_species.items():
        resolver = GeneResolver(species=species)
        if direction == "ensembl":
            res, diag = resolver.symbols_to_ensembl_batch(batch, organism=species, return_diagnostics=True)
        else:
            res, diag = resolver.ensembl_to_symbols_batch(batch, organism=species, return_diagnostics=True)
        if diag.get("warning") and diag.get("unresolved_count"):
            warnings.append(str(diag["warning"]))
        if diag.get("action_required"):
            actions.append(str(diag["action_required"]))
        ambiguous.update(diag.get("ambiguous") or {})
        for item in batch:
            value = res.get(item)
            if not value or not str(value).strip():
                continue
            value = str(value)
            if direction == "ensembl":
                from embpy.resources.gene.resolver import ensembl_id_matches_species

                if not ensembl_id_matches_species(value, species):
                    invalid[item] = f"resolved to {value}, which is not a {species} gene"
                    continue
                approved = getattr(resolver, "_ensembl_cache", {}).get((species, value))
                if approved and approved != item:
                    if approved.upper() == item.upper():
                        notes[item] = f"matched {species} symbol {approved} (case-insensitive)"
                    else:
                        notes[item] = f"alias of {species} symbol {approved}"
            mapping[item] = value

    return {
        "mapping": mapping,
        "invalid": invalid,
        "ambiguous": ambiguous,
        "species_of": species_of,
        "directions": sorted({direction for _, direction in by_species}),
        "notes": notes,
        "warnings": warnings,
        "actions": actions,
    }


def _resolve_molecule_group(items: list[str], *, source: str) -> dict[str, Any]:
    """Canonicalize SMILES and resolve compound names to canonical SMILES."""
    from embpy.resources.molecule.resolver import DrugResolver

    resolver = DrugResolver()
    mapping: dict[str, str] = {}
    invalid: dict[str, str] = {}
    errors: list[str] = []
    for item in items:
        is_smiles = source == "smiles" or (source == "auto" and _parses_as_smiles(item, require_smiles_syntax=False))
        try:
            value = resolver.canonicalize_smiles(item) if is_smiles else resolver.name_to_smiles(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item}: {type(exc).__name__}: {exc}")
            continue
        if value:
            mapping[item] = str(value)
        elif is_smiles:
            invalid[item] = "is not a valid SMILES string"
    return {"mapping": mapping, "invalid": invalid, "errors": errors}


def resolve_identifiers(
    identifiers: list[str] | str | None = None,
    genes: list[str] | None = None,
    path: str | None = None,
    entity_type: Literal["auto", "gene", "molecule"] = "auto",
    source_id_type: Literal["auto", "symbol", "ensembl", "name", "smiles"] = "auto",
    target_id_type: Literal["auto", "ensembl", "symbol", "smiles"] = "auto",
    target_type: Literal["auto", "ensembl", "symbol", "smiles"] | None = None,
    organism: Literal["human", "mouse", "auto"] | None = None,
    column: str | None = None,
    obs_column: str | None = None,
    var_column: str | None = None,
    update_h5ad_path: str | None = None,
    output_path: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Map entity identifiers: gene symbols <-> Ensembl gene ids, compound names/SMILES -> canonical SMILES.

    Harmonizes biological vocabularies to prevent mismatch errors in downstream models.
    Returns compact mapping metrics rather than dumping large dictionaries.

    Parameters
    ----------
    identifiers : list[str] or str, optional
        Entity identifiers to resolve (a comma-separated string is also accepted).
    genes : list[str], optional
        Backward compatibility alias for identifiers.
    path : str, optional
        Path to an AnnData (.h5ad) dataset whose var_names or obs/var column to resolve.
    entity_type : {'auto', 'gene', 'molecule'}, default='auto'
        'auto' treats strings that parse as SMILES as molecules and everything else
        as genes. Pass 'molecule' for compound names such as 'aspirin'.
    source_id_type : {'auto', 'symbol', 'ensembl', 'name', 'smiles'}, default='auto'
        Format of the inputs. Genes: 'symbol' or 'ensembl'. Molecules: 'name' or 'smiles'.
        Inputs that do not match an explicit source type are reported as invalid.
    target_id_type : {'auto', 'ensembl', 'symbol', 'smiles'}, default='auto'
        Genes: 'ensembl' or 'symbol' ('auto' converts each id to the other kind).
        Molecules: 'smiles' (canonical SMILES). Other targets are rejected.
    target_type : optional
        Alias for target_id_type.
    organism : {'human', 'mouse', 'auto'}, optional
        Species to resolve genes in.
        - 'human' / 'mouse': always used. An Ensembl id from another species is an error;
          a symbol is looked up in this species whatever its capitalization
          ('Egfr' with organism='human' resolves to human EGFR, and the response notes it).
        - omitted: human, except that Ensembl ids resolve in the species their prefix
          names (ENSMUSG... -> mouse). Symbol capitalization is never used.
        - 'auto': opt-in detection from Ensembl prefixes and symbol capitalization
          (TP53 -> human, Trp53 -> mouse). The species used is reported per identifier.
    column : str, optional
        Identifier column in adata.obs or adata.var.
    obs_column : str, optional
        Column in adata.obs containing identifiers (e.g. 'smiles').
    var_column : str, optional
        Column in adata.var containing identifiers (e.g. 'symbol').
    update_h5ad_path : str, optional
        If provided, writes the AnnData with the mapped ids added as '<column>_mapped'.
    output_path : str, optional
        Alias for update_h5ad_path.
    strict : bool, default=False
        If True, any identifier that is not mapped (including blank entries) makes the
        call fail and nothing is written.

    Returns
    -------
    dict[str, Any]
        - status: "success" (every entry mapped), "partial_success" (some mapped) or "error" (none mapped,
          invalid request, or strict mode with unmapped entries)
        - total_submitted / mapped_count / unmapped_count: counted per input entry (duplicates and blanks included)
        - blank_count: Blank or missing entries
        - target_id_type: Target format used (a dict per entity type when inputs mix genes and molecules)
        - organism_used: Number of gene identifiers resolved in each species
        - sample_mapping: First 5 mapped pairs
        - unmapped_sample: Up to 5 unmapped identifiers
        - invalid: Up to 10 identifiers rejected before lookup, with the reason
        - ambiguous: Aliases shared by several genes, with the candidate symbols
        - notes: Case-insensitive matches and aliases resolved to an approved symbol
    """
    try:
        rows: list[str] = []
        adata = None
        source_series = None
        source_target = "var"
        mapped_col_name = None
        out_path = update_h5ad_path or output_path

        if path:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"Dataset file not found: {path}",
                }
            import anndata
            import pandas as pd

            adata = anndata.read_h5ad(resolved_path)

            if obs_column:
                if obs_column not in adata.obs:
                    return {
                        "status": "error",
                        "error_type": "KeyError",
                        "error": f"obs_column '{obs_column}' not found in adata.obs. Available: {list(adata.obs.columns)}",
                    }
                source_series = adata.obs[obs_column]
                source_target = "obs"
                mapped_col_name = f"{obs_column}_mapped"
            elif var_column:
                if var_column not in adata.var:
                    return {
                        "status": "error",
                        "error_type": "KeyError",
                        "error": f"var_column '{var_column}' not found in adata.var. Available: {list(adata.var.columns)}",
                    }
                source_series = adata.var[var_column]
                source_target = "var"
                mapped_col_name = f"{var_column}_mapped"
            elif column:
                if column in adata.obs:
                    source_series = adata.obs[column]
                    source_target = "obs"
                    mapped_col_name = f"{column}_mapped"
                elif column in adata.var:
                    source_series = adata.var[column]
                    source_target = "var"
                    mapped_col_name = f"{column}_mapped"
                else:
                    return {
                        "status": "error",
                        "error_type": "KeyError",
                        "error": f"column '{column}' not found in adata.obs or adata.var.",
                    }
            else:
                if entity_type == "molecule" and "smiles" in adata.obs:
                    source_series = adata.obs["smiles"]
                    source_target = "obs"
                    mapped_col_name = "smiles_mapped"
                else:
                    source_series = pd.Series(list(adata.var_names), index=adata.var_names)
                    source_target = "var"
                    mapped_col_name = "gene_mapped_id"

            rows = [str(x).strip() if pd.notna(x) else "" for x in source_series]
        elif identifiers is not None:
            if isinstance(identifiers, str):
                rows = [s.strip() for s in identifiers.split(",")]
            else:
                rows = [str(x).strip() if x is not None else "" for x in identifiers]
        elif genes is not None:
            rows = [str(x).strip() if x is not None else "" for x in genes]
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Must provide 'identifiers' or 'path' to dataset.",
            }

        total_submitted = len(rows)
        if total_submitted == 0:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "No identifiers provided for resolution.",
            }

        org = organism.lower() if isinstance(organism, str) else None
        if org not in (None, "human", "mouse", "auto"):
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"organism must be 'human', 'mouse' or 'auto' (or omitted), got {organism!r}.",
            }

        requested_tgt = str(target_type if target_type is not None else target_id_type).lower()
        requested_src = str(source_id_type).lower()
        unique_items = list(dict.fromkeys(r for r in rows if r))

        if entity_type == "auto":
            groups = {"gene": [], "molecule": []}
            for item in unique_items:
                groups["molecule" if _parses_as_smiles(item) else "gene"].append(item)
            groups = {k: v for k, v in groups.items() if v}
        elif entity_type in ("gene", "molecule"):
            groups = {entity_type: unique_items}
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"entity_type must be 'auto', 'gene' or 'molecule', got {entity_type!r}.",
            }

        # Validate source and target formats for every entity type actually present.
        allowed = {
            "gene": (_GENE_ID_TYPES, _GENE_ID_TYPES),
            "molecule": (_MOLECULE_SOURCE_TYPES, _MOLECULE_TARGET_TYPES),
        }
        targets: dict[str, str] = {}
        sources: dict[str, str] = {}
        for ent in groups or {entity_type if entity_type != "auto" else "gene": []}:
            src_map, tgt_map = allowed[ent]
            if requested_tgt != "auto" and requested_tgt not in tgt_map:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"target_id_type {requested_tgt!r} is not supported for {ent} identifiers. "
                        f"Allowed: {sorted(set(tgt_map.values()))} or 'auto'."
                    ),
                }
            if requested_src != "auto" and requested_src not in src_map:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"source_id_type {requested_src!r} is not supported for {ent} identifiers. "
                        f"Allowed: {sorted(set(src_map.values()))} or 'auto'."
                    ),
                }
            targets[ent] = tgt_map[requested_tgt] if requested_tgt != "auto" else ("auto" if ent == "gene" else "smiles")
            sources[ent] = src_map[requested_src] if requested_src != "auto" else "auto"

        mapping: dict[str, str] = {}
        invalid: dict[str, str] = {}
        ambiguous: dict[str, list[str]] = {}
        notes: dict[str, str] = {}
        warnings: list[str] = []
        actions: list[str] = []
        organism_used: dict[str, int] = {}

        with contextlib.redirect_stdout(sys.stderr):
            if "gene" in groups:
                gene_res = _resolve_gene_group(
                    groups["gene"], organism=org, source=sources["gene"], target=targets["gene"]
                )
                if gene_res.get("conflicts"):
                    conflicts = gene_res["conflicts"]
                    return {
                        "status": "error",
                        "error_type": "SpeciesMismatchError",
                        "error": (
                            f"organism={organism!r} was requested, but {len(conflicts)} Ensembl id(s) belong to "
                            f"another species: {dict(list(conflicts.items())[:5])}. Pass the matching organism, "
                            "omit it to let Ensembl prefixes decide, or use organism='auto'."
                        ),
                        "conflicting_identifiers": list(conflicts)[:10],
                    }
                mapping.update(gene_res["mapping"])
                invalid.update(gene_res["invalid"])
                ambiguous.update(gene_res["ambiguous"])
                notes.update(gene_res["notes"])
                warnings += gene_res["warnings"]
                actions += gene_res["actions"]
                for species in gene_res["species_of"].values():
                    organism_used[species] = organism_used.get(species, 0) + 1
                if targets["gene"] == "auto" and len(gene_res["directions"]) == 1:
                    targets["gene"] = gene_res["directions"][0]
                elif targets["gene"] == "auto" and gene_res["directions"]:
                    targets["gene"] = "per_identifier (ensembl <-> symbol)"
            if "molecule" in groups:
                mol_res = _resolve_molecule_group(groups["molecule"], source=sources["molecule"])
                mapping.update(mol_res["mapping"])
                invalid.update(mol_res["invalid"])
                if mol_res["errors"]:
                    warnings.append(f"Molecule resolution failed for {len(mol_res['errors'])} identifier(s): {mol_res['errors'][:3]}")

        mapped_count = sum(1 for r in rows if r and r in mapping)
        blank_count = sum(1 for r in rows if not r)
        unmapped_count = total_submitted - mapped_count
        rate = round((mapped_count / total_submitted) * 100.0, 2)
        unmapped_unique = [u for u in unique_items if u not in mapping]

        if mapped_count == total_submitted:
            status = "success"
        elif mapped_count > 0:
            status = "partial_success"
        else:
            status = "error"

        result_payload: dict[str, Any] = {
            "status": status,
            "entity_type": next(iter(groups)) if len(groups) == 1 else ("mixed" if groups else entity_type),
            "target_id_type": next(iter(targets.values())) if len(targets) == 1 else targets,
            "organism_used": organism_used,
            "total_submitted": total_submitted,
            "total_genes": total_submitted,
            "mapped_count": mapped_count,
            "unmapped_count": unmapped_count,
            "unresolved_count": unmapped_count,
            "blank_count": blank_count,
            "mapping_success_rate": rate,
            "mapping_rate": rate,
            "sample_mapping": {k: mapping[k] for k in list(mapping)[:5]},
            "mapping_sample": {k: mapping[k] for k in list(mapping)[:5]},
            "unmapped_sample": unmapped_unique[:5],
            "output_path": None,
        }
        if invalid:
            result_payload["invalid"] = dict(list(invalid.items())[:10])
        if ambiguous:
            result_payload["ambiguous"] = dict(list(ambiguous.items())[:10])
        if notes:
            result_payload["notes"] = dict(list(notes.items())[:10])
        if warnings:
            result_payload["warning"] = " ".join(dict.fromkeys(warnings))
        if actions and unmapped_count:
            result_payload["action_required"] = actions[0]

        if status == "error":
            result_payload["error_type"] = "UnresolvedIdentifiersError"
            result_payload["error"] = "None of the submitted identifiers could be mapped."
        elif strict and unmapped_count:
            result_payload["status"] = "error"
            result_payload["error_type"] = "UnresolvedIdentifiersError"
            result_payload["error"] = (
                f"strict=True and {unmapped_count} of {total_submitted} entries were not mapped "
                f"({blank_count} blank). Nothing was written."
            )

        if result_payload["status"] != "error" and out_path and adata is not None and source_series is not None:
            import pandas as pd

            saved_path = os.path.abspath(out_path)
            # Map back over the EXACT source series element-by-element
            # Preserves original row order, duplicate names, and missing/NaN values
            aligned_values = [
                mapping.get(str(x).strip()) if pd.notna(x) else None
                for x in source_series
            ]
            if source_target == "obs":
                adata.obs[mapped_col_name] = aligned_values
            else:
                adata.var[mapped_col_name] = aligned_values
            _save_anndata_safely(adata, saved_path)
            result_payload["output_path"] = saved_path

        return result_payload

    except Exception as exc:
        logger.error("resolve_identifiers failed: %s", exc, exc_info=True)
        resp: dict[str, Any] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        is_conn = (
            isinstance(exc, (requests.ConnectionError, requests.Timeout, GeneResolutionError))
            or "timeout" in type(exc).__name__.lower()
            or "connection" in type(exc).__name__.lower()
        )
        if is_conn:
            failed = getattr(exc, "failed_genes", None) or [r for r in locals().get("rows", []) if r]
            resp.update(
                {
                    "mapping_rate": 0.0,
                    "unresolved_count": len(failed),
                    "warning": f"Ensembl REST API timed out on {len(failed)} genes. No identifiers were mapped.",
                    "action_required": "Install embpy[genome] for offline resolution.",
                }
            )
        return resp


# Alias for backward compatibility
map_gene_identifiers = resolve_identifiers


def _small_dataset_thresholds(
    n_obs: int, n_vars: int, min_genes: int, min_cells: int, n_top_genes: int
) -> tuple[int, int, int, dict[str, str]]:
    """Scale QC and HVG thresholds down for small datasets so filtering cannot remove every cell.

    Returns the effective (min_genes, min_cells, n_top_genes) and a description of each change.
    """
    eff_genes = min(min_genes, max(1, n_vars // 3)) if n_vars < 500 else min_genes
    eff_cells = min(min_cells, max(1, n_obs // 4)) if n_obs < 100 else min_cells
    eff_hvg = min(n_top_genes, max(10, n_vars - 1)) if n_vars < 2000 else n_top_genes
    changes = {
        name: f"Adjusted from {old} to {new}"
        for name, old, new in (
            ("min_genes", min_genes, eff_genes),
            ("min_cells", min_cells, eff_cells),
            ("n_top_genes", n_top_genes, eff_hvg),
        )
        if old != new
    }
    return eff_genes, eff_cells, eff_hvg, changes


def preprocess_counts(
    path: str,
    output_path: str | None = None,
    preprocessing: Literal["auto", "standard", "none"] = "standard",
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    select_hvg: bool = True,
    log_transform: bool = True,
    scale: bool = False,
    auto_adjust: bool = False,
) -> dict[str, Any]:
    """Preprocess single-cell AnnData counts and save the preprocessed object to disk.

    Applies standard single-cell quality control, library size normalization (sc.pp.normalize_total),
    log1p transformation (sc.pp.log1p), highly variable gene (HVG) selection, and optional scaling.
    Preserves raw counts in .layers['counts'] and standardized expression in .layers['log_normalized'].

    Token-Saving: Does not return expression matrices to the agent; writes preprocessed
    dataset to disk and returns compact summary statistics.

    Parameters
    ----------
    path : str
        Path to the input raw .h5ad AnnData file.
    output_path : str, optional
        Filesystem destination to write the preprocessed .h5ad dataset.
    preprocessing : {'auto', 'standard', 'none'}, default='standard'
        Single-cell preprocessing strategy:
        - 'auto': Automatically assesses dataset scale; dynamically adjusts QC thresholds (min_genes, min_cells) and HVG targets if dataset is small to prevent cell starvation.
        - 'standard': Applies strict filtering cutoffs (min_genes, min_cells), normalize_total, log1p transformation, and selects n_top_genes HVGs.
        - 'none': Bypasses filtering and transformations; preserves raw counts in .layers['counts'] without alterations.
    min_genes : int, default=200
        Minimum genes expressed per cell. Strictly enforced unless auto_adjust=True or preprocessing='none'.
    min_cells : int, default=3
        Minimum cells expressing a gene. Strictly enforced unless auto_adjust=True or preprocessing='none'.
    target_sum : float, default=1e4
        Target total counts per cell for normalization.
    n_top_genes : int, default=2000
        Number of highly variable genes to select.
    select_hvg : bool, default=True
        Whether to compute and select highly variable genes.
    log_transform : bool, default=True
        Whether to log1p transform counts.
    scale : bool, default=False
        Whether to scale expression to unit variance (stored in .layers['scaled']).
    auto_adjust : bool, default=False
        If True, automatically lowers thresholds for small tutorial/test datasets.

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
        - layers: Available layers (e.g. ['counts', 'log_normalized', 'scaled'])
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

        if preprocessing == "auto":
            auto_adjust = True
        elif preprocessing == "none":
            if not output_path:
                base, ext = os.path.splitext(resolved_path)
                saved_path = f"{base}_preprocessed{ext}"
            else:
                saved_path = os.path.abspath(output_path)

            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()
            _save_anndata_safely(adata, saved_path)
            return {
                "status": "success",
                "input_path": str(resolved_path),
                "output_path": saved_path,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_hvg": 0,
                "hvg_count": 0,
                "layers": [str(k) for k in adata.layers.keys()],
                "summary": (
                    f"Preprocessing bypassed (mode='none'). Saved {adata.n_obs} cells and "
                    f"{adata.n_vars} genes with raw counts preserved in .layers['counts']."
                ),
            }

        adjustments_applied: dict[str, Any] = {}
        if auto_adjust:
            effective_min_genes, effective_min_cells, effective_n_hvg, adjustments_applied = _small_dataset_thresholds(
                adata.n_obs, adata.n_vars, min_genes, min_cells, n_top_genes
            )
        else:
            effective_min_genes = min_genes
            effective_min_cells = min_cells
            effective_n_hvg = min(n_top_genes, adata.n_vars)

        with contextlib.redirect_stdout(sys.stderr):
            # Store raw counts in layers
            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()

            # QC Filtering
            if effective_min_genes > 0 and adata.n_obs > 0:
                sc.pp.filter_cells(adata, min_genes=effective_min_genes)
            if effective_min_cells > 0 and adata.n_vars > 0:
                sc.pp.filter_genes(adata, min_cells=effective_min_cells)

            # Normalization & Log1p
            if target_sum and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            if log_transform and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.log1p(adata)

            # Store log-normalized representation in layers
            if adata.n_obs > 0 and adata.n_vars > 0:
                adata.layers["log_normalized"] = adata.X.copy()

            # Scale if requested
            if scale and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.scale(adata, max_value=10)
                adata.layers["scaled"] = adata.X.copy()

            # HVG Selection
            if select_hvg and adata.n_vars > 2 and adata.n_obs > 0:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=min(effective_n_hvg, adata.n_vars),
                    flavor="seurat" if log_transform else "seurat_v3",
                )

            # Restore .X to raw counts to honor embpy output contract
            if "counts" in adata.layers and adata.n_obs > 0:
                adata.X = adata.layers["counts"].copy()

        # Determine output destination
        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_preprocessed{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        _save_anndata_safely(adata, saved_path)

        n_hvg_count = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var else 0

        res_payload = {
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
        if adjustments_applied:
            res_payload["adjustments_applied"] = adjustments_applied
        return res_payload

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

EmbeddingModality = Literal[
    "gene",
    "protein",
    "molecule",
    "dna",
    "single_cell",
    "morphology",
    "text",
    "perturbation",
]

# Friendly names for registry keys. Every alias names the same model as its
# target; a name the registry does not have (e.g. 'bge_large') is not aliased
# to a different model.
MODEL_ALIASES: dict[str, str] = {
    # Single-cell foundation models
    "geneformer": "geneformer_v2_12L",
    "geneformer_v2": "geneformer_v2_12L",
    # Protein foundation models
    "esm2_t6_8m": "esm2_8M",
    "esm2_t12_35m": "esm2_35M",
    "esm2_t30_150m": "esm2_150M",
    "esm2_t33_650m_ur50d": "esm2_650M",
    "esm2_t33_650m": "esm2_650M",
    "esm2_650m_ur50d": "esm2_650M",
    "esm1v": "esm1v_1",
    "prott5": "prot_t5_xl",
    "prot_t5": "prot_t5_xl",
    # Small molecule foundation models & chemical fingerprints
    "chemberta": "chemberta2MTR",
    "rdkit_morgan": "morgan_fp",
    # DNA / genomic foundation models
    "enformer": "enformer_human_rough",
    "borzoi": "borzoi_v0",
    "nt": "nt_500m_human_ref",
    "evo": "evo1_8k",
    # Cellular morphology models
    "subcell": "subcell_mae_rybg",
    # Text models
    "minilm": "minilm_l6_v2",
    "llama": "llama3.2_1b",
}

# How each compute_embeddings modality runs: the BioEmbedder.embed entity_type
# it routes to, the registry model types that entity type can execute, whether
# gene-level static lookup tables (genept, gene2vec, ...) apply, and the
# input each one expects. 'single_cell' runs BioEmbedder.embed_cells instead.
_EMBED_ROUTES: dict[str, dict[str, Any]] = {
    "gene": {
        "backend": "gene",
        "model_types": ("dna", "protein", "text"),
        "static_tables": True,
        "input": "gene symbols or Ensembl ids (var_names by default, or obs_column/var_column)",
    },
    "perturbation": {
        "backend": "gene",
        "model_types": ("dna", "protein", "text"),
        "static_tables": True,
        "is_perturbation": True,
        "input": "per-cell genetic perturbation labels (gene symbols) in obs_column; one vector per cell in .obsm",
    },
    "dna": {
        "backend": "sequence",
        "model_types": ("dna",),
        "input": "raw nucleotide sequences in obs_column or var_column",
    },
    "protein": {
        "backend": "protein",
        "model_types": ("protein",),
        "input": "protein/gene identifiers or amino-acid sequences (set id_type)",
    },
    "molecule": {
        "backend": "molecule",
        "model_types": ("molecule",),
        "input": "SMILES (default) or compound names with id_type='name', usually obs_column='smiles'",
    },
    "text": {
        "backend": "text",
        "model_types": ("text",),
        "input": "free text in obs_column or var_column",
    },
    "morphology": {
        "backend": "perturbation",
        "model_types": ("morphology",),
        "input": "perturbation labels whose microscopy images (HPA/JUMP) are embedded, in obs_column",
    },
    "single_cell": {
        "backend": "embed_cells",
        "model_types": ("single_cell",),
        "input": "expression matrix of the AnnData (cells x genes)",
    },
}

# Curated starting points per modality. Every entry is a canonical registry key
# that _validate_embedding_model accepts for that modality (checked in tests).
_RECOMMENDED_MODELS: dict[str, list[str]] = {
    "single_cell": ["pca", "scvi", "scgpt", "geneformer_v2_12L", "state"],
    "gene": ["genept", "gene2vec", "esm2_35M", "nt_v2_100m", "minilm_l6_v2"],
    "perturbation": ["genept", "gene2vec", "esm2_35M", "minilm_l6_v2"],
    "dna": ["hyenadna_tiny_1k", "nt_v2_100m", "caduceus_ph_131k", "enformer_human_rough", "borzoi_v0"],
    "protein": ["esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M", "esm1v_1", "prot_t5_xl"],
    "molecule": ["morgan_fp", "rdkit_fp", "maccs_fp", "chemberta2MTR", "molformer_base"],
    "text": ["minilm_l6_v2", "bert_base_uncased", "llama3.2_1b"],
    "morphology": ["subcell_mae_rybg", "subcell_vit_rybg"],
}

_SUPPORTED_MODALITIES: list[str] = [
    "gene",
    "protein",
    "molecule",
    "dna",
    "single_cell",
    "morphology",
    "text",
    "perturbation",
]


def _canonical_model_name(name: str) -> str:
    raw = str(name).strip()
    return MODEL_ALIASES.get(raw.lower(), raw)


def _model_kind(model: str) -> str | None:
    """Return the modality family a canonical model key belongs to, or None if unknown."""
    from embpy.embedder_registry.flat import MODEL_REGISTRY
    from embpy.models.singlecell_models import list_singlecell_models
    from embpy.pp.static_embeddings import static_embedding_keys

    if model in list_singlecell_models():
        return "single_cell"
    if model in static_embedding_keys("gene"):
        return "static_gene_table"
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        return None
    return getattr(entry[0], "model_type", None) or "unavailable"


def _modalities_accepting(kind: str | None) -> list[str]:
    if kind is None:
        return []
    return [
        m
        for m, route in _EMBED_ROUTES.items()
        if kind in route["model_types"] or (kind == "static_gene_table" and route.get("static_tables"))
    ]


def _validate_embedding_model(modality: str, model: str) -> str | None:
    """Return an error message if ``model`` cannot run under ``modality``, else None."""
    kind = _model_kind(model)
    route = _EMBED_ROUTES[modality]
    if kind is None:
        return (
            f"Unknown model {model!r}. Call list_model_registry(entity_type={modality!r}) "
            "for the models this modality can run."
        )
    if kind == "unavailable":
        return f"Model {model!r} is registered but its wrapper is unavailable; install its optional dependencies."
    if kind in route["model_types"] or (kind == "static_gene_table" and route.get("static_tables")):
        return None
    accepted = _modalities_accepting(kind)
    return (
        f"Model {model!r} is a {kind} model and cannot run under entity_type={modality!r} "
        f"(which expects {route['input']}). Use entity_type in {accepted} for this model, "
        f"or one of {_RECOMMENDED_MODELS[modality]} for {modality!r}."
    )


def list_model_registry(
    entity_type: EmbeddingModality | None = None,
) -> dict[str, Any]:
    """Report the models compute_embeddings can run for each modality, and the compute hardware.

    Each modality has one execution path, and only models that path can execute are listed:
    - 'single_cell': cell embeddings from the expression matrix (PCA, scVI, scGPT, Geneformer, STATE)
    - 'gene': gene-level vectors from DNA, protein or text models, or static tables (genept, gene2vec)
    - 'perturbation': the same gene-level models applied to per-cell genetic perturbation labels
    - 'dna': raw nucleotide sequences (HyenaDNA, Nucleotide Transformer, Caduceus, Enformer, Borzoi)
    - 'protein': protein language models (ESM-2, ESM-1v, ProtT5, Boltz-2)
    - 'molecule': small molecules (Morgan/RDKit fingerprints, ChemBERTa, MoLFormer)
    - 'text': text embedding models (MiniLM, BERT, LLaMA)
    - 'morphology': microscopy images of perturbations (SubCell)

    Parameters
    ----------
    entity_type : {'gene', 'protein', 'molecule', 'dna', 'single_cell', 'morphology', 'text', 'perturbation'}, optional
        Filter model roster by biological modality.

    Returns
    -------
    dict[str, Any]
        - status: "success"
        - total_models_available: Total models across the registry and single-cell/static tables
        - supported_entities: The 8 modalities accepted by compute_embeddings
        - compute_device: Active compute hardware (cuda / mps / cpu)
        - recommended_models: Curated canonical model keys per modality (filtered when entity_type is set)
        - execution_paths: Input each modality expects and the backend that runs it
        - available_models: Every model each modality can run (filtered when entity_type is set)
        - runnable_singlecell_models: Single-cell models whose dependencies are installed locally
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

        from embpy.embedder_registry.flat import MODEL_REGISTRY
        from embpy.models.singlecell_models import list_singlecell_models
        from embpy.pp.static_embeddings import static_embedding_keys

        sc_models = sorted(list_singlecell_models())
        static_gene = sorted(static_embedding_keys("gene"))
        by_type: dict[str, list[str]] = {}
        for key, (wrapper, _) in MODEL_REGISTRY.items():
            mtype = getattr(wrapper, "model_type", None)
            if mtype:
                by_type.setdefault(mtype, []).append(key)

        available_models: dict[str, list[str]] = {}
        for modality, route in _EMBED_ROUTES.items():
            if modality == "single_cell":
                available_models[modality] = sc_models
                continue
            keys = [k for t in route["model_types"] for k in by_type.get(t, [])]
            if route.get("static_tables"):
                keys += static_gene
            available_models[modality] = sorted(set(keys))

        total_models = len(set(sc_models) | set(static_gene) | set(MODEL_REGISTRY))

        runnable_sc: list[str] = ["pca"]
        if importlib.util.find_spec("scvi") is not None:
            runnable_sc.extend(["scvi", "scanvi", "totalvi"])
        if importlib.util.find_spec("helical") is not None:
            runnable_sc.extend(["scgpt", "geneformer_v2_12L"])
        if importlib.util.find_spec("state") is not None:
            runnable_sc.append("state")

        execution_paths = {
            m: {"input": r["input"], "backend": r["backend"], "model_types": list(r["model_types"])}
            for m, r in _EMBED_ROUTES.items()
        }

        if entity_type is not None:
            norm_entity = str(entity_type).lower().strip()
            if norm_entity in ("cell", "singlecell"):
                norm_entity = "single_cell"
            if norm_entity not in _EMBED_ROUTES:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": f"entity_type must be one of {_SUPPORTED_MODALITIES}, got {entity_type!r}.",
                }
            selected = [norm_entity]
        else:
            norm_entity = None
            selected = list(_SUPPORTED_MODALITIES)

        res_payload: dict[str, Any] = {
            "status": "success",
            "total_models_available": total_models,
            "total_models": total_models,
            "supported_entities": list(_SUPPORTED_MODALITIES),
            "compute_device": device_str,
            "device": device_str,
            "hardware_device": device_str,
            "device_details": device_name,
            "recommended_models": {m: _RECOMMENDED_MODELS[m] for m in selected},
            "execution_paths": {m: execution_paths[m] for m in selected},
            "available_models": {m: available_models[m] for m in selected},
            "runnable_models": runnable_sc,
            "runnable_singlecell_models": runnable_sc,
            "all_singlecell_models": sc_models,
            "optional_extras_status": extras,
        }
        if norm_entity is not None:
            res_payload["entity_type"] = norm_entity
            res_payload["models"] = available_models[norm_entity]

        return res_payload

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
    entity_type: EmbeddingModality = "single_cell",
    obs_column: str | None = None,
    var_column: str | None = None,
    input_axis: Literal["auto", "obs", "var"] = "auto",
    id_type: str | None = None,
    organism: Literal["human", "mouse"] = "human",
    missing: Literal["error", "nan"] = "error",
    batch_size: int = 32,
    output_key: str | None = None,
    output_path: str | None = None,
    preprocessing: Literal["auto", "standard", "raw", "none"] = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    """Compute embeddings with BioEmbedder and save them into the AnnData on disk.

    Every model is checked against the modality before anything runs: a model is
    accepted only if the execution path for ``entity_type`` can run it (call
    list_model_registry for the per-modality lists). Modalities and their inputs:
    - 'single_cell': the expression matrix; one vector per cell in .obsm['X_<model>']
    - 'gene': gene symbols / Ensembl ids (var_names by default, or obs_column / var_column)
    - 'perturbation': per-cell genetic perturbation labels in obs_column; one vector per cell
    - 'dna': raw nucleotide sequences in obs_column / var_column
    - 'protein': protein or gene identifiers, or amino-acid sequences (set id_type)
    - 'molecule': SMILES (or names with id_type='name'), e.g. obs_column='smiles'
    - 'text': free text in obs_column / var_column
    - 'morphology': perturbation labels whose HPA/JUMP microscopy images are embedded

    For every modality except 'single_cell', the call is the same as the Python API
    ``BioEmbedder().embed(adata, entity_type=..., model=..., obs_column=..., var_column=...,
    anndata_axis=..., id_type=..., organism=...)``, so both produce the same matrix.

    Token-Saving: Returns array dimensions and the output file rather than matrices.

    Parameters
    ----------
    path : str
        Filesystem path to the input AnnData (.h5ad) dataset.
    models : list[str] or str, default="pca"
        Canonical registry keys (or aliases such as 'esm2_t33_650m' -> 'esm2_650M').
        All models must belong to the chosen modality.
    entity_type : {'gene', 'protein', 'molecule', 'dna', 'single_cell', 'morphology', 'text', 'perturbation'}, default='single_cell'
        Modality to embed; selects the execution path.
    obs_column : str, optional
        Column in adata.obs holding the inputs (e.g. 'smiles'). Embeddings align to obs rows.
    var_column : str, optional
        Column in adata.var holding the inputs. Embeddings align to var rows.
    input_axis : {'auto', 'obs', 'var'}, default='auto'
        Axis whose names are the inputs when no column is given ('auto' lets embpy choose:
        var_names for genes, obs_names otherwise).
    id_type : str, optional
        Identifier type of the inputs (e.g. 'symbol', 'ensembl_id', 'uniprot_id', 'sequence',
        'smiles', 'name'). Defaults to the embpy default for the modality.
    organism : {'human', 'mouse'}, default='human'
        Species used to resolve gene/protein identifiers.
    missing : {'error', 'nan'}, default='error'
        What to do when some rows get no embedding (unresolvable id, invalid SMILES):
        'error' fails and names them; 'nan' fills those rows with NaN.
    batch_size : int, default=32
        Inference batch size for 'single_cell' models.
    output_key : str, optional
        Key for the new embedding (only when computing a single model).
    output_path : str, optional
        Filesystem path to save the updated dataset (default: '<input>_embedded.h5ad').
    preprocessing : {'auto', 'standard', 'raw', 'none'}, default='auto'
        Single-cell preprocessing strategy (ignored by other modalities).
    device : str, default='auto'
        Compute device target ('auto', 'cpu', 'cuda', 'mps').

    Returns
    -------
    dict[str, Any]
        - status: "success" or "error"
        - output_path: Saved AnnData destination path
        - entity_type: Modality embedded
        - models_computed: Canonical model keys that were run
        - model_aliases_applied: Requested names that were mapped to a canonical key
        - embedding_keys: Keys added by this call (.obsm / .varm / .uns)
        - embedding_shapes: Shapes of the added embedding matrices
        - execution_time_seconds: Wall-clock execution time
    """
    t0 = time.time()
    model_list: list[str] = []
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

        ent_str = str(entity_type).lower().strip()
        if ent_str in ("cell", "singlecell"):
            ent_str = "single_cell"
        if ent_str not in _EMBED_ROUTES:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"entity_type must be one of {_SUPPORTED_MODALITIES}, got {entity_type!r}.",
                "path": str(path),
            }
        route = _EMBED_ROUTES[ent_str]

        resolved_models = [_canonical_model_name(m) for m in model_list]
        aliases_applied = {str(m): r for m, r in zip(model_list, resolved_models, strict=True) if str(m).strip() != r}
        problems = [msg for m in resolved_models if (msg := _validate_embedding_model(ent_str, m))]
        if problems:
            return {
                "status": "error",
                "error_type": "UnsupportedModelError",
                "error": " ".join(problems),
                "path": str(path),
                "models": resolved_models,
            }
        if obs_column and var_column:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Pass either obs_column or var_column, not both.",
                "path": str(path),
            }
        if route.get("is_perturbation") and not obs_column:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "entity_type='perturbation' needs obs_column: the adata.obs column holding each cell's perturbed gene.",
                "path": str(path),
            }
        if output_key and len(resolved_models) != 1:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "output_key can only be used when computing a single model.",
                "path": str(path),
            }

        import anndata
        from embpy import BioEmbedder

        adata = anndata.read_h5ad(resolved_path)
        before = {("obsm", k) for k in adata.obsm} | {("varm", k) for k in adata.varm} | {("uns", k) for k in adata.uns}

        with contextlib.redirect_stdout(sys.stderr):
            embedder = BioEmbedder(device=device)

            if ent_str == "single_cell":
                if obs_column or var_column:
                    return {
                        "status": "error",
                        "error_type": "ValueError",
                        "error": "obs_column/var_column do not apply to entity_type='single_cell', which embeds the expression matrix.",
                        "path": str(path),
                    }
                extra_kwargs: dict[str, Any] = {}
                if preprocessing == "auto":
                    # embed_cells' defaults (min_genes=200, 2000 HVGs) would filter out
                    # every cell of a small dataset; scale them as preprocess_counts does.
                    min_g, min_c, n_hvg, adjustments = _small_dataset_thresholds(adata.n_obs, adata.n_vars, 200, 3, 2000)
                    extra_kwargs.update(min_genes=min_g, min_cells=min_c, n_top_genes=n_hvg)
                    if adata.n_vars < 2000:
                        extra_kwargs["pca_use_hvg"] = False

                adata_out = embedder.embed_cells(
                    adata,
                    models=resolved_models,
                    batch_size=batch_size,
                    preprocessing=preprocessing,
                    obsm_prefix="X_",
                    **extra_kwargs,
                )
            else:
                adata_out = embedder.embed(
                    adata,
                    entity_type=route["backend"],
                    model=resolved_models[0] if len(resolved_models) == 1 else resolved_models,
                    id_type=id_type,
                    organism=organism,
                    obs_column=obs_column,
                    var_column=var_column,
                    anndata_axis=None if input_axis == "auto" else input_axis,
                    # Attach to the axis the inputs were read from.
                    attach_to="obs" if obs_column else "var" if var_column else "auto" if input_axis == "auto" else input_axis,
                    is_perturbation=bool(route.get("is_perturbation")),
                    key=output_key,
                    missing=missing,
                    output="anndata",
                )

        after = {("obsm", k) for k in adata_out.obsm} | {("varm", k) for k in adata_out.varm} | {("uns", k) for k in adata_out.uns}
        added = sorted(after - before)

        if output_key and ent_str == "single_cell":
            new_obsm = [k for slot, k in added if slot == "obsm"]
            if len(new_obsm) == 1 and new_obsm[0] != output_key:
                adata_out.obsm[output_key] = adata_out.obsm.pop(new_obsm[0])
                added = sorted((set(added) - {("obsm", new_obsm[0])}) | {("obsm", output_key)})

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_embedded{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        _save_anndata_safely(adata_out, saved_path)

        # Matrices land in .obsm (row-aligned) or .varm (feature-aligned); .uns
        # only gains provenance records, reported separately.
        matrix_keys = [(slot, key) for slot, key in added if slot in ("obsm", "varm")]
        shapes = {str(key): [int(d) for d in getattr(adata_out, slot)[key].shape] for slot, key in matrix_keys}

        return {
            "status": "success",
            "input_path": str(resolved_path),
            "output_path": saved_path,
            "entity_type": ent_str,
            "n_cells": int(adata_out.n_obs),
            "models_computed": resolved_models,
            "model_aliases_applied": aliases_applied,
            "preprocessing_adjustments": locals().get("adjustments") or {},
            "embedding_keys": [str(key) for _, key in matrix_keys],
            "embedding_slots": {str(key): slot for slot, key in matrix_keys},
            "embedding_shapes": shapes,
            "metadata_keys": [str(key) for slot, key in added if slot == "uns"],
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
                "models": [str(m) for m in (model_list or [models])],
            }
        logger.error("compute_embeddings failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": msg,
            "path": str(path),
            "models": [str(m) for m in (model_list or [models])],
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = "Install the model's optional dependencies (see list_model_registry optional_extras_status)."
        return resp


# Alias for backward compatibility
embed_cells = compute_embeddings


# =========================================================================
# 3. Annotation Skills
# =========================================================================

AnnotationType = Literal[
    "go",
    "pathways",
    "domains",
    "diseases",
    "interactions",
    "expression",
    "function",
    "location",
    "properties",
    "cell_line",
]

# Annotation types each entity type supports, and the databases that serve each one.
_ANNOTATION_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "gene": {
        "go": ("uniprot",),
        "domains": ("uniprot", "interpro"),
        "pathways": ("reactome", "kegg", "wikipathways"),
        "interactions": ("string", "dorothea"),
        "diseases": ("open_targets", "gwas_catalog"),
        "expression": ("gtex", "hpa"),
    },
    "protein": {
        "go": ("uniprot",),
        "domains": ("uniprot", "interpro"),
        "diseases": ("uniprot",),
        "interactions": ("uniprot",),
        "function": ("uniprot",),
        "location": ("uniprot",),
    },
    "molecule": {
        "properties": ("rdkit",),
        "interactions": ("chembl",),
        "pathways": ("kegg",),
        "diseases": ("pubchem",),
    },
    "cell_line": {
        "cell_line": ("depmap",),
    },
}
_DEFAULT_ANNOTATION_TYPE = {"gene": "pathways", "protein": "go", "molecule": "properties", "cell_line": "cell_line"}
# Served only for human genes (GTEx, HPA, GWAS Catalog, Open Targets).
_HUMAN_ONLY_ANNOTATIONS = {("gene", "diseases"), ("gene", "expression")}
_UNIPROT_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$"
)


def _n_annotation_items(value: Any) -> int:
    """Count annotation records: list entries, summed through nested dicts."""
    if isinstance(value, dict):
        return sum(_n_annotation_items(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value)
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return 0
    return 1


def _protein_id_type(identifier: str, id_type: str) -> str:
    if id_type != "auto":
        return id_type
    if _ENSEMBL_ID_RE.match(identifier):
        return "ensembl_id"
    if _UNIPROT_ACCESSION_RE.match(identifier):
        return "uniprot_id"
    return "symbol"


def _annotation_fetchers(entity: str, organism: str, id_type: str) -> dict[str, Any]:
    """Return {annotation_type: fn(identifier, sources) -> {source: data}} for ``entity``."""
    if entity == "gene":
        from embpy.resources.gene.annotator import GeneAnnotator
        from embpy.resources.protein.annotator import ProteinAnnotator

        ga = GeneAnnotator(organism=organism)
        pa = ProteinAnnotator(organism=organism)

        def _uniprot(gene: str, section: str) -> dict[str, Any]:
            res = pa.annotate(gene, id_type=_protein_id_type(gene, id_type), sources=[section])
            return res

        def go(gene, srcs):
            return {"uniprot": _uniprot(gene, "go").get("go_terms", {})}

        def domains(gene, srcs):
            res = _uniprot(gene, "domains")
            out = {"uniprot": res.get("uniprot_domains", []), "interpro": res.get("interpro_domains", [])}
            return {k: v for k, v in out.items() if k in srcs}

        def pathways(gene, srcs):
            return {k: v for k, v in ga.get_pathways(gene).items() if k in srcs}

        def interactions(gene, srcs):
            out: dict[str, Any] = {}
            if "string" in srcs:
                out["string"] = ga.get_protein_interactions(gene)
            if "dorothea" in srcs:
                out["dorothea"] = ga.get_transcription_factors(gene)
            return out

        def diseases(gene, srcs):
            out: dict[str, Any] = {}
            if "open_targets" in srcs:
                out["open_targets"] = ga.get_disease_associations(gene)
            if "gwas_catalog" in srcs:
                out["gwas_catalog"] = ga.get_gwas_associations(gene)
            return out

        def expression(gene, srcs):
            out: dict[str, Any] = {}
            if "gtex" in srcs:
                out["gtex"] = ga.get_tissue_expression(gene)
            if "hpa" in srcs:
                out["hpa"] = ga.get_subcellular_localization(gene)
            return out

        return {
            "go": go,
            "domains": domains,
            "pathways": pathways,
            "interactions": interactions,
            "diseases": diseases,
            "expression": expression,
        }

    if entity == "protein":
        from embpy.resources.protein.annotator import ProteinAnnotator

        pa = ProteinAnnotator(organism=organism)
        sections = {
            "go": (("go", "go_terms", "uniprot"),),
            "domains": (("domains", "uniprot_domains", "uniprot"), ("domains", "interpro_domains", "interpro")),
            "diseases": (("diseases", "disease_associations", "uniprot"),),
            "interactions": (("interactions", "interactions", "uniprot"),),
            "function": (("function", "function", "uniprot"),),
            "location": (("location", "subcellular_location", "uniprot"),),
        }

        def _make(ann_type):
            def fetch(ident, srcs):
                section = sections[ann_type][0][0]
                res = pa.annotate(ident, id_type=_protein_id_type(ident, id_type), sources=[section])
                return {src: res.get(key) for _, key, src in sections[ann_type] if src in srcs and key in res}

            return fetch

        return {t: _make(t) for t in sections}

    if entity == "molecule":
        from embpy.resources.molecule.annotator import MoleculeAnnotator

        ma = MoleculeAnnotator()
        native = {
            "properties": ("structural", ("physicochemical",), "rdkit"),
            "interactions": ("bioactivity", ("targets", "bioactivities", "mechanism_of_action"), "chembl"),
            "pathways": ("pathways", ("kegg_pathways",), "kegg"),
            "diseases": ("diseases", ("disease_associations",), "pubchem"),
        }

        def _make(ann_type):
            group, keys, src = native[ann_type]

            def fetch(ident, srcs):
                res = ma.annotate(ident, sources=[group])
                return {src: {k: res.get(k) for k in keys if k in res}}

            return fetch

        return {t: _make(t) for t in native}

    return {}


def annotate_metadata(
    path: str,
    entity_type: Literal["auto", "gene", "protein", "molecule", "cell_line"] = "auto",
    annotation_types: list[AnnotationType] | AnnotationType | None = None,
    annotation_type: AnnotationType | None = None,
    column: str | None = None,
    obs_column: str | None = None,
    sources: list[str] | str | None = None,
    id_type: Literal["auto", "symbol", "ensembl_id", "uniprot_id"] = "auto",
    output_path: str | None = None,
    organism: Literal["human", "mouse"] = "human",
) -> dict[str, Any]:
    """Annotate the entities in an adata.obs column with external biological knowledge.

    Every requested annotation type is run for every unique entity, in the requested
    organism, against the requested sources. An entity counts as annotated for a type
    only if that type's sources returned records: echoing the input name back is not
    an annotation. Empty answers ("no matching annotations") and failed requests
    ("the service failed") are counted separately.

    Supported annotation types (and sources) per entity type:
    - gene: go (uniprot), domains (uniprot, interpro), pathways (reactome, kegg, wikipathways),
      interactions (string, dorothea), diseases (open_targets, gwas_catalog; human only),
      expression (gtex, hpa; human only)
    - protein: go, domains (uniprot, interpro), diseases, interactions, function, location (uniprot)
    - molecule: properties (rdkit), interactions (chembl), pathways (kegg), diseases (pubchem)
    - cell_line: cell_line (depmap, via pertpy)

    Token-Saving: Writes the enriched dataset to disk and returns per-type counts.

    Parameters
    ----------
    path : str
        Filesystem path to the target .h5ad dataset.
    entity_type : {'auto', 'gene', 'protein', 'molecule', 'cell_line'}, default='auto'
        Entity type held in the column. 'auto' guesses from column names.
    annotation_types : list or str, optional
        Annotation types to retrieve (see above). Defaults per entity type: gene 'pathways',
        protein 'go', molecule 'properties', cell_line 'cell_line'.
    annotation_type : str, optional
        Singular form; combined with annotation_types.
    column : str, optional
        Column in adata.obs holding entity identifiers. Required when no column name
        obviously matches the entity type.
    obs_column : str, optional
        Alias for column.
    sources : list[str] or str, optional
        Restrict the databases queried (e.g. ['reactome'] for pathways). Every source must
        serve at least one requested type, and every requested type needs one of them.
    id_type : {'auto', 'symbol', 'ensembl_id', 'uniprot_id'}, default='auto'
        Identifier type for UniProt-backed annotations (gene/protein go, domains, ...).
    output_path : str, optional
        Filesystem destination (default: '<input>_annotated.h5ad'). Not written when
        nothing was annotated.
    organism : {'human', 'mouse'}, default='human'
        Species for gene and protein annotations.

    Returns
    -------
    dict[str, Any]
        - status: "success" (every entity has every requested type), "partial_success",
          or "error" (nothing annotated; error_type is "AnnotationServiceError" when requests
          failed and "NoAnnotationsFound" when the services answered with nothing)
        - per_type: For each type, counts of entities annotated / no_match / service_error
        - sources_queried: Databases queried per type
        - successful_count: Entities annotated for every requested type
        - failed_count: Entities missing at least one requested type
        - new_columns_added: '<entity>_<type>_n' record counts per row in .obs
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
        import pandas as pd

        from embpy.resources._service_errors import record_service_errors

        adata = anndata.read_h5ad(resolved_path)

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
        if resolved_entity not in _ANNOTATION_SOURCES:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"entity_type must be one of {['auto', *_ANNOTATION_SOURCES]}, got {entity_type!r}.",
            }
        supported = _ANNOTATION_SOURCES[resolved_entity]

        # Requested annotation types: every entry of both arguments, in order.
        requested: list[str] = []
        for value in (annotation_types, annotation_type):
            if value is None:
                continue
            for t in [value] if isinstance(value, str) else list(value):
                if str(t) not in requested:
                    requested.append(str(t))
        if not requested:
            requested = [_DEFAULT_ANNOTATION_TYPE[resolved_entity]]
        unsupported = [t for t in requested if t not in supported]
        if unsupported:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": (
                    f"Annotation type(s) {unsupported} are not available for {resolved_entity} entities. "
                    f"Supported: {list(supported)}."
                ),
            }
        org = str(organism).lower()
        if org not in ("human", "mouse"):
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"organism must be 'human' or 'mouse', got {organism!r}.",
            }
        human_only = [t for t in requested if (resolved_entity, t) in _HUMAN_ONLY_ANNOTATIONS]
        if human_only and org != "human":
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"Annotation type(s) {human_only} are only available for human genes (organism={organism!r}).",
            }

        # Sources: default to every source of each requested type; otherwise validate.
        source_list = [sources] if isinstance(sources, str) else list(sources or [])
        sources_queried: dict[str, list[str]] = {}
        if source_list:
            offered = {s for t in requested for s in supported[t]}
            unknown = [s for s in source_list if s not in offered]
            if unknown:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"Source(s) {unknown} do not serve the requested annotation types {requested}. "
                        f"Available: { {t: list(supported[t]) for t in requested} }."
                    ),
                }
            for t in requested:
                chosen = [s for s in supported[t] if s in source_list]
                if not chosen:
                    return {
                        "status": "error",
                        "error_type": "ValueError",
                        "error": f"None of the sources {source_list} serves annotation type {t!r}; it needs one of {list(supported[t])}.",
                    }
                sources_queried[t] = chosen
        else:
            for t in requested:
                sources_queried[t] = [
                    s for s in supported[t] if not (s == "dorothea" and importlib.util.find_spec("decoupler") is None)
                ]
        if any(s == "dorothea" for t in requested for s in sources_queried[t]) and importlib.util.find_spec("decoupler") is None:
            return {
                "status": "error",
                "error_type": "ImportError",
                "error": "Source 'dorothea' needs the decoupler package: pip install decoupler.",
            }

        query_col = obs_column or column
        if not query_col:
            keywords = {
                "molecule": ("smiles", "compound", "drug", "molecule"),
                "cell_line": ("cell_line", "cellline", "depmap", "modelid"),
                "protein": ("protein", "uniprot", "gene"),
                "gene": ("perturbation", "gene", "symbol"),
            }[resolved_entity]
            candidates = [c for c in adata.obs.columns if any(k in str(c).lower() for k in keywords)]
            if not candidates:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"No adata.obs column looks like it holds {resolved_entity} identifiers. "
                        f"Pass column=... (available: {list(adata.obs.columns)})."
                    ),
                }
            query_col = candidates[0]

        if query_col not in adata.obs.columns:
            return {
                "status": "error",
                "error_type": "KeyError",
                "error": f"Target column '{query_col}' not found in adata.obs. Available: {list(adata.obs.columns)}",
            }

        values = [str(v).strip() if pd.notna(v) else "" for v in adata.obs[query_col]]
        entities = list(dict.fromkeys(v for v in values if v))
        blank_count = sum(1 for v in values if not v)
        if not entities:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": f"Column '{query_col}' holds no identifiers.",
            }

        # outcome[type][entity] = "annotated" | "no_match" | "service_error"
        outcome: dict[str, dict[str, str]] = {t: {} for t in requested}
        counts: dict[str, dict[str, int]] = {t: {} for t in requested}
        results: dict[str, dict[str, Any]] = {e: {} for e in entities}
        service_errors: list[dict[str, str]] = []
        new_cols: list[str] = []

        with contextlib.redirect_stdout(sys.stderr):
            if resolved_entity == "cell_line":
                import embpy.tl as tl

                existing = set(adata.obs.columns)
                try:
                    adata = tl.annotate_cell_lines(adata, query_id=query_col)
                except (ImportError, ModuleNotFoundError):
                    raise
                except Exception as exc:  # the DepMap download or lookup failed
                    service_errors.append({"entity": "*", "type": "cell_line", "error": f"{type(exc).__name__}: {exc}"})
                added = [c for c in adata.obs.columns if c not in existing]
                for e in entities:
                    if service_errors:
                        outcome["cell_line"][e] = "service_error"
                        continue
                    rows = adata.obs.loc[[v == e for v in values], added] if added else None
                    n = int(rows.notna().any(axis=0).sum()) if rows is not None and len(rows) else 0
                    counts["cell_line"][e] = n
                    outcome["cell_line"][e] = "annotated" if n else "no_match"
                new_cols.extend(str(c) for c in added)
            else:
                fetchers = _annotation_fetchers(resolved_entity, org, id_type)
                for e in entities:
                    for t in requested:
                        with record_service_errors() as errs:
                            try:
                                data = fetchers[t](e, sources_queried[t])
                            except Exception as exc:  # noqa: BLE001
                                data = {}
                                errs.append({"url": "", "error": f"{type(exc).__name__}: {exc}"})
                        n = _n_annotation_items(data)
                        counts[t][e] = n
                        results[e][t] = data
                        if n:
                            outcome[t][e] = "annotated"
                        elif errs:
                            outcome[t][e] = "service_error"
                            service_errors.append({"entity": e, "type": t, "error": errs[0]["error"]})
                        else:
                            outcome[t][e] = "no_match"

        per_type = {
            t: {k: sum(1 for v in outcome[t].values() if v == k) for k in ("annotated", "no_match", "service_error")}
            for t in requested
        }
        fully = [e for e in entities if all(outcome[t].get(e) == "annotated" for t in requested)]
        any_annotated = [e for e in entities if any(outcome[t].get(e) == "annotated" for t in requested)]
        n_annotated_pairs = sum(p["annotated"] for p in per_type.values())

        type_label = {"go": "GO terms", "cell_line": "cell-line metadata"}
        summary_parts = []
        for t in requested:
            p = per_type[t]
            part = f"{type_label.get(t, t)}: {p['annotated']}/{len(entities)} {resolved_entity} entities annotated"
            extras = [f"{p[k]} {k.replace('_', ' ')}" for k in ("no_match", "service_error") if p[k]]
            summary_parts.append(part + (f" ({', '.join(extras)})" if extras else ""))
        summary = "; ".join(summary_parts) + f" (column '{query_col}', organism {org})."

        base_payload: dict[str, Any] = {
            "entity_type": resolved_entity,
            "organism": org,
            "target_column": query_col,
            "annotation_types": requested,
            "annotation_type": requested[0] if len(requested) == 1 else requested,
            "sources_queried": sources_queried,
            "total_entities": len(entities),
            "blank_count": blank_count,
            "per_type": per_type,
            "successful_count": len(fully),
            "failed_count": len(entities) - len(fully),
            "entities_annotated": len(fully),
            "entities_with_any_annotation": len(any_annotated),
            "successful_items": fully[:10],
            "failed_items": [e for e in entities if e not in fully][:10],
            "summary": summary,
        }
        if service_errors:
            base_payload["service_errors_sample"] = service_errors[:5]

        if n_annotated_pairs == 0:
            any_service_error = any(p["service_error"] for p in per_type.values())
            return {
                "status": "error",
                "error_type": "AnnotationServiceError" if any_service_error else "NoAnnotationsFound",
                "error": (
                    f"Annotation requests failed for the entities in column '{query_col}'; nothing was annotated."
                    if any_service_error
                    else f"The services returned no {requested} annotations for the entities in column '{query_col}'."
                ),
                **base_payload,
            }

        if resolved_entity != "cell_line":
            prefix = {"gene": "gene", "protein": "prot", "molecule": "mol"}[resolved_entity]
            for t in requested:
                col = f"{prefix}_{t}_n"
                adata.obs[col] = [counts[t].get(v, 0) if v else 0 for v in values]
                new_cols.append(col)
            if resolved_entity == "molecule" and "properties" in requested:
                # Scalar RDKit properties as columns, as tl.annotate_molecules writes them.
                props = {e: (results[e].get("properties", {}).get("rdkit") or {}).get("physicochemical") or {} for e in entities}
                prop_keys = list(dict.fromkeys(k for p in props.values() for k, v in p.items() if not isinstance(v, (dict, list))))
                for key in prop_keys:
                    adata.obs[f"mol_{key}"] = [props.get(v, {}).get(key) if v else None for v in values]
                    new_cols.append(f"mol_{key}")
            adata.uns["embpy_annotations"] = {
                "entity_type": resolved_entity,
                "column": query_col,
                "organism": org,
                "annotation_types": requested,
                "results_json": json.dumps(results, default=str),
            }

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_annotated{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        _save_anndata_safely(adata, saved_path)

        return {
            "status": "success" if n_annotated_pairs == len(entities) * len(requested) else "partial_success",
            "output_path": saved_path,
            "new_columns_added": new_cols,
            "added_columns": new_cols,
            **base_payload,
        }

    except Exception as exc:
        logger.error("annotate_metadata failed: %s", exc, exc_info=True)
        resp: dict[str, Any] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "successful_count": 0,
            "suggestion": "Check that the target dataset contains valid identifiers in .obs.",
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = "Cell-line annotation needs pertpy: pip install 'embpy[pertpy]'."
        return resp


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

        _save_anndata_safely(adata, saved_path)

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


def _same_observations(a: Any, b: Any) -> bool:
    """True when two AnnData objects hold the same cells with the same expression values."""
    if a.shape != b.shape or list(a.obs_names) != list(b.obs_names):
        return False
    import scipy.sparse as sp

    xa = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    xb = b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X)
    return bool(np.array_equal(xa, xb))


def benchmark_perturbations(
    path: str,
    control_pert: str,
    pert_col: str = "perturbation",
    split_col: str | None = None,
    pred_label: str | None = None,
    ref_label: str | None = None,
    path_pred: str | None = None,
    embedding_keys: list[str] | str | None = None,
    output_csv: str | None = None,
    output_dir: str | None = None,
    profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "minimal",
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
        Column in adata.obs indicating data splits. Requires pred_label and ref_label.
    pred_label : str, optional
        Category in adata.obs[split_col] designating model predictions.
    ref_label : str, optional
        Category in adata.obs[split_col] designating real reference observations.
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
            if os.path.normcase(resolved_pred) == os.path.normcase(resolved_real):
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": "path_pred is the same file as path: predictions would be scored against themselves.",
                }
            adata_real = anndata.read_h5ad(resolved_real)
            adata_pred = anndata.read_h5ad(resolved_pred)
            if _same_observations(adata_pred, adata_real):
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        "The prediction and reference datasets contain the same cells and expression values: "
                        "predictions would be scored against themselves."
                    ),
                }
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

            # Reject ambiguous split inputs: require explicit pred_label and ref_label
            if not pred_label or not ref_label:
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"Ambiguous benchmark inputs: when using 'split_col', you must explicitly specify "
                        f"both 'pred_label' (e.g. 'test') and 'ref_label' (e.g. 'train'). "
                        f"Available categories in adata.obs['{split_col}']: {unique_splits}."
                    ),
                }

            # Comparing a group with itself scores perfectly and means nothing.
            if str(pred_label) == str(ref_label):
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"pred_label and ref_label are both '{pred_label}': predictions would be scored against "
                        f"themselves. Choose two different categories of adata.obs['{split_col}']: {unique_splits}."
                    ),
                }

            split_values = adata_full.obs[split_col].astype(str)
            if str(pred_label) not in set(split_values) or str(ref_label) not in set(split_values):
                return {
                    "status": "error",
                    "error_type": "ValueError",
                    "error": (
                        f"Specified pred_label '{pred_label}' or ref_label '{ref_label}' not found in "
                        f"adata.obs['{split_col}']. Available values: {unique_splits}."
                    ),
                }

            pred_mask = (split_values == str(pred_label)).to_numpy()
            ref_mask = (split_values == str(ref_label)).to_numpy()
            adata_pred = adata_full[pred_mask].copy()
            adata_real = adata_full[ref_mask].copy()
        else:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Either 'path_pred' (path to predicted AnnData) or 'split_col' with explicit 'pred_label' and 'ref_label' must be specified.",
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
