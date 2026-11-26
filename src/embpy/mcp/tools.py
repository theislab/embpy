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
import threading
import time
import uuid
from typing import Any, Literal

import numpy as np
import requests

from embpy._counts import NOT_COUNTS
from embpy._counts import counts_source as _counts_source
from embpy._counts import describe_values as _describe_values
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
    """Write an AnnData object to an .h5ad file without ever leaving a half-written file there.

    The data is written to a temporary file in the destination's folder and moved
    over ``path`` only once the write has succeeded, so a failed write (for example
    a column h5py cannot store) leaves an existing file at ``path`` untouched.

    Access is what collaborators expect: the temporary file is created like any
    other new file, so a new output gets the folder's sharing settings (umask or
    default ACL), and when an existing file is replaced its group, mode and access
    control list are copied onto the new file first.
    """
    target = os.path.abspath(path)
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    if getattr(adata, "uns", None):
        for k in list(adata.uns.keys()):
            adata.uns[k] = _sanitize_for_h5ad(adata.uns[k])
    tmp_path = _new_sibling_file(target_dir)
    try:
        adata.write_h5ad(tmp_path)
        if os.path.exists(target):
            _copy_access(target, tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _new_sibling_file(folder: str) -> str:
    """Create an empty, uniquely named ``.h5ad`` file in ``folder`` as an ordinary new file.

    ``os.open(..., 0o666)`` lets the system apply the umask or the folder's default
    ACL exactly as for any program's new file; ``mkstemp`` would create it owner-only
    and an ACL mask derived from that would lock collaborators out.
    """
    import secrets

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        candidate = os.path.join(folder, f".embpy-tmp-{os.getpid()}-{secrets.token_hex(6)}.h5ad")
        try:
            fd = os.open(candidate, flags, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise FileExistsError(f"Could not create a temporary file in {folder}")


_POSIX_ACL_XATTR = "system.posix_acl_access"


def _copy_access(source: str, destination: str) -> None:
    """Give ``destination`` the group, mode and POSIX ACL of ``source`` (best effort).

    The group change only succeeds for groups the user belongs to, and ACLs only
    where the system exposes them as extended attributes (Linux); elsewhere the
    mode bits are copied.
    """
    import stat

    info = os.stat(source)
    if hasattr(os, "chown"):
        with contextlib.suppress(OSError):
            os.chown(destination, -1, info.st_gid)
    os.chmod(destination, stat.S_IMODE(info.st_mode))
    if hasattr(os, "getxattr"):
        try:
            acl = os.getxattr(source, _POSIX_ACL_XATTR)
        except OSError:  # no ACL beyond the mode bits, or ACLs unsupported here
            acl = None
        if acl:
            with contextlib.suppress(OSError):
                os.setxattr(destination, _POSIX_ACL_XATTR, acl)


def _detect_expression_nature(adata: Any, layer: str | None = None) -> dict[str, Any]:
    """Inspect AnnData to determine if .X (or ``layer``) holds raw integer counts or normalized expression.

    Uses the same rule as embpy.pp.preprocess_counts (embpy._counts.describe_values), so
    every tool reads a file the same way: integers are raw_counts, non-integers with
    count-like magnitudes (corrected or EM counts) non_integer_counts, and log-transformed
    (marked by .uns['log1p'] or in the log range), normalized or scaled values are not counts.
    """
    has_counts_layer = "counts" in adata.layers
    has_norm_layer = any(k in adata.layers for k in ("log_normalized", "normalized", "log1p"))

    nature = "unknown"
    is_integer = False
    has_negative = False
    min_val: float | None = None
    max_val: float | None = None

    try:
        info = _describe_values(
            adata.X if layer is None else adata.layers[layer],
            log1p_marked=layer is None and "log1p" in adata.uns,
        )
        nature = info["kind"]
        is_integer, has_negative = info["is_integer"], info["has_negative"]
        min_val, max_val = info["min"], info["max"]
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


def _layer_names(adata: Any) -> list[str]:
    """Names of the layers in ``adata``; anndata >= 0.13 also lists .X there, as layers[None]."""
    return [str(k) for k in adata.layers.keys() if k is not None]


# detected_nature values that are clearly not counts; the same kinds embpy.pp refuses to use as counts.
_NOT_RAW_COUNTS = NOT_COUNTS


def _found_values(nature: dict[str, Any], counts_layer: str | None) -> str:
    """'<where> holds <what>' for a raw-count source that holds something else."""
    source = f".layers['{counts_layer}']" if counts_layer else ".X (and there is no .layers['counts'])"
    return (
        f"{source} holds {nature['detected_nature']} values (sampled range {nature['sample_min']:.3g} to "
        f"{nature['sample_max']:.3g}, {'integers' if nature['is_discrete_integers'] else 'not integers'})"
    )


def _not_raw_counts_error(
    path: str,
    nature: dict[str, Any],
    counts_layer: str | None,
    needed_by: str = "preprocess_counts",
    consequence: str = "Normalizing and log-transforming them again would distort the data.",
    remedy: str = (
        "If the data is already normalized, skip preprocess_counts. If these values are counts "
        "(e.g. ambient-corrected, non-integer counts), call again with assume_counts=True."
    ),
) -> dict[str, Any]:
    """Error reply when a step that starts from raw counts is given normalized values."""
    return {
        "status": "error",
        "error_type": "NotRawCountsError",
        "error": f"{needed_by} needs raw counts, but {_found_values(nature, counts_layer)}. {consequence} {remedy}",
        "path": str(path),
        "expression_nature": _sanitize_val(nature),
    }


def _raw_count_model_problems(adata: Any, models: list[str]) -> dict[str, str]:
    """Models that read raw counts (model card input_layer 'counts') whose input holds other values.

    Their input is .layers['counts'], or .X when that layer is missing. Returns model -> error.
    """
    from embpy.models.singlecell_models import singlecell_info

    needs_counts = [m for m in models if singlecell_info(m).input_layer == "counts"]
    if not needs_counts:
        return {}
    layer = "counts" if "counts" in adata.layers else None
    nature = _detect_expression_nature(adata, layer=layer)
    if nature["detected_nature"] not in _NOT_RAW_COUNTS:
        return {}
    found = _found_values(nature, layer)
    return {m: f"{m} needs raw counts in .X or .layers['counts'], but {found}." for m in needs_counts}


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
                "layers": _layer_names(adata),
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
    assume_counts: bool = False,
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
    assume_counts : bool, default=False
        The counts are read from .X when it holds counts (integers, or non-integer counts
        such as ambient-corrected or EM counts, with a warning), and from .layers['counts']
        only when .X is clearly not counts (log-transformed, normalized or scaled) and that
        layer holds counts: the rule compute_embeddings follows too (counts_source in the
        reply names the matrix used). If neither holds counts, the call fails with
        error_type 'NotRawCountsError' instead of normalizing normalized data again. Pass
        True to use .X as the counts anyway.

    Returns
    -------
    dict[str, Any]
        Preprocessed dataset metadata:
        - status: "success" or "error" ('NotRawCountsError' for input that is not raw
          counts; 'EmptyAfterFilteringError', with the thresholds used, when filtering
          would leave no cell or no gene; no file is written in either case)
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

        # The counts to start from, by the rule embpy.pp.preprocess_counts (and so
        # compute_embeddings) follows: .X when it holds counts (integers, or corrected / EM
        # counts), .layers['counts'] only when .X is clearly not counts.
        counts_layer = "counts" if "counts" in adata.layers else None
        choice = _counts_source(adata)
        source_warnings: list[str] = []
        # 'none' only writes .layers['counts'] (from .X) when it is missing.
        if preprocessing != "none" or counts_layer is None:
            if not choice["found"] and not assume_counts:
                return _not_raw_counts_error(path, _detect_expression_nature(adata, layer=counts_layer), counts_layer)
            if choice["found"] and choice["message"]:
                source_warnings.append(choice["message"])
        use_counts_layer = choice["source"] == "counts"

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
            reply = {
                "status": "success",
                "input_path": str(resolved_path),
                "output_path": saved_path,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_hvg": 0,
                "hvg_count": 0,
                "layers": _layer_names(adata),
                "summary": (
                    f"Preprocessing bypassed (mode='none'). Saved {adata.n_obs} cells and "
                    f"{adata.n_vars} genes with raw counts preserved in .layers['counts']."
                ),
            }
            if source_warnings:
                reply["warning"] = " ".join(source_warnings)
            return reply

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
            # Normalize from the counts chosen above, and keep them in .X and layers['counts'].
            if use_counts_layer:
                adata.X = adata.layers["counts"].copy()
                # .uns['log1p'] described the replaced .X, which now holds counts.
                adata.uns.pop("log1p", None)
            else:
                adata.layers["counts"] = adata.X.copy()

            # QC Filtering
            n_cells_input, n_genes_input = int(adata.n_obs), int(adata.n_vars)
            if effective_min_genes > 0 and adata.n_obs > 0:
                sc.pp.filter_cells(adata, min_genes=effective_min_genes)
            if effective_min_cells > 0 and adata.n_vars > 0 and adata.n_obs > 0:
                sc.pp.filter_genes(adata, min_cells=effective_min_cells)
            if adata.n_obs == 0 or adata.n_vars == 0:
                emptied = "cell" if adata.n_obs == 0 else "gene"
                return {
                    "status": "error",
                    "error_type": "EmptyAfterFilteringError",
                    "error": (
                        f"Filtering with min_genes={effective_min_genes} and min_cells={effective_min_cells} "
                        f"removed every {emptied} (input: {n_cells_input} cells x {n_genes_input} genes), so "
                        "no file was written. Lower min_genes / min_cells, or use preprocessing='auto' to "
                        "scale them to the dataset."
                    ),
                    "path": str(path),
                    "thresholds": {"min_genes": int(effective_min_genes), "min_cells": int(effective_min_cells)},
                    "n_cells_input": n_cells_input,
                    "n_genes_input": n_genes_input,
                    "n_cells_after_filtering": int(adata.n_obs),
                    "n_genes_after_filtering": int(adata.n_vars),
                    **({"adjustments_applied": adjustments_applied} if adjustments_applied else {}),
                }

            # Normalization & Log1p
            if target_sum and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.normalize_total(adata, target_sum=target_sum)
            if log_transform and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.log1p(adata)

            # Store log-normalized representation in layers
            if adata.n_obs > 0 and adata.n_vars > 0:
                adata.layers["log_normalized"] = adata.X.copy()

            # HVG selection runs before scaling: 'seurat' expects log-normalized
            # expression (.X here) and 'seurat_v3' expects raw counts.
            if select_hvg and adata.n_vars > 2 and adata.n_obs > 0:
                flavor = "seurat" if log_transform else "seurat_v3"
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=min(effective_n_hvg, adata.n_vars),
                    flavor=flavor,
                    layer="counts" if flavor == "seurat_v3" else None,
                )

            # Scale if requested
            if scale and adata.n_obs > 0 and adata.n_vars > 0:
                sc.pp.scale(adata, max_value=10)
                adata.layers["scaled"] = adata.X.copy()

            # Restore .X to raw counts to honor embpy output contract. uns['log1p'] tells
            # scanpy (and cell-eval) that .X is log-transformed, which it no longer is.
            if "counts" in adata.layers and adata.n_obs > 0:
                adata.X = adata.layers["counts"].copy()
                adata.uns.pop("log1p", None)

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
            "layers": _layer_names(adata),
            "counts_source": ".layers['counts']" if use_counts_layer else ".X",
            "summary": (
                f"Successfully preprocessed {adata.n_obs} cells and {adata.n_vars} genes. "
                f"Stored raw counts in .layers['counts'], log-normalized expression in .layers['log_normalized'], "
                f"and selected {n_hvg_count} highly-variable genes."
            ),
        }
        if adjustments_applied:
            res_payload["adjustments_applied"] = adjustments_applied
        if source_warnings:
            res_payload["warning"] = " ".join(source_warnings)
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
            resp["suggestion"] = _install_hint(str(exc)) or 'Install scanpy: pip install "embpy[mcp]".'
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
        extras = {
            "mcp": {"installed": bool(importlib.util.find_spec("mcp") is not None)},
            "models": {"installed": bool(importlib.util.find_spec("torch") is not None)},
            "seqmodels": {"installed": bool(importlib.util.find_spec("borzoi_pytorch") is not None)},
            "genome": {"installed": bool(importlib.util.find_spec("pyensembl") is not None or importlib.util.find_spec("pysam") is not None)},
            "eval": {"installed": bool(importlib.util.find_spec("cell_eval") is not None)},
            "scib": {"installed": bool(importlib.util.find_spec("scib") is not None)},
            "scanpy": {"installed": bool(importlib.util.find_spec("scanpy") is not None)},
        }

        try:
            import torch
        except ImportError as exc:
            # Every embedding model (PCA included) runs through embpy's torch-based stack.
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": f"PyTorch is not installed, so no embedding model can run ({exc}).",
                "suggestion": _install_hint(str(exc)) or _INSTALL_FOR_MODULE["torch"],
                "optional_extras_status": extras,
            }

        if torch.cuda.is_available():
            device_str = "cuda"
            device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_name = "Apple Silicon (MPS)"
        else:
            device_str = "cpu"
            device_name = "CPU"

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
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        hint = _install_hint(str(exc)) if isinstance(exc, ImportError) else None
        if hint:
            resp["suggestion"] = hint
        return resp


# Alias for backward compatibility
list_available_models = list_model_registry


# .uns entries that collect one record per embedding key; merged, not replaced.
_UNS_COLLECTIONS = ("embeddings", "uns_embeddings", "perturbations")

# embed_cells writes this call's matrices under this prefix first; a matrix moves to
# its final key (X_<model> or output_key) only when its model succeeded.
_NEW_CELL_EMBEDDING_PREFIX = "_embpy_new_X_"
_CELL_EMBEDDING_RECORDS = "embpy_cell_embeddings"

# Missing module -> what to install, for the hints in error responses.
_INSTALL_FOR_MODULE = {
    "torch": (
        'pip install "embpy[models]" (on Linux without a GPU, first: '
        "pip install torch --index-url https://download.pytorch.org/whl/cpu)"
    ),
    "transformers": 'pip install "embpy[models]"',
    "scvi": "pip install scvi-tools",
    "skmisc": "pip install scikit-misc",
    "scanpy": 'pip install "embpy[mcp]"',
    # Same pin as the mcp extra in pyproject.toml.
    "umap": 'pip install "umap-learn<0.6"',
    "scib": 'pip install "embpy[scib]"',
    "cell_eval": 'pip install "embpy[eval]"',
    "pertpy": 'pip install "embpy[pertpy]"',
    "pyensembl": 'pip install "embpy[genome]"',
    "rdkit": "pip install rdkit",
}


def _install_hint(message: str) -> str | None:
    """What to install for the first "No module named ..." in ``message``, if there is one."""
    match = re.search(r"No module named '([A-Za-z0-9_]+)", message)
    if not match:
        return None
    module = match.group(1)
    command = _INSTALL_FOR_MODULE.get(module)
    if command:
        return f"The package providing '{module}' is not installed: {command}"
    return f"The Python module '{module}' is not installed; install the package that provides it."


def _settle_cell_embeddings(
    adata: Any,
    earlier_records: dict[str, Any],
    models: list[str],
    output_key: str | None,
    prefix: str = _NEW_CELL_EMBEDDING_PREFIX,
) -> tuple[list[tuple[str, str]], list[str], dict[str, str], list[str]]:
    """Move each successful model's matrix to its final key; leave a failed model's data alone.

    embed_cells wrote this call's matrices under ``prefix`` (unique per call) and
    recorded, per model, the key it wrote or the error it hit. A successful model's
    matrix replaces whatever ``X_<model>`` (or ``output_key``) held, and its record
    points at that key. A failed model changes nothing: an earlier matrix and its
    record stay as they were, and the key is reported as not updated.

    Returns ``(written, succeeded, failed, not_updated)``.
    """
    run_records = dict(adata.uns.get(_CELL_EMBEDDING_RECORDS) or {})
    records = dict(earlier_records)
    written: list[tuple[str, str]] = []
    succeeded: list[str] = []
    failed: dict[str, str] = {}
    not_updated: list[str] = []
    for model in models:
        record = run_records.get(model)
        new_key = record.get("obsm_key") if isinstance(record, dict) else None
        final_key = output_key or f"X_{model}"
        if isinstance(record, dict) and "error" not in record and new_key in adata.obsm:
            adata.obsm[final_key] = adata.obsm[new_key]
            del adata.obsm[new_key]
            records[model] = {**record, "obsm_key": final_key}
            if f"{model}__vocab" in run_records:
                records[f"{model}__vocab"] = run_records[f"{model}__vocab"]
            succeeded.append(model)
            written.append(("obsm", final_key))
        else:
            error = record.get("error") if isinstance(record, dict) else None
            failed[model] = str(error or "the model returned no embedding")
            if final_key in adata.obsm:
                not_updated.append(final_key)
    # Only this call's temporary matrices; other keys that share the prefix stay.
    for key in [k for k in adata.obsm.keys() if str(k).startswith(prefix)]:
        del adata.obsm[key]
    if succeeded:
        if "__preprocessing__" in run_records:
            records["__preprocessing__"] = run_records["__preprocessing__"]
        adata.uns[_CELL_EMBEDDING_RECORDS] = records
        written.append(("uns", _CELL_EMBEDDING_RECORDS))
    elif earlier_records:
        adata.uns[_CELL_EMBEDDING_RECORDS] = records
    else:
        adata.uns.pop(_CELL_EMBEDDING_RECORDS, None)
    return written, succeeded, failed, not_updated


def _cell_key_conflicts(
    earlier_records: dict[str, Any], models: list[str], output_key: str | None
) -> dict[str, list[str]]:
    """Keys this call would write that an earlier record of another model points at: key -> owners."""
    owners: dict[str, list[str]] = {}
    for owner, record in earlier_records.items():
        if isinstance(record, dict) and record.get("obsm_key") and not str(owner).startswith("__"):
            owners.setdefault(str(record["obsm_key"]), []).append(str(owner))
    conflicts: dict[str, list[str]] = {}
    for model in models:
        key = output_key or f"X_{model}"
        others = [owner for owner in owners.get(key, []) if owner != model]
        if others:
            conflicts[key] = others
    return conflicts


def _key_conflict_error(path: str, models: list[str], conflicts: dict[str, list[str]]) -> dict[str, Any]:
    """Error reply when compute_embeddings would save over another model's recorded embedding."""
    details = "; ".join(
        f".obsm['{key}'] holds the embedding of {', '.join(repr(o) for o in owners)} "
        f"(uns['{_CELL_EMBEDDING_RECORDS}'])"
        for key, owners in conflicts.items()
    )
    return {
        "status": "error",
        "error_type": "OutputKeyConflictError",
        "error": (
            f"{details}. Saving {models} there would leave two model records pointing at one matrix. "
            "Pass an output_key no other model uses (or recompute that model itself). "
            "Nothing was computed or written."
        ),
        "path": str(path),
        "models": models,
        "conflicting_keys": conflicts,
    }


def _note_preprocessing_choice(adata: Any, requested: str, reason: str) -> None:
    """Record in embed_cells' provenance which mode was requested and why it was not run."""
    records = adata.uns.get(_CELL_EMBEDDING_RECORDS) or {}
    plans = [records.get("__preprocessing__")]
    plans += [record.get("preprocessing") for record in records.values() if isinstance(record, dict)]
    for plan in plans:
        if isinstance(plan, dict):
            plan["requested"] = requested
            plan["reason"] = reason


def _removed_by_preprocessing(
    adata_in: Any, adata_out: Any, thresholds: dict[str, int], mode: str | None
) -> str | None:
    """A warning naming the cells and genes that single-cell preprocessing removed, if any."""
    n_cells = int(adata_in.n_obs) - int(adata_out.n_obs)
    n_genes = int(adata_in.n_vars) - int(adata_out.n_vars)
    if n_cells <= 0 and n_genes <= 0:
        return None
    parts = []
    if n_cells > 0:
        gone = adata_in.obs_names[~adata_in.obs_names.isin(adata_out.obs_names)]
        parts.append(
            f"{n_cells} of {adata_in.n_obs} cells (fewer than {thresholds['min_genes']} genes detected; "
            f"e.g. {[str(name) for name in gone[:5]]})"
        )
    if n_genes > 0:
        parts.append(f"{n_genes} of {adata_in.n_vars} genes (detected in fewer than {thresholds['min_cells']} cells)")
    return (
        f"Preprocessing ('{mode}') removed {' and '.join(parts)}, so the saved file holds "
        f"{adata_out.n_obs} cells and {adata_out.n_vars} genes. For data that is already "
        "preprocessed, pass preprocessing='none' to keep every cell and gene."
    )


def _keys_for_model(adata: Any, model: str) -> list[str]:
    """Embedding keys in ``adata`` that an earlier run of ``model`` wrote.

    Keys look like ``X_emb__molecule__morgan_fp__pool_mean`` or, for tables without
    pooling, ``X_emb__gene__genept``; dots in model names become underscores
    (``llama3.2_1b`` -> ``llama3_2_1b``).
    """
    names = {model, re.sub(r"[^A-Za-z0-9_]", "_", model)}
    keys = [str(k) for k in (*adata.obsm.keys(), *adata.varm.keys())]
    return [
        k for k in keys
        if any(f"__{n}__" in k or k.endswith(f"__{n}") or k == f"X_{n}" for n in names)
    ]


def _no_model_succeeded(path: str, models: list[str], failed: dict[str, str], not_updated: list[str]) -> dict[str, Any]:
    """Error response when every requested model failed; nothing is written in that case."""
    details = "; ".join(f"{model}: {error}" for model, error in failed.items())
    response: dict[str, Any] = {
        "status": "error",
        "error_type": "ModuleNotFoundError" if "No module named" in details else "EmbeddingError",
        "error": f"No model produced an embedding, so nothing was written. {details}",
        "path": str(path),
        "models": models,
        "models_computed": [],
        "models_failed": failed,
    }
    if not_updated:
        response["not_updated"] = not_updated
        response["note"] = (
            f"The input already holds {not_updated} from an earlier run; those results were not recomputed."
        )
    hint = _install_hint(details)
    if hint:
        response["suggestion"] = hint
    return response


def _embedding_workspace(adata: Any, mask: np.ndarray | None, axis: str | None) -> Any:
    """Return a container with the rows to embed and none of ``adata``'s earlier results.

    Only ``.X``, ``.obs`` and ``.var`` are carried over, so after embedding its
    ``.obsm``/``.varm``/``.uns`` hold exactly what that call computed.
    """
    import anndata

    obs_idx = mask if mask is not None and axis == "obs" else slice(None)
    var_idx = mask if mask is not None and axis == "var" else slice(None)
    view = adata[obs_idx, var_idx]
    return anndata.AnnData(X=view.X, obs=view.obs.copy(), var=view.var.copy())


def _merge_embedding_workspace(
    adata: Any, workspace: Any, mask: np.ndarray | None, axis: str | None
) -> list[tuple[str, str]]:
    """Write every result in ``workspace`` into ``adata``, overwriting earlier values.

    Rows excluded by ``mask`` (no identifier) become NaN, also when the key held
    an earlier result. Returns the ``(slot, key)`` pairs written.
    """
    import pandas as pd

    written: list[tuple[str, str]] = []
    for slot, slot_axis in (("obsm", "obs"), ("varm", "var")):
        row_mask = mask if mask is not None and axis == slot_axis else None
        for key, part in getattr(workspace, slot).items():
            part = np.asarray(part)
            if row_mask is not None:
                full = np.full((len(row_mask), part.shape[1]), np.nan, dtype=np.float32)
                full[row_mask] = part
                part = full
            getattr(adata, slot)[key] = part
            written.append((slot, str(key)))

    for key, value in workspace.uns.items():
        old = adata.uns.get(key)
        if key in _UNS_COLLECTIONS and isinstance(value, dict) and isinstance(old, dict):
            adata.uns[key] = {**old, **value}
        else:
            adata.uns[key] = value
        written.append(("uns", str(key)))

    # Alias columns (e.g. resolved ids) that embedding added to or changed in .obs/.var.
    # Every other column -- a 'dose' with gaps, a categorical label -- is left exactly
    # as it was, values and dtype, including in the rows that were not embedded.
    for frame_axis in ("obs", "var"):
        row_mask = mask if mask is not None and axis == frame_axis else None
        source, target = getattr(workspace, frame_axis), getattr(adata, frame_axis)
        for col in source.columns:
            new = source[col]
            if col in target.columns:
                old = target[col] if row_mask is None else target[col][row_mask]
                if _same_column_values(old, new):
                    continue
                if row_mask is None:
                    target[col] = pd.Series(new.array, index=target.index)
                else:
                    target[col] = _update_masked_rows(target[col], new, row_mask)
            elif row_mask is None:
                target[col] = pd.Series(new.array, index=target.index)
            else:
                target[col] = _expand_masked_column(new, row_mask, target.index)
    return written


def _is_missing(value: Any) -> bool:
    """True for None, NaN and pandas NA (never raises on list-like values)."""
    import pandas as pd

    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _same_column_values(a: Any, b: Any) -> bool:
    """Positionwise equality of two columns, with missing values equal to each other."""
    if len(a) != len(b):
        return False
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if a.equals(b):
        return True
    for x, y in zip(a.astype(object), b.astype(object), strict=True):
        if _is_missing(x) or _is_missing(y):
            if not (_is_missing(x) and _is_missing(y)):
                return False
        elif x != y:
            return False
    return True


def _update_masked_rows(column: Any, new: Any, row_mask: np.ndarray) -> Any:
    """``column`` with the rows in ``row_mask`` replaced by ``new``; other rows keep their values."""
    import pandas as pd

    out = column.copy()
    if isinstance(out.dtype, pd.CategoricalDtype):
        extra = [v for v in pd.unique(new.astype(object)) if not _is_missing(v) and v not in out.cat.categories]
        if extra:
            out = out.cat.add_categories(extra)
    positions = np.flatnonzero(row_mask)
    try:
        out.iloc[positions] = new.to_numpy()
    except (TypeError, ValueError):
        out = out.astype(object)
        out.iloc[positions] = new.astype(object).to_numpy()
    return out


def _expand_masked_column(new: Any, row_mask: np.ndarray, index: Any) -> Any:
    """A full-length column from ``new`` (the rows in ``row_mask``); other rows are missing.

    The dtype is one .h5ad can store with gaps: float with NaN for numbers, a
    nullable boolean for flags, a categorical for text and labels.
    """
    import pandas as pd

    positions = np.flatnonzero(row_mask)
    if pd.api.types.is_bool_dtype(new.dtype):
        values = pd.array([pd.NA] * len(index), dtype="boolean")
        values[positions] = new.astype("boolean").array
        return pd.Series(values, index=index)
    if pd.api.types.is_numeric_dtype(new.dtype):
        values = np.full(len(index), np.nan, dtype=float)
        values[positions] = new.to_numpy(dtype=float, na_value=np.nan)
        return pd.Series(values, index=index)
    labels: list[Any] = [None] * len(index)
    for pos, value in zip(positions, new.astype(object), strict=True):
        labels[pos] = None if _is_missing(value) else str(value)
    categories = None
    if isinstance(new.dtype, pd.CategoricalDtype):
        categories = [str(c) for c in new.cat.categories]
    return pd.Series(pd.Categorical(labels, categories=categories), index=index)


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
        Key for the new embedding (only when computing a single model). For 'single_cell',
        a key (this one or the default X_<model>) that .uns['embpy_cell_embeddings'] records
        for another model is refused with error_type 'OutputKeyConflictError' before
        anything runs; a model may reuse its own key.
    output_path : str, optional
        Filesystem path to save the updated dataset (default: '<input>_embedded.h5ad').
    preprocessing : {'auto', 'standard', 'raw', 'none'}, default='auto'
        Single-cell preprocessing strategy (ignored by other modalities). 'auto' uses an
        input that already holds .layers['log_normalized'] (preprocess_counts output) as
        is; otherwise it filters, normalizes and selects HVGs as each model needs, with
        thresholds scaled down for small datasets. 'none' never filters or transforms.
        A .X that is already normalized, with neither .layers['counts'] nor
        .layers['log_normalized'], is embedded as is by 'auto' (with a warning), and
        refused by 'standard' and 'raw' (error_type 'NotRawCountsError'). Without
        preprocessing, models that read raw counts (scvi, scanvi, totalvi) fail when
        .layers['counts'] (or .X) does not hold counts.
    device : str, default='auto'
        Compute device target ('auto', 'cpu', 'cuda', 'mps').

    Returns
    -------
    dict[str, Any]
        - status: "success" (every model produced an embedding), "partial_success"
          (some failed; see models_failed) or "error" (none did; nothing is written)
        - output_path: Saved AnnData destination path
        - entity_type: Modality embedded
        - n_obs_input / n_obs_output, n_vars_input / n_vars_output: Cells and genes in the
          input and in the saved file; single-cell preprocessing can remove some, and then
          ``warning`` says how many and why
        - preprocessing_applied: Single-cell preprocessing that ran ('standard', 'raw' or
          'none'); preprocessing_note explains when 'auto' used the input as is
        - models_computed: Canonical keys of the models that produced an embedding
        - models_failed: Model -> error, for the models that did not (only when some failed)
        - not_updated: Keys already in the file from an earlier run of a failed model;
          they were left unchanged and are not this call's results
        - model_aliases_applied: Requested names that were mapped to a canonical key
        - embedding_keys: Keys written by this call (.obsm / .varm)
        - embedding_shapes: Shapes of the written embedding matrices
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

        # Each model once, also when two requested names are aliases of one model.
        resolved_models = list(dict.fromkeys(_canonical_model_name(m) for m in model_list))
        aliases_applied = {
            str(m): _canonical_model_name(m) for m in model_list if str(m).strip() != _canonical_model_name(m)
        }
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
        n_obs_input, n_vars_input = int(adata.n_obs), int(adata.n_vars)
        before = {("obsm", k) for k in adata.obsm} | {("varm", k) for k in adata.varm} | {("uns", k) for k in adata.uns}
        succeeded: list[str] = []
        failed: dict[str, str] = {}
        not_updated: list[str] = []
        adjustments: dict[str, str] = {}
        applied_preprocessing: str | None = None
        preprocessing_note: str | None = None
        input_warning: str | None = None
        # Models that need raw counts but would get other values: model -> error.
        blocked: dict[str, str] = {}
        # embed_cells' QC defaults; preprocessing='auto' scales them for small datasets.
        qc_thresholds = {"min_genes": 200, "min_cells": 3}

        x_not_counts: dict[str, Any] | None = None
        if ent_str == "single_cell":
            # A key another model's record points at would leave two records describing
            # one matrix; refuse before anything runs.
            conflicts = _cell_key_conflicts(
                dict(adata.uns.get(_CELL_EMBEDDING_RECORDS) or {}), resolved_models, output_key
            )
            if conflicts:
                return _key_conflict_error(path, resolved_models, conflicts)
            # Normalized .X with no raw counts and no log_normalized layer (e.g. a file
            # normalized with scanpy): 'standard' and 'raw' would treat .X as counts.
            if "counts" not in adata.layers and "log_normalized" not in adata.layers:
                nature = _detect_expression_nature(adata)
                if nature["detected_nature"] in _NOT_RAW_COUNTS:
                    x_not_counts = nature
            if x_not_counts and preprocessing in ("standard", "raw"):
                return _not_raw_counts_error(
                    path,
                    x_not_counts,
                    None,
                    needed_by=f"compute_embeddings(preprocessing='{preprocessing}')",
                    consequence=(
                        "Normalizing and log-transforming them again would distort the data."
                        if preprocessing == "standard"
                        else "It would store them in .layers['counts'] as if they were counts."
                    ),
                    remedy=(
                        "The data is already normalized: preprocessing='auto' or 'none' embeds .X as it is. "
                        "If these values are counts (e.g. ambient-corrected, non-integer counts), run "
                        "preprocess_counts(assume_counts=True) first."
                    ),
                )

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
                run_preprocessing = preprocessing
                if preprocessing == "auto" and "log_normalized" in adata.layers:
                    # preprocess_counts output: cells and genes are filtered, the data is
                    # normalized and the HVGs are chosen. Doing it again would drop cells and
                    # genes that step kept and replace its HVGs, so use the data as it is.
                    run_preprocessing = "none"
                    preprocessing_note = (
                        "The input already holds normalized expression (.layers['log_normalized'], "
                        "as written by preprocess_counts), so it was used as is: no cells or genes "
                        "were filtered and the highly variable genes in .var were not selected again."
                    )
                elif preprocessing == "auto" and x_not_counts:
                    # Already normalized (e.g. with scanpy), without raw counts: embed .X as
                    # the normalized expression instead of normalizing it a second time.
                    run_preprocessing = "none"
                    kind = x_not_counts["detected_nature"]
                    preprocessing_note = (
                        f"The input .X already holds {kind} values and there is no .layers['counts'], so .X "
                        "was used as the normalized expression: no cells or genes were filtered and nothing "
                        "was normalized again."
                    )
                    input_warning = (
                        f"The input .X holds {kind} values (not raw counts) and there is no .layers['counts'], "
                        "so preprocessing='auto' embedded .X as the normalized expression; it was not "
                        "normalized again."
                    )
                elif preprocessing == "auto":
                    # embed_cells' defaults (min_genes=200, 2000 HVGs) would filter out
                    # every cell of a small dataset; scale them as preprocess_counts does.
                    min_g, min_c, n_hvg, adjustments = _small_dataset_thresholds(adata.n_obs, adata.n_vars, 200, 3, 2000)
                    extra_kwargs.update(min_genes=min_g, min_cells=min_c, n_top_genes=n_hvg)
                    qc_thresholds = {"min_genes": min_g, "min_cells": min_c}
                    if adata.n_vars < 2000:
                        extra_kwargs["pca_use_hvg"] = False

                # Without preprocessing, models that read raw counts (scVI family) get the
                # file's values as they are; normalized values are refused per model.
                if run_preprocessing == "none":
                    blocked = _raw_count_model_problems(adata, resolved_models)
                runnable = [m for m in resolved_models if m not in blocked]

                # embed_cells records a failed model as {"error": ...} and carries on.
                # New matrices go to temporary keys and move to X_<model> (or
                # output_key) only for the models that succeeded, so a failure
                # never leaves an earlier result looking freshly computed. The
                # prefix is unique to this call, so the cleanup cannot touch other keys.
                earlier_records = dict(adata.uns.get(_CELL_EMBEDDING_RECORDS) or {})
                if runnable:
                    run_prefix = f"{_NEW_CELL_EMBEDDING_PREFIX}{uuid.uuid4().hex[:12]}_"
                    adata_out = embedder.embed_cells(
                        adata,
                        models=runnable,
                        batch_size=batch_size,
                        preprocessing=run_preprocessing,
                        obsm_prefix=run_prefix,
                        **extra_kwargs,
                    )
                    plan = (adata_out.uns.get(_CELL_EMBEDDING_RECORDS) or {}).get("__preprocessing__") or {}
                    applied_preprocessing = str(plan.get("resolved", run_preprocessing))
                    if preprocessing_note:
                        _note_preprocessing_choice(adata_out, requested=preprocessing, reason=preprocessing_note)
                    # The blocked models have no run record, so they are settled as failed.
                    written, succeeded, failed, not_updated = _settle_cell_embeddings(
                        adata_out, earlier_records, resolved_models, output_key, prefix=run_prefix
                    )
                else:
                    adata_out = adata
                    applied_preprocessing = run_preprocessing
                    keys = dict.fromkeys(output_key or f"X_{m}" for m in blocked)
                    not_updated = [key for key in keys if key in adata.obsm]
                failed.update(blocked)
            else:
                # Rows with no identifier in the chosen column (e.g. control wells after
                # name -> SMILES resolution) cannot be embedded; embpy rejects them.
                column, axis = (obs_column, "obs") if obs_column else (var_column, "var") if var_column else (None, None)
                has_id = None
                if column is not None:
                    values = getattr(adata, axis)[column]
                    has_id = (values.notna() & (values.astype(str).str.strip() != "")).to_numpy()
                    if not has_id.any():
                        return {
                            "status": "error",
                            "error_type": "ValueError",
                            "error": f"Column '{column}' holds no identifiers to embed.",
                            "path": str(path),
                        }
                    if not has_id.all() and missing == "error":
                        return {
                            "status": "error",
                            "error_type": "ValueError",
                            "error": (
                                f"{int((~has_id).sum())} rows have no value in '{column}' (e.g. controls). "
                                "Pass missing='nan' to embed the other rows and leave these as NaN."
                            ),
                            "path": str(path),
                        }
                    if has_id.all():
                        has_id = None

                # Embed each model into a fresh container that holds only the rows to
                # embed and no earlier embeddings, so every matrix in it afterwards was
                # computed by this call -- including keys that already existed in the
                # file. One model at a time, so a failing model does not discard the
                # others' results.
                written = []
                for model in resolved_models:
                    workspace = _embedding_workspace(adata, has_id, axis)
                    try:
                        embedder.embed(
                            workspace,
                            entity_type=route["backend"],
                            model=model,
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
                    except Exception as exc:  # noqa: BLE001 -- reported per model below
                        if len(resolved_models) == 1:
                            raise
                        logger.warning("compute_embeddings: model %s failed on %s: %s", model, path, exc)
                        failed[model] = f"{type(exc).__name__}: {exc}"
                        not_updated += _keys_for_model(adata, model)
                        continue
                    # Copy new and recomputed results back; rows without an id become NaN.
                    written += _merge_embedding_workspace(adata, workspace, has_id, axis)
                    succeeded.append(model)
                adata_out = adata

        if not succeeded:
            response = _no_model_succeeded(path, resolved_models, failed, not_updated)
            if blocked and set(failed) <= set(blocked):
                response["error_type"] = "NotRawCountsError"
            return response

        after = {("obsm", k) for k in adata_out.obsm} | {("varm", k) for k in adata_out.varm} | {("uns", k) for k in adata_out.uns}
        added = sorted((after - before) | set(written))

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

        response: dict[str, Any] = {
            "status": "partial_success" if failed else "success",
            "input_path": str(resolved_path),
            "output_path": saved_path,
            "entity_type": ent_str,
            "n_cells": int(adata_out.n_obs),
            "n_obs_input": n_obs_input,
            "n_obs_output": int(adata_out.n_obs),
            "n_vars_input": n_vars_input,
            "n_vars_output": int(adata_out.n_vars),
            "models_computed": succeeded,
            "model_aliases_applied": aliases_applied,
            "preprocessing_adjustments": adjustments,
            "embedding_keys": [str(key) for _, key in matrix_keys],
            "embedding_slots": {str(key): slot for slot, key in matrix_keys},
            "embedding_shapes": shapes,
            "metadata_keys": [str(key) for slot, key in added if slot == "uns"],
            "execution_time_seconds": round(time.time() - t0, 3),
        }
        if applied_preprocessing is not None:
            response["preprocessing_applied"] = applied_preprocessing
        if preprocessing_note:
            response["preprocessing_note"] = preprocessing_note
        warnings: list[str] = []
        if input_warning:
            if blocked:
                input_warning += f" Models that need raw counts were not run: {list(blocked)}."
            warnings.append(input_warning)
        if failed:
            response["models_failed"] = failed
            warning = (
                f"{len(failed)} of {len(resolved_models)} models failed ({', '.join(failed)}); "
                f"only {succeeded} were computed."
            )
            if not_updated:
                response["not_updated"] = not_updated
                warning += f" {not_updated} in the file come from an earlier run and were not recomputed."
            warnings.append(warning)
            hint = _install_hint(" ".join(failed.values()))
            if hint:
                response["suggestion"] = hint
        removed = _removed_by_preprocessing(adata, adata_out, qc_thresholds, applied_preprocessing)
        if removed:
            warnings.append(removed)
        if warnings:
            response["warning"] = " ".join(warnings)
        return response

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
            resp["suggestion"] = _install_hint(msg) or (
                "Install the model's optional dependencies (see list_model_registry optional_extras_status)."
            )
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
# Reactome species names for the organisms the tools accept.
_REACTOME_SPECIES = {"human": "Homo sapiens", "mouse": "Mus musculus"}
# Host fragment -> service name, for "Couldn't reach <service>." messages.
_SERVICE_NAMES = (
    ("reactome.org", "Reactome"),
    ("mygene.info", "MyGene.info"),
    ("uniprot.org", "UniProt"),
    ("ebi.ac.uk/interpro", "InterPro"),
    ("ebi.ac.uk/chembl", "ChEMBL"),
    ("ebi.ac.uk/gwas", "the GWAS Catalog"),
    ("opentargets", "Open Targets"),
    ("string-db.org", "STRING"),
    ("gtexportal.org", "GTEx"),
    ("proteinatlas.org", "the Human Protein Atlas"),
    ("pubchem", "PubChem"),
    ("kegg.jp", "KEGG"),
    ("wikipathways", "WikiPathways"),
)


def _service_name(url: str) -> str:
    """Name of the web service a failed request went to (its host if not a known one)."""
    for fragment, name in _SERVICE_NAMES:
        if fragment in url:
            return name
    host = re.sub(r"^[a-z]+://", "", url).split("/")[0]
    return host or "the annotation service"
# Served only for human genes (GTEx, HPA, GWAS Catalog, Open Targets).
_HUMAN_ONLY_ANNOTATIONS = {("gene", "diseases"), ("gene", "expression")}
_UNIPROT_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$"
)


# Fields that describe where or how a result was obtained (e.g. HPA's
# {"locations": [], "reliability": "Approved", "source": "HPA"}); they are not results.
_ANNOTATION_METADATA_KEYS = frozenset(
    {"source", "sources", "database", "reliability", "query", "organism", "taxonomy_id", "version", "url",
     "last_modified", "retrieved_at"}
)


_PREVIEW_ENTITIES = 10
_PREVIEW_FIELDS = (
    "id", "name", "term", "label", "symbol", "description", "aspect", "disease", "tissue",
    "location", "partner", "target", "pathway", "value", "score",
)


def _join_names(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def _preview_records(data: Any, limit: int) -> Any:
    """A short, JSON-safe view of annotation records: per list its length and first entries."""
    if isinstance(data, dict):
        return {str(k): _preview_records(v, limit) for k, v in data.items() if k not in _ANNOTATION_METADATA_KEYS}
    if isinstance(data, (list, tuple)):
        items = []
        for record in list(data)[:limit]:
            if isinstance(record, dict):
                compact = {k: record[k] for k in _PREVIEW_FIELDS if k in record}
                items.append(_sanitize_val(compact or dict(list(record.items())[:3])))
            else:
                items.append(_sanitize_val(record))
        return {"n": len(data), "items": items}
    return _sanitize_val(data)


def _n_annotation_items(value: Any) -> int:
    """Count annotation records: list entries, summed through nested dicts.

    Metadata fields (``_ANNOTATION_METADATA_KEYS``) are skipped, so an answer that
    carries only provenance counts as zero records.
    """
    if isinstance(value, dict):
        return sum(_n_annotation_items(v) for k, v in value.items() if k not in _ANNOTATION_METADATA_KEYS)
    if isinstance(value, (list, tuple)):
        return len(value)
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return 0
    return 1


# .uns entry holding the annotation records annotate_metadata stored in a file.
_ANNOTATIONS_KEY = "embpy_annotations"


def _json_field(record: Any, key: str) -> dict[str, Any]:
    """The JSON object an earlier call stored under ``record[key]``; {} if there is none."""
    text = record.get(key) if isinstance(record, dict) else None
    if not isinstance(text, str):
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        logger.warning("Ignoring an unreadable .uns['%s']['%s'] entry", _ANNOTATIONS_KEY, key)
        return {}
    return value if isinstance(value, dict) else {}


# Layout of the stored records (results_json):
# identifier -> entity type -> organism -> annotation type -> source -> records.
# Files written before carry no "version" and keep identifier -> type -> source.
_RECORDS_VERSION = 2
# Molecule annotations do not depend on the organism; their records are stored under this key.
_ANY_ORGANISM = "any"
_STORED_ENTITY_TYPES = ("gene", "protein", "molecule")
# Prefix of the .obs count columns <prefix>_<type>_n per entity type.
_COUNT_PREFIX = {"gene": "gene", "protein": "prot", "molecule": "mol"}


def _records_organism(entity_type: str, organism: str) -> str:
    """The organism key a call's records are stored under."""
    return _ANY_ORGANISM if entity_type == "molecule" else organism


def _child(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """``parent[key]``, made an empty dict first unless it is one."""
    value = parent.get(key)
    if not isinstance(value, dict):
        value = parent[key] = {}
    return value


def _scoped_records(records: dict[str, Any], entity: str, entity_type: str, organism: str) -> dict[str, Any]:
    """The records stored for ``entity`` as ``entity_type`` in ``organism``: type -> source -> records."""
    value: Any = records.get(entity)
    for key in (entity_type, organism):
        value = value.get(key) if isinstance(value, dict) else None
    return value if isinstance(value, dict) else {}


def _scalar_properties(records: dict[str, Any], entity: str) -> dict[str, Any]:
    """The stored RDKit properties of molecule ``entity`` with a single value (no lists or dicts)."""
    value: Any = _scoped_records(records, entity, "molecule", _ANY_ORGANISM).get("properties")
    for key in ("rdkit", "physicochemical"):
        value = value.get(key) if isinstance(value, dict) else None
    return {k: v for k, v in value.items() if not isinstance(v, (dict, list))} if isinstance(value, dict) else {}


def _load_annotation_store(
    record: Any, obs_columns: Any
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """The records, provenance sources and column registries stored in ``.uns["embpy_annotations"]``.

    Returns ``(records, sources, count_columns, property_columns)``. Files written before
    records were kept per entity type and organism hold identifier -> type -> source: each
    type and source goes to the entity type and organism their provenance names; without
    provenance, to the entity type whose sources include it, preferring the one of the call
    that wrote the record (whose organism and column then apply).

    Those versions named count and property columns after the type alone
    (``<prefix>_<type>_n``, ``mol_<property>``), and the first did not record them. Such a
    column in ``obs_columns`` that no registry lists is taken over for the identifier column
    its type's records came from.
    """
    results = _json_field(record, "results_json")
    provenance = _json_field(record, "provenance_json")
    count_columns = provenance.get("count_columns")
    count_columns = dict(count_columns) if isinstance(count_columns, dict) else {}
    property_columns = provenance.get("property_columns")
    property_columns = dict(property_columns) if isinstance(property_columns, dict) else {}
    meta = record if isinstance(record, dict) else {}
    version = meta.get("version")
    if isinstance(version, (int, np.integer)) and int(version) == _RECORDS_VERSION:
        records = {str(e): v for e, v in results.items() if isinstance(v, dict)}
        sources = [dict(s) for s in provenance.get("sources") or () if isinstance(s, dict)]
        return records, sources, count_columns, property_columns

    entity_type = str(meta.get("entity_type") or "gene")
    organism = str(meta.get("organism") or "human")
    column = str(meta.get("column") or "")
    # (type, source) -> [(entity type, organism, column)] named by the earlier provenance.
    owners: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for et in _STORED_ENTITY_TYPES:
        by_type = provenance.get(et)
        for t, by_source in (by_type.items() if isinstance(by_type, dict) else ()):
            for s, info in (by_source.items() if isinstance(by_source, dict) else ()):
                info = info if isinstance(info, dict) else {}
                owners.setdefault((str(t), str(s)), []).append((
                    et, _records_organism(et, str(info.get("organism") or organism)), str(info.get("column") or column)
                ))

    def owner_of(ann_type: str, source: str) -> tuple[str, str, str]:
        found = owners.get((ann_type, source))
        if found:
            return next((o for o in found if o[0] == entity_type), found[0])
        serving = [et for et in _STORED_ENTITY_TYPES if source in _ANNOTATION_SOURCES[et].get(ann_type, ())]
        if not serving or entity_type in serving:
            return entity_type, _records_organism(entity_type, organism), column
        # Another entity type's records, kept by a later call: their organism and column are unknown.
        return serving[0], _records_organism(serving[0], "human"), ""

    records: dict[str, Any] = {}
    sources: list[dict[str, str]] = []
    for e, by_type in results.items():
        for t, by_source in (by_type.items() if isinstance(by_type, dict) else ()):
            for s, data in (by_source.items() if isinstance(by_source, dict) else ()):
                et, org, col = owner_of(str(t), str(s))
                _child(_child(_child(records, str(e)), et), org).setdefault(str(t), {})[str(s)] = data
                entry = {"entity_type": et, "organism": org, "column": col, "annotation_type": str(t), "source": str(s)}
                if entry not in sources:
                    sources.append(entry)
    for info in count_columns.values():
        if isinstance(info, dict) and "organism" not in info:
            organisms = {
                s["organism"] for s in sources
                if (s["entity_type"], s["annotation_type"], s["column"])
                == (info.get("entity_type"), info.get("annotation_type"), info.get("column"))
            }
            info["organism"] = (
                organisms.pop() if len(organisms) == 1 else _records_organism(str(info.get("entity_type")), organism)
            )

    # Take over the columns those versions wrote: (entity type, type) -> {(column, organism)}.
    columns_of: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for s in sources:
        if s["column"]:
            columns_of.setdefault((s["entity_type"], s["annotation_type"]), set()).add((s["column"], s["organism"]))
    types = meta.get("annotation_types", [])
    if entity_type in _STORED_ENTITY_TYPES and column:  # also types whose lookups stored no record
        for t in [types] if isinstance(types, str) else [str(t) for t in types]:
            columns_of.setdefault((entity_type, t), {(column, _records_organism(entity_type, organism))})
    for (et, t), found in columns_of.items():
        if len(found) > 1:  # several columns: the column of the call that wrote the record, if one of them
            found = {f for f in found if f[0] == column}
        if len(found) != 1:
            continue
        ((col, org),) = found
        name = f"{_COUNT_PREFIX[et]}_{t}_n"
        if name in obs_columns and name not in count_columns:
            count_columns[name] = {"column": col, "entity_type": et, "organism": org, "annotation_type": t}
        if (et, t) == ("molecule", "properties"):
            for key in dict.fromkeys(k for e in records for k in _scalar_properties(records, e)):
                name = f"mol_{key}"
                if name in obs_columns and name not in property_columns and name not in count_columns:
                    property_columns[name] = {"column": col, "property": key}
    return records, sources, count_columns, property_columns


def _merge_annotation_records(
    earlier: dict[str, Any],
    results: dict[str, dict[str, Any]],
    sources_queried: dict[str, list[str]],
    failed: set[tuple[str, str]],
    entity_type: str,
    organism: str,
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """Merge this call's records into those stored by earlier calls.

    Only the records of this call's entity type and organism are touched (identifier ->
    ``entity_type`` -> ``organism`` -> type -> source). Among them, a source this call
    looked up replaces its earlier records for each entity looked up, also with an empty
    answer. When the request for an entity and type failed, a source that returned
    nothing keeps its earlier records. Other types, sources and entities stay as they were.

    Returns the merged records and the (type, source) pairs this call wrote.
    """
    merged = earlier
    updated: set[tuple[str, str]] = set()
    for entity, by_type in results.items():
        stored = _child(_child(_child(merged, entity), entity_type), organism)
        for ann_type, data in by_type.items():
            slot = dict(stored[ann_type]) if isinstance(stored.get(ann_type), dict) else {}
            new = data if isinstance(data, dict) else {}
            for source in sources_queried.get(ann_type, ()):
                if (entity, ann_type) in failed and not _n_annotation_items(new.get(source)):
                    continue  # the request failed: keep what an earlier call found
                if source in new:
                    slot[source] = new[source]
                else:
                    slot.pop(source, None)
                updated.add((ann_type, source))
            stored[ann_type] = slot
    return merged, updated


def _owned_by(registry: dict[str, Any], name: str, column: str) -> bool:
    """Whether the registry lists ``.obs[name]`` as written for identifier column ``column``."""
    info = registry.get(name)
    return isinstance(info, dict) and info.get("column") == column


# Names a KEGG reference folder (written by kegg_index.build_kegg_reference).
_KEGG_REFERENCE_ENV = "EMBPY_KEGG_REFERENCE"


def _load_kegg_reference() -> tuple[Any, dict[str, str] | None]:
    """The KEGG reference for molecule pathways, or ``(None, {"error": ..., "suggestion": ...})``.

    The folder is ``$EMBPY_KEGG_REFERENCE`` if set, otherwise ``$EMBPY_CACHE/kegg``
    (default ``~/.cache/embpy/kegg``). It is never built here: KEGG's REST API is for
    academic use only, so building the reference is left to the user.
    """
    from embpy.resources.molecule.kegg_index import INCHIKEY_FILE, KeggIndex, KeggReferenceError

    configured = os.environ.get(_KEGG_REFERENCE_ENV)
    if configured:
        folder = os.path.abspath(os.path.expanduser(configured))
    else:
        cache = os.environ.get("EMBPY_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "embpy")
        folder = os.path.abspath(os.path.join(cache, "kegg"))
    build = (
        "Build the KEGG reference once (KEGG's REST API is for academic use only): python -c "
        f"\"from embpy.resources.molecule.kegg_index import build_kegg_reference; build_kegg_reference(r'{folder}')\""
    )
    if not configured and not os.path.isfile(os.path.join(folder, INCHIKEY_FILE)):
        return None, {
            "error": (
                "The KEGG search could not run: KEGG pathways of molecules are looked up in a local KEGG "
                f"reference, and there is none ({_KEGG_REFERENCE_ENV} is not set and {folder} holds none)."
            ),
            "suggestion": (
                f"{build}. Or set {_KEGG_REFERENCE_ENV} in the MCP server's environment to a folder written "
                "by build_kegg_reference."
            ),
        }
    try:
        return KeggIndex.from_dir(folder), None
    except (OSError, ValueError, KeggReferenceError) as exc:
        return None, {
            "error": f"The KEGG search could not run: the KEGG reference in {folder} could not be loaded ({exc}).",
            "suggestion": f"{build}. Or point {_KEGG_REFERENCE_ENV} to a folder written by build_kegg_reference.",
        }


# NCBI taxonomy id and species of the organisms the tools accept, as UniProt entries name them.
_ORGANISM_TAXA = {"human": (9606, "Homo sapiens"), "mouse": (10090, "Mus musculus")}


class _SpeciesMismatch(Exception):
    """The UniProt entry an identifier led to belongs to another species than the one asked for."""

    def __init__(self, accession: str, species: str, taxonomy_id: int) -> None:
        super().__init__(f"{accession} is a {species} entry (taxon {taxonomy_id})")
        self.accession, self.species, self.taxonomy_id = accession, species, taxonomy_id


def _uniprot_annotator(organism: str) -> Any:
    """A ProteinAnnotator that keeps the organism of every UniProt entry it fetches.

    Symbols and Ensembl ids are resolved in ``organism``, but an accession names one
    entry of whatever species, so :func:`_check_uniprot_species` checks the entry.
    """
    from embpy.resources.protein.annotator import ProteinAnnotator

    class _SpeciesRecordingAnnotator(ProteinAnnotator):
        def _fetch_uniprot_entry(self, accession: str) -> dict | None:
            entry = super()._fetch_uniprot_entry(accession)
            if isinstance(entry, dict) and isinstance(entry.get("organism"), dict):
                self.entry_organisms[accession] = entry["organism"]
            return entry

    annotator = _SpeciesRecordingAnnotator(organism=organism)
    annotator.entry_organisms = {}
    return annotator


def _check_uniprot_species(annotator: Any, result: dict[str, Any], organism: str) -> None:
    """Raise :class:`_SpeciesMismatch` when the entry behind ``result`` is not of ``organism``."""
    found = getattr(annotator, "entry_organisms", {}).get(result.get("uniprot_accession"))
    if not found or found.get("taxonId") is None:
        return  # no entry was fetched, so there is nothing to label
    taxonomy_id = int(found["taxonId"])
    if taxonomy_id != _ORGANISM_TAXA[organism][0]:
        species = str(found.get("scientificName") or f"taxon {taxonomy_id}")
        raise _SpeciesMismatch(str(result["uniprot_accession"]), species, taxonomy_id)


def _species_mismatch_text(mismatches: list[dict[str, Any]], organism: str) -> str:
    """'Species mismatch: the UniProt entry of P04637 (Homo sapiens, taxon 9606) is not mouse ...'."""
    taxonomy_id, species = _ORGANISM_TAXA[organism]
    named = []
    for m in mismatches[:5]:
        accession = "" if m["entity"] == m["uniprot_accession"] else f"{m['uniprot_accession']}: "
        named.append(f"{m['entity']} ({accession}{m['uniprot_species']}, taxon {m['uniprot_taxonomy_id']})")
    if len(mismatches) > 5:
        named.append(f"{len(mismatches) - 5} more")
    one = len(mismatches) == 1
    return (
        f"Species mismatch: the UniProt entr{'y' if one else 'ies'} of {_join_names(named)} "
        f"{'is' if one else 'are'} not {organism} ({species}, taxon {taxonomy_id}), so "
        f"{'it was' if one else 'they were'} not annotated."
    )


def _protein_id_type(identifier: str, id_type: str) -> str:
    if id_type != "auto":
        return id_type
    if _ENSEMBL_ID_RE.match(identifier):
        return "ensembl_id"
    if _UNIPROT_ACCESSION_RE.match(identifier):
        return "uniprot_id"
    return "symbol"


def _annotation_fetchers(entity: str, organism: str, id_type: str, kegg_index: Any = None) -> dict[str, Any]:
    """Return {annotation_type: fn(identifier, sources) -> {source: data}} for ``entity``.

    ``kegg_index`` (a KeggIndex) is what molecule KEGG pathways are looked up in.
    """
    if entity == "gene":
        from embpy.resources.gene.annotator import GeneAnnotator

        ga = GeneAnnotator(organism=organism)
        pa = _uniprot_annotator(organism)

        def _uniprot(gene: str, section: str) -> dict[str, Any]:
            res = pa.annotate(gene, id_type=_protein_id_type(gene, id_type), sources=[section])
            _check_uniprot_species(pa, res, organism)
            return res

        def go(gene, srcs):
            return {"uniprot": _uniprot(gene, "go").get("go_terms", {})}

        def domains(gene, srcs):
            res = _uniprot(gene, "domains")
            out = {"uniprot": res.get("uniprot_domains", []), "interpro": res.get("interpro_domains", [])}
            return {k: v for k, v in out.items() if k in srcs}

        reactome: dict[str, Any] = {}

        def _reactome_index():
            """Reactome's release index, downloaded and cached on first use (once per server)."""
            if "index" not in reactome:
                from embpy.resources.gene.reactome import ReactomeUnavailableError, load_reactome_index

                try:
                    reactome["index"] = load_reactome_index(species=_REACTOME_SPECIES[organism])
                except (ReactomeUnavailableError, OSError) as exc:
                    reactome["index"] = None
                    reactome["error"] = exc
            return reactome["index"]

        def pathways(gene, srcs):
            index = None
            if "reactome" in srcs:
                index = _reactome_index()
                if index is None:
                    from embpy.resources._service_errors import report_service_error
                    from embpy.resources.gene.reactome import REACTOME_FILES

                    report_service_error(REACTOME_FILES["NCBI2Reactome.txt"], reactome["error"])
            return {k: v for k, v in ga.get_pathways(gene, reactome_index=index).items() if k in srcs}

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
        pa = _uniprot_annotator(organism)
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
                _check_uniprot_species(pa, res, organism)
                return {src: res.get(key) for _, key, src in sections[ann_type] if src in srcs and key in res}

            return fetch

        return {t: _make(t) for t in sections}

    if entity == "molecule":
        from embpy.resources.molecule.annotator import MoleculeAnnotator

        ma = MoleculeAnnotator(kegg_index=kegg_index)
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
    path: str | None = None,
    entity_type: Literal["auto", "gene", "protein", "molecule", "cell_line"] = "auto",
    annotation_types: list[AnnotationType] | AnnotationType | None = None,
    annotation_type: AnnotationType | None = None,
    column: str | None = None,
    obs_column: str | None = None,
    sources: list[str] | str | None = None,
    id_type: Literal["auto", "symbol", "ensembl_id", "uniprot_id"] = "auto",
    output_path: str | None = None,
    organism: Literal["human", "mouse"] = "human",
    identifiers: list[str] | str | None = None,
) -> dict[str, Any]:
    """Annotate genes, proteins, molecules or cell lines with external biological knowledge.

    The entities come from an adata.obs column (``path`` + ``column``) or directly from
    ``identifiers``, e.g. ``annotate_metadata(identifiers=["TP53"], entity_type="gene",
    annotation_types="pathways", sources=["reactome"])`` answers "which human Reactome
    pathways include TP53?" without any file. Reactome's release files are downloaded
    on first use and cached (``$EMBPY_CACHE/reactome``, default ``~/.cache/embpy``).

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

    Molecule KEGG pathways are looked up in a local KEGG reference ($EMBPY_KEGG_REFERENCE,
    else $EMBPY_CACHE/kegg), built once with kegg_index.build_kegg_reference; without one
    the reply is error_type "KeggReferenceUnavailable" ("The KEGG search could not run").

    Token-Saving: Writes the enriched dataset to disk and returns per-type counts plus a
    short preview of the records (ids and names, first entries per source).

    Parameters
    ----------
    path : str, optional
        Filesystem path to the target .h5ad dataset. Give either path or identifiers.
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
        Filesystem destination (default: '<input>_annotated.h5ad'; with identifiers,
        nothing is written unless output_path is given). Not written when nothing was
        annotated. Records stored in the input by earlier calls are kept, per entity
        type and organism: a type and source looked up again replace only their records
        of this entity type and organism, for the entities looked up.
    organism : {'human', 'mouse'}, default='human'
        Species for gene and protein annotations. A UniProt entry of another species
        (e.g. an accession of a human protein with organism='mouse') is not annotated.
    identifiers : list[str] or str, optional
        Entities to annotate directly (e.g. ['TP53', 'EGFR']), instead of an .h5ad column.
        Set entity_type explicitly ('auto' assumes genes).

    Returns
    -------
    dict[str, Any]
        - status: "success" (every entity has every requested type), "partial_success",
          or "error" (nothing annotated; error_type is "AnnotationServiceError" when a
          service could not be reached -- the error then starts "Couldn't reach <service>." --
          "NoAnnotationsFound" when the lookups completed and matched nothing, e.g.
          "No pathways found ...", and "SpeciesMismatch" when the UniProt entries belong
          to another species than organism)
        - per_type: For each type, counts of entities annotated / no_match / service_error,
          and species_mismatch when a UniProt entry was of another species
        - species_mismatches: those entities, with the entry's and the requested species
        - records_preview: entity -> type -> source -> {"n": total, "items": first records}
        - unreachable_services: type -> services that failed, also when other sources answered
        - kegg_reference: folder and release of the KEGG reference used (molecule pathways)
        - sources_queried: Databases queried per type
        - successful_count: Entities annotated for every requested type
        - failed_count: Entities missing at least one requested type
        - new_columns_added: '<entity>_<type>_n' record counts per row in .obs (empty when
          nothing was written, e.g. identifiers without output_path); '<entity>_<type>_n_<column>'
          when that name already holds the counts of another identifier column. Molecule
          properties: 'mol_<property>', or 'mol_<property>_<column>' by the same rule
        - summary: Human-readable description
    """
    resolved_entity = str(entity_type)  # the entity type resolved below; for the install hint
    try:
        import anndata
        import pandas as pd

        from embpy.resources._service_errors import record_service_errors

        resolved_path: str | None = None
        if identifiers is not None:
            ids = [identifiers] if isinstance(identifiers, str) else [str(i) for i in identifiers]
            adata = anndata.AnnData(obs=pd.DataFrame({"identifier": ids}, index=[f"id_{i}" for i in range(len(ids))]))
            column, obs_column = "identifier", None
        elif not path:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": "Pass path (an .h5ad with an identifier column) or identifiers (a list of names or IDs).",
            }
        else:
            resolved_path = os.path.abspath(path)
            if not os.path.exists(resolved_path):
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "error": f"Dataset file not found: {path}",
                }
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
        if resolved_entity == "molecule":
            # Every molecule lookup starts from RDKit. MoleculeAnnotator carries on without it
            # (sending SMILES to PubChem's name search, skipping properties), which would be
            # reported as "nothing found"; raise so the reply names the package instead.
            import rdkit  # noqa: F401

        # KEGG pathways of molecules are looked up in a local KEGG reference; without one
        # the search cannot run, which is not the same as finding no pathways.
        kegg_index = None
        kegg_reference: dict[str, Any] | None = None
        if resolved_entity == "molecule" and "kegg" in sources_queried.get("pathways", ()):
            kegg_index, kegg_problem = _load_kegg_reference()
            if kegg_index is None:
                return {"status": "error", "error_type": "KeggReferenceUnavailable", **kegg_problem}
            kegg_reference = {"path": kegg_index.source_dir, "release": kegg_index.release}

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

        # outcome[type][entity] = "annotated" | "no_match" | "service_error" | "species_mismatch"
        outcome: dict[str, dict[str, str]] = {t: {} for t in requested}
        results: dict[str, dict[str, Any]] = {e: {} for e in entities}
        service_errors: list[dict[str, str]] = []
        # unreachable[type] = names of the services whose requests failed for that type
        unreachable: dict[str, set[str]] = {}
        # (entity, type) pairs without a usable answer -- a failed request (also when another
        # source answered) or a UniProt entry of another species: earlier records stay.
        failed_lookups: set[tuple[str, str]] = set()
        # entity -> its UniProt entry's species, when that is not the requested organism
        mismatched: dict[str, dict[str, Any]] = {}
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
                    outcome["cell_line"][e] = "annotated" if n else "no_match"
                    if n:  # the annotated fields, for records_preview (and identifiers mode)
                        first = rows.iloc[0]
                        results[e]["cell_line"] = {
                            str(c): first[c] for c in added if not _is_missing(first[c])
                        }
                new_cols.extend(str(c) for c in added)
            else:
                fetchers = _annotation_fetchers(resolved_entity, org, id_type, kegg_index=kegg_index)
                for e in entities:
                    for t in requested:
                        mismatch = None
                        with record_service_errors() as errs:
                            try:
                                data = fetchers[t](e, sources_queried[t])
                            except ImportError:
                                raise  # a missing package, not an unreachable service
                            except _SpeciesMismatch as exc:
                                data, mismatch = {}, exc
                            except Exception as exc:  # noqa: BLE001
                                data = {}
                                errs.append({"url": "", "error": f"{type(exc).__name__}: {exc}"})
                        n = _n_annotation_items(data)
                        results[e][t] = data
                        # Recorded even when another source answered, so a partial outage is reported.
                        unreachable.setdefault(t, set()).update(_service_name(err.get("url") or "") for err in errs)
                        if errs or mismatch is not None:
                            failed_lookups.add((e, t))
                        if mismatch is not None:
                            # Annotations of another species' entry are not labeled with this organism.
                            outcome[t][e] = "species_mismatch"
                            mismatched.setdefault(e, {
                                "entity": e,
                                "uniprot_accession": mismatch.accession,
                                "uniprot_species": mismatch.species,
                                "uniprot_taxonomy_id": mismatch.taxonomy_id,
                                "requested_species": _ORGANISM_TAXA[org][1],
                                "requested_taxonomy_id": _ORGANISM_TAXA[org][0],
                            })
                        elif n:
                            outcome[t][e] = "annotated"
                        elif errs:
                            outcome[t][e] = "service_error"
                            service_errors.append({
                                "entity": e, "type": t,
                                "service": _service_name(errs[0].get("url") or ""), "error": errs[0]["error"],
                            })
                        else:
                            outcome[t][e] = "no_match"

        per_type = {
            t: {k: sum(1 for v in outcome[t].values() if v == k) for k in ("annotated", "no_match", "service_error")}
            for t in requested
        }
        for t in requested:  # reported only when it happened
            n_mismatched = sum(1 for v in outcome[t].values() if v == "species_mismatch")
            if n_mismatched:
                per_type[t]["species_mismatch"] = n_mismatched
        fully = [e for e in entities if all(outcome[t].get(e) == "annotated" for t in requested)]
        any_annotated = [e for e in entities if any(outcome[t].get(e) == "annotated" for t in requested)]
        n_annotated_pairs = sum(p["annotated"] for p in per_type.values())

        type_label = {"go": "GO terms", "cell_line": "cell-line metadata"}
        unreachable = {t: names for t, names in unreachable.items() if names}
        couldnt_reach = {t: f"Couldn't reach {_join_names(sorted(names))}." for t, names in unreachable.items()}
        where = "the given identifiers" if resolved_path is None else f"column '{query_col}'"
        n_label = f"{len(entities)} {resolved_entity}{'' if len(entities) == 1 else 's'}"
        summary_parts = []
        for t in requested:
            p = per_type[t]
            if p["annotated"] == 0 and p["service_error"] == 0 and not p.get("species_mismatch"):
                part = f"No {type_label.get(t, t)} found ({n_label} looked up)"
            else:
                part = f"{type_label.get(t, t)}: {p['annotated']}/{len(entities)} {resolved_entity} entities annotated"
                extras = [
                    f"{p[k]} {k.replace('_', ' ')}" for k in ("no_match", "service_error", "species_mismatch") if p.get(k)
                ]
                part += f" ({', '.join(extras)})" if extras else ""
            if t in couldnt_reach:
                part += f" -- {couldnt_reach[t]}"
            summary_parts.append(part)
        summary = "; ".join(summary_parts) + f" ({where}, organism {org})."

        preview_limit = 50 if len(entities) == 1 else 20 if len(entities) <= 3 else 5
        records_preview = {
            e: {t: _preview_records(results[e].get(t), preview_limit) for t in requested
                if outcome[t].get(e) == "annotated"}
            for e in entities[:_PREVIEW_ENTITIES]
        }

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
            "records_preview": records_preview,
        }
        if kegg_reference is not None:
            base_payload["kegg_reference"] = kegg_reference
        mismatch_text = _species_mismatch_text(list(mismatched.values()), org) if mismatched else ""
        if mismatched:
            base_payload["species_mismatches"] = list(mismatched.values())[:10]
        if service_errors:
            base_payload["service_errors_sample"] = service_errors[:5]
        if unreachable:
            base_payload["unreachable_services"] = {t: sorted(names) for t, names in unreachable.items()}

        if n_annotated_pairs == 0:
            if any(p["service_error"] for p in per_type.values()):
                reached = " ".join(dict.fromkeys(couldnt_reach.values())) or "Annotation requests failed."
                error_type = "AnnotationServiceError"
                error = " ".join(filter(None, (reached, mismatch_text))) + (
                    f" Nothing was annotated for the {n_label} in {where}."
                )
            elif mismatched:
                error_type = "SpeciesMismatch"
                error = f"{mismatch_text} Nothing was annotated for the {n_label} in {where}."
            else:
                error_type = "NoAnnotationsFound"
                kinds = " or ".join(type_label.get(t, t) for t in requested)
                error = f"No {kinds} found for the {n_label} in {where}; the lookups completed without a match."
            return {"status": "error", "error_type": error_type, "error": error, **base_payload}

        if resolved_entity != "cell_line":
            # Records of earlier calls on this file are kept; this call's lookups update those
            # of its own entity type and organism.
            earlier = adata.uns.get(_ANNOTATIONS_KEY)
            records_org = _records_organism(resolved_entity, org)
            stored, stored_sources, count_columns, property_columns = _load_annotation_store(
                earlier, adata.obs.columns
            )
            stored, updated = _merge_annotation_records(
                stored, results, sources_queried, failed_lookups, resolved_entity, records_org
            )
            for t in requested:
                col = f"{_COUNT_PREFIX[resolved_entity]}_{t}_n"
                if col in adata.obs.columns and not _owned_by(count_columns, col, query_col):
                    # Another identifier column's counts, or a column this tool did not write:
                    # keep it, and name this column's counts after it.
                    col = f"{col}_{query_col}"
                # The number of records stored for the row's entity (this entity type and organism),
                # from every source of the type.
                n_stored = {
                    e: _n_annotation_items(_scoped_records(stored, e, resolved_entity, records_org).get(t))
                    for e in entities
                }
                adata.obs[col] = [n_stored.get(v, 0) if v else 0 for v in values]
                new_cols.append(col)
                count_columns[col] = {
                    "column": query_col, "entity_type": resolved_entity, "organism": records_org, "annotation_type": t,
                }
            # The other identifier columns' counts follow the stored records too: this call may
            # have looked up their entities again.
            for name, info in list(count_columns.items()):
                if name not in adata.obs.columns or not isinstance(info, dict):
                    del count_columns[name]  # removed from the file since
                    continue
                column = info.get("column")
                if name in new_cols or column not in adata.obs.columns:
                    continue
                rows = [str(v).strip() if pd.notna(v) else "" for v in adata.obs[column]]
                scope = (str(info.get("entity_type")), str(info.get("organism")))
                ann_type = info.get("annotation_type")
                adata.obs[name] = [
                    _n_annotation_items(_scoped_records(stored, v, *scope).get(ann_type)) if v else 0 for v in rows
                ]
            if resolved_entity == "molecule" and "properties" in requested:
                # Scalar RDKit properties as columns, as tl.annotate_molecules writes them, from the
                # stored records: a compound whose lookup failed this time keeps its earlier values.
                props = {e: _scalar_properties(stored, e) for e in entities}
                prop_keys = list(dict.fromkeys(k for p in props.values() for k in p))
                for key in prop_keys:
                    name = f"mol_{key}"
                    if name in adata.obs.columns and not _owned_by(property_columns, name, query_col):
                        # Another identifier column's properties, or a column this tool did not
                        # write: keep it, and name this column's after it.
                        name = f"{name}_{query_col}"
                    column_values = [props.get(v, {}).get(key) if v else None for v in values]
                    present = [x for x in column_values if x is not None]
                    # Rows without the compound (controls) have no value. Keep numbers and flags
                    # as floats with NaN, text as a categorical: both write to .h5ad with gaps.
                    if all(isinstance(x, (bool, int, float, np.number, np.bool_)) for x in present):
                        adata.obs[name] = np.array(
                            [np.nan if x is None else float(x) for x in column_values], dtype=float
                        )
                    else:
                        adata.obs[name] = pd.Categorical([None if x is None else str(x) for x in column_values])
                    new_cols.append(name)
                    property_columns[name] = {"column": query_col, "property": key}
            # Where the stored records came from: one entry per entity type, organism, identifier
            # column, annotation type and source.
            for t, source in sorted(updated):
                entry = {
                    "entity_type": resolved_entity, "organism": records_org, "column": query_col,
                    "annotation_type": t, "source": source,
                }
                if entry not in stored_sources:
                    stored_sources.append(entry)
            new_provenance = {
                "sources": stored_sources, "count_columns": count_columns, "property_columns": property_columns,
            }
            earlier_types = earlier.get("annotation_types", []) if isinstance(earlier, dict) else []
            if isinstance(earlier_types, str):
                earlier_types = [earlier_types]
            adata.uns[_ANNOTATIONS_KEY] = {
                # entity_type, column and organism describe this call; provenance_json says
                # where each stored entity type, organism, type and source came from.
                "version": _RECORDS_VERSION,
                "entity_type": resolved_entity,
                "column": query_col,
                "organism": org,
                "annotation_types": list(dict.fromkeys([*(str(t) for t in earlier_types), *requested])),
                "results_json": json.dumps(stored, default=str),
                "provenance_json": json.dumps(new_provenance),
            }

        if output_path:
            saved_path = os.path.abspath(output_path)
        elif resolved_path is not None:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_annotated{ext}"
        else:
            saved_path = None  # identifiers only: the answer is in records_preview
        if saved_path:
            _save_anndata_safely(adata, saved_path)
        else:
            new_cols = []  # nothing was written, so no file gained columns

        response = {
            "status": "success" if n_annotated_pairs == len(entities) * len(requested) else "partial_success",
            "output_path": saved_path,
            "new_columns_added": new_cols,
            "added_columns": new_cols,
            **base_payload,
        }
        if couldnt_reach:
            response["warning"] = " ".join(dict.fromkeys(couldnt_reach.values())) + (
                " Results from the other sources are included."
            )
        if mismatch_text:
            response["warning"] = " ".join(filter(None, (response.get("warning"), mismatch_text)))
        return response

    except Exception as exc:
        logger.error("annotate_metadata failed: %s", exc, exc_info=True)
        resp: dict[str, Any] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "successful_count": 0,
            "suggestion": "Check that the target dataset contains valid identifiers in .obs.",
        }
        if isinstance(exc, ImportError):
            # A ModuleNotFoundError names the module also when its message does not say "No module named".
            module = exc.name if isinstance(exc, ModuleNotFoundError) else None
            hint = _install_hint(str(exc)) or (_install_hint(f"No module named '{module}'") if module else None)
            if resolved_entity == "cell_line" and (hint is None or module == "pertpy"):
                hint = "Cell-line annotation needs pertpy: pip install 'embpy[pertpy]'."
            resp["suggestion"] = hint or "Install the Python package named in the error."
        return resp


# =========================================================================
# 4. Analysis & Benchmarking Skills
# =========================================================================

def _finite_rows(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """An embedding as a 2-D array, and which of its rows hold only finite values."""
    import scipy.sparse as sp

    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError(f"An embedding must be a 2-D matrix (one row per cell), got shape {values.shape}.")
    return values, np.isfinite(values).all(axis=1)


def _nonfinite_rows_note(names: Any, keys: list[str], n_total: int) -> str:
    """Warning text for cells left out because their embedding has NaN or infinite values."""
    sample = [str(name) for name in list(names)[:5]]
    return (
        f"{len(names)} of {n_total} cells have NaN or infinite values in .obsm {keys} and were left "
        f"out (e.g. {sample}). Such rows usually come from compute_embeddings(..., missing='nan'), "
        "which leaves rows without an identifier (e.g. controls) as NaN."
    )


# Known reasons a scib metric cannot run here, matched in its error message.
_SCIB_METRIC_HINTS = (
    (("WinError 193", "not a valid Win32"), "scib's LISI metrics run a compiled helper that does not run on Windows."),
    (("rpy2",), "kBET needs R with the kBET package and the Python package rpy2."),
    (("value_counts",), "This scib version calls pandas.value_counts, which pandas 3 removed."),
)


def _scib_metric_hint(reason: str) -> str | None:
    """A short explanation for a scib metric failure with a known cause."""
    return next((hint for needles, hint in _SCIB_METRIC_HINTS if any(n in reason for n in needles)), None)


def _metrics_markdown(table: Any, index_name: str) -> str:
    """A Markdown table of a metrics DataFrame, with its row labels as the first column."""

    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "NaN" if math.isnan(value) else f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join([index_name, *(str(c) for c in table.columns)]) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(table.columns)) + " |",
    ]
    for label, row in table.iterrows():
        lines.append("| " + " | ".join([str(label), *(cell(v) for v in row)]) + " |")
    return "\n".join(lines)


def _no_finite_rows_error(path: str, names: Any, keys: list[str]) -> dict[str, Any]:
    """Error reply when no cell has a finite embedding."""
    return {
        "status": "error",
        "error_type": "NonFiniteEmbeddingError",
        "error": (
            f"Every cell has NaN or infinite values in .obsm {keys} (e.g. {[str(n) for n in list(names)[:5]]}), "
            "so nothing can be computed. Such rows usually come from compute_embeddings(..., missing='nan'); "
            "check which identifiers could not be embedded."
        ),
        "path": str(path),
    }


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

    Only two things are added to the file: the coordinates and this embedding's kNN graph,
    stored as scanpy does with ``neighbors_key='neighbors_<model>'`` (.uns['neighbors_<model>'],
    .obsp['neighbors_<model>_connectivities'] and .obsp['neighbors_<model>_distances'];
    ``neighbors_<output_key>`` when output_key is given). An existing .obsm['X_umap'],
    .obsp['connectivities'] / ['distances'] or .uns['neighbors'] / ['umap'] is left as it is.
    Rows of the embedding with NaN or infinite values (e.g. from compute_embeddings with
    missing='nan') are left out: they get NaN coordinates and no graph edges, and the reply
    counts them in n_excluded_nonfinite with a warning.

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
        - neighbors_key: Key of the kNN graph in .uns (its matrices are in .obsp)
        - umap_shape: Array dimensions [n_cells, 2]
        - n_cells: Total cells embedded
        - n_excluded_nonfinite: Cells left out because their embedding is not finite
          (named in ``warning``)
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

        name = embedding_key.replace("X_", "")
        target_key = output_key or f"X_umap_{name}"
        neighbors_key = f"neighbors_{output_key or name}"

        values, finite = _finite_rows(adata.obsm[embedding_key])
        n_excluded = int((~finite).sum())
        if not finite.any():
            return _no_finite_rows_error(path, adata.obs_names, [embedding_key])

        import pandas as pd
        import scipy.sparse as sp

        # Build the graph and the layout in a container holding only the cells with a
        # finite embedding, so scanpy's default keys (X_umap, connectivities, distances,
        # neighbors, umap) in the file are neither overwritten nor fed NaN rows.
        work = anndata.AnnData(obs=pd.DataFrame(index=adata.obs_names[finite]))
        work.obsm["rep"] = values[finite]
        # Adjust n_neighbors for small datasets
        effective_neighbors = min(n_neighbors, max(2, work.n_obs - 1))

        with contextlib.redirect_stdout(sys.stderr):
            sc.pp.neighbors(work, use_rep="rep", n_neighbors=effective_neighbors, random_state=random_state)
            sc.tl.umap(work, min_dist=min_dist, random_state=random_state)

        layout = np.asarray(work.obsm["X_umap"])
        coords = np.full((adata.n_obs, layout.shape[1]), np.nan, dtype=np.result_type(layout.dtype, np.float32))
        coords[finite] = layout
        adata.obsm[target_key] = coords
        # The graph under this embedding's own key, scanpy layout (neighbors_key=...).
        rows = np.flatnonzero(finite)
        for kind in ("connectivities", "distances"):
            graph = sp.coo_matrix(work.obsp[kind])
            adata.obsp[f"{neighbors_key}_{kind}"] = sp.csr_matrix(
                (graph.data, (rows[graph.row], rows[graph.col])), shape=(adata.n_obs, adata.n_obs)
            )
        adata.uns[neighbors_key] = {
            "connectivities_key": f"{neighbors_key}_connectivities",
            "distances_key": f"{neighbors_key}_distances",
            "params": {**dict(work.uns["neighbors"]["params"]), "use_rep": str(embedding_key)},
        }

        if not output_path:
            base, ext = os.path.splitext(resolved_path)
            saved_path = f"{base}_umap{ext}"
        else:
            saved_path = os.path.abspath(output_path)

        _save_anndata_safely(adata, saved_path)

        response = {
            "status": "success",
            "path": str(resolved_path),
            "output_path": saved_path,
            "embedding_key": str(embedding_key),
            "umap_key": str(target_key),
            "neighbors_key": neighbors_key,
            "n_cells": int(adata.n_obs),
            "n_excluded_nonfinite": n_excluded,
            "umap_shape": [int(d) for d in adata.obsm[target_key].shape],
        }
        if n_excluded:
            response["warning"] = (
                _nonfinite_rows_note(adata.obs_names[~finite], [embedding_key], adata.n_obs)
                + " They have NaN UMAP coordinates and no edges in the graph."
            )
        return response
    except Exception as exc:
        logger.error("compute_umap failed on %s: %s", path, exc, exc_info=True)
        resp = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": str(path),
        }
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            resp["suggestion"] = _install_hint(str(exc)) or (
                "Install scanpy using 'pip install scanpy' or 'pip install embpy[mcp]' to enable compute_umap."
            )
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

    Cells whose embedding has NaN or infinite values in any of ``embedding_keys`` (e.g.
    from compute_embeddings with missing='nan') are left out, so every embedding is scored
    on the same cells; the reply counts them in n_excluded_nonfinite with a warning.

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
        - n_cells_scored / n_excluded_nonfinite: Cells scored, and cells left out because
          an embedding is not finite (nonfinite_rows_by_key and ``warning`` name them)
        - metrics_unavailable: Metric -> {reason, embedding_keys, hint} for metrics scib
          could not compute (NaN in the table, left out of the aggregate scores)
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

        # Every embedding is scored on the same cells: those finite in all of them.
        keep = np.ones(adata.n_obs, dtype=bool)
        nonfinite_by_key: dict[str, int] = {}
        for key in keys_list:
            _, finite = _finite_rows(adata.obsm[key])
            if not finite.all():
                nonfinite_by_key[key] = int((~finite).sum())
            keep &= finite
        if not keep.any():
            return _no_finite_rows_error(path, adata.obs_names, list(nonfinite_by_key))
        n_excluded = int((~keep).sum())
        excluded_names = adata.obs_names[~keep]
        n_total = int(adata.n_obs)
        if n_excluded:
            adata = adata[keep].copy()

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

        # Built here rather than with DataFrame.to_markdown, which needs the optional
        # 'tabulate' package; without it the old fallback dropped the embedding names.
        md_table = _metrics_markdown(metrics_df, index_name="embedding")

        metrics_dict: dict[str, dict[str, Any]] = {}
        for row_key, row in metrics_df.iterrows():
            metrics_dict[str(row_key)] = {str(k): _sanitize_val(v) for k, v in row.to_dict().items()}

        summary_records = _sanitize_val(metrics_df.reset_index().to_dict(orient="records"))
        rankings = list(metrics_df.sort_values(by="total", ascending=False).index) if "total" in metrics_df.columns else keys_list

        # Metrics scib could not compute are NaN in the table; say which and why.
        unavailable: dict[str, dict[str, Any]] = {}
        for metric, entry in (metrics_df.attrs.get("metrics_unavailable") or {}).items():
            reason = str(entry.get("reason", ""))
            unavailable[str(metric)] = {
                "reason": reason,
                "embedding_keys": [str(k) for k in entry.get("embedding_keys", [])],
            }
            hint = _scib_metric_hint(reason)
            if hint:
                unavailable[str(metric)]["hint"] = hint

        response = {
            "status": "success",
            "path": str(resolved_path),
            "embedding_keys": keys_list,
            "label_key": label_key,
            "batch_key": batch_key,
            "output_csv": csv_path,
            "n_cells_scored": int(adata.n_obs),
            "n_excluded_nonfinite": n_excluded,
            "markdown_table": md_table,
            "summary": summary_records,
            "rankings": [str(r) for r in rankings],
            "metrics": metrics_dict,
            "metrics_unavailable": unavailable,
        }
        warnings: list[str] = []
        if n_excluded:
            response["nonfinite_rows_by_key"] = nonfinite_by_key
            warnings.append(
                _nonfinite_rows_note(excluded_names, list(nonfinite_by_key), n_total)
                + " Every embedding was scored on the remaining cells."
            )
        if unavailable:
            warnings.append(
                f"{len(unavailable)} scIB metrics could not be computed ({', '.join(unavailable)}): they are "
                "NaN in the table and left out of bio_conservation, batch_correction and total. "
                "metrics_unavailable gives the reason for each."
            )
        if warnings:
            response["warning"] = " ".join(warnings)
        return response

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


def _embedding_key_list(embedding_keys: list[str] | str | None) -> list[str]:
    """The requested embedding keys as a list, without repeats."""
    if not embedding_keys:
        return []
    keys = [embedding_keys] if isinstance(embedding_keys, str) else list(embedding_keys)
    return list(dict.fromkeys(str(key) for key in keys))


def _check_embedding_keys(keys: list[str], sources: list[tuple[str, Any]]) -> dict[str, Any] | None:
    """Error response if a requested embedding is not usable in every source, else None.

    ``sources`` holds (description, AnnData) pairs for the prediction and the reference
    data. A key must be in ``.obsm`` of each, with the same number of columns.
    """
    present = [[str(key) for key in adata.obsm.keys()] for _, adata in sources]
    available = [key for key in present[0] if all(key in keys_ for keys_ in present)]
    missing = {
        key: [name for (name, _), keys_ in zip(sources, present) if key not in keys_]
        for key in keys
        if any(key not in keys_ for keys_ in present)
    }
    if missing:
        sentences = [f"Embedding '{key}' is missing from .obsm of {' and '.join(names)}." for key, names in missing.items()]
        sentences.append(f"Available embedding{'s' if len(available) != 1 else ''}: {', '.join(available) or 'none'}.")
        return {
            "status": "error",
            "error_type": "KeyError",
            "error": " ".join(sentences),
            "missing_keys": list(missing),
            "available_keys": available,
        }
    for key in keys:
        widths = [np.shape(adata.obsm[key])[1] if np.ndim(adata.obsm[key]) > 1 else 1 for _, adata in sources]
        if len(set(widths)) > 1:
            return {
                "status": "error",
                "error_type": "ValueError",
                "error": (
                    f"Embedding '{key}' has {widths[0]} columns in {sources[0][0]} but {widths[1]} columns in "
                    f"{sources[1][0]}. The prediction and the reference must be embedded the same way."
                ),
            }
    return None


# The cell-eval metrics that score an .obsm embedding, the two that `cell-eval run --embed-key`
# configures, and the profiles that run them. Every other metric reads gene expression (.X):
# the DE metrics always, and discrimination_score_l1 ignores embed_key.
_EMBEDDING_METRICS = ("discrimination_score_l2", "discrimination_score_cosine")
_EMBEDDING_PROFILES = ("anndata", "full")
_EXPRESSION_DATA = "gene expression (.X)"


@contextlib.contextmanager
def _logged_errors(logger_name: str):
    """Collect the error messages that ``logger_name`` and its children log in the block.

    Only records logged on the calling thread are kept: the MCP server runs overlapping
    tool calls on worker threads, and cell-eval logs a call's metric errors on its own thread.
    """
    messages: list[str] = []
    caller = threading.get_ident()

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.thread == caller:
                messages.append(record.getMessage())

    handler = _Collect(logging.ERROR)
    target = logging.getLogger(logger_name)
    target.addHandler(handler)
    try:
        yield messages
    finally:
        target.removeHandler(handler)


def _unscored_embedding(key: str, metrics: list[str], errors: list[str]) -> ValueError:
    """The error for an embedding that ``metrics`` failed on, quoting what cell-eval logged."""
    details = " ".join(e for e in errors if any(f"'{m}'" in e for m in metrics)) or "see the server log."
    return ValueError(f"Embedding '{key}' could not be scored with {' or '.join(metrics)}: {details}")


def _score_perturbations(
    adata_pred: Any,
    adata_real: Any,
    control_pert: str,
    pert_col: str,
    profile: str,
    keys: list[str],
    **evaluator_kwargs: Any,
) -> tuple[Any, dict[str, str], list[str]]:
    """Score with cell-eval: gene expression with ``profile``, then each embedding in ``keys``.

    Returns the per-perturbation table, the data each of its columns was scored on, and the
    embeddings that were scored. Each embedding gets its own columns, ``<metric>[<key>]``.
    cell-eval logs a metric that raises and leaves its column out; an embedding metric that
    did that raises ValueError here instead.
    """
    from embpy.tl import run_cell_eval

    # embed_key=None: in this run the embedding metrics score .X, like every other metric.
    with contextlib.redirect_stdout(sys.stderr):
        table = run_cell_eval(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=control_pert,
            pert_col=pert_col,
            profile=profile,
            metric_configs={metric: {"embed_key": None} for metric in _EMBEDDING_METRICS},
            **evaluator_kwargs,
        )
    scored_on = {str(column): _EXPRESSION_DATA for column in table.columns if column != "perturbation"}
    if not keys:
        return table, scored_on, []

    import cell_eval

    other_metrics = [
        metric
        for metric in cell_eval.metrics_registry.list_metrics(cell_eval.MetricType.ANNDATA_PAIR)
        if metric not in _EMBEDDING_METRICS
    ]
    for key in keys:
        with _logged_errors("cell_eval") as errors, contextlib.redirect_stdout(sys.stderr):
            try:
                scores = run_cell_eval(
                    adata_pred=adata_pred,
                    adata_real=adata_real,
                    control_pert=control_pert,
                    pert_col=pert_col,
                    profile="anndata",
                    metric_configs={metric: {"embed_key": key} for metric in _EMBEDDING_METRICS},
                    skip_metrics=other_metrics,
                    skip_de=True,
                    **evaluator_kwargs,
                )
            except Exception as exc:
                # When no metric produced a value, cell-eval fails while summarizing the results.
                if not errors:
                    raise
                raise _unscored_embedding(key, list(_EMBEDDING_METRICS), errors) from exc
        failed = [m for m in _EMBEDDING_METRICS if m not in scores.columns or scores[m].isna().all()]
        if failed:
            raise _unscored_embedding(key, failed, errors)
        labelled = {metric: f"{metric}[{key}]" for metric in _EMBEDDING_METRICS}
        scores = scores[["perturbation", *_EMBEDDING_METRICS]].rename(columns=labelled)
        table = table.merge(scores, on="perturbation", how="left") if "perturbation" in table.columns else scores
        scored_on.update({column: f"embedding {key} (.obsm['{key}'])" for column in labelled.values()})
    return table, scored_on, list(keys)


def _write_benchmark_reports(table: Any, folder: str) -> list[str]:
    """Write results.csv and agg_results.csv as ``cell-eval run`` does; return the reports in ``folder``.

    agg_results.csv holds the same statistics cell-eval computes per score (polars
    ``describe``). cell-eval itself has already written its DE tables to ``folder``.
    """
    import polars as pl

    table.to_csv(os.path.join(folder, "results.csv"), index=False)
    pl.from_pandas(table.drop(columns=["perturbation"], errors="ignore")).describe().write_csv(
        os.path.join(folder, "agg_results.csv")
    )
    names = ("results.csv", "agg_results.csv", "real_de.csv", "pred_de.csv")
    return [os.path.join(folder, name) for name in names if os.path.exists(os.path.join(folder, name))]


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
        ``.obsm`` embeddings to evaluate. Each must be in ``.obsm`` of both the prediction
        and the reference data, with the same number of columns; otherwise the call fails
        and lists the embeddings that are available. As in ``cell-eval run --embed-key``,
        discrimination_score_l2 and discrimination_score_cosine score an embedding; they
        run with profile 'anndata' or 'full' and give each key its own columns, e.g.
        ``discrimination_score_cosine[X_pca]``. Profiles 'minimal', 'vcc' and 'de' score
        gene expression only, so the keys are then reported in ``unscored_keys``.
    output_csv : str, optional
        Filesystem path to export results as CSV.
    output_dir : str, optional
        Directory for the cell-eval reports: results.csv (the per-perturbation scores),
        agg_results.csv (statistics per score), and real_de.csv and pred_de.csv
        (differential expression against control_pert). Without it, nothing is written
        apart from output_csv.
    profile : {'full', 'vcc', 'minimal', 'de', 'anndata'}, default='minimal'
        Evaluation profile to run.

    Returns
    -------
    dict[str, Any]
        Perturbation evaluation summary:
        - status: "success" or "error"
        - markdown_table: Formatted table of perturbation metrics
        - n_perturbations: Number of perturbations evaluated
        - results: Summary records, one per perturbation
        - scored_on: The data each score column was computed on, "gene expression (.X)"
          or "embedding <key> (.obsm['<key>'])"
        - evaluated_keys: The embeddings a metric read
        - unscored_keys: Requested embeddings that the profile does not score (see warning)
        - output_csv: Path to exported CSV
        - output_dir, report_files: The report folder and the reports written to it
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

        key_list = _embedding_key_list(embedding_keys)
        if key_list:
            if path_pred:
                sources = [(f"the prediction file ({path_pred})", adata_pred), (f"the reference file ({path})", adata_real)]
            else:
                sources = [(f"the dataset ({path})", adata_real)]
            problem = _check_embedding_keys(key_list, sources)
            if problem:
                return problem

        scored_keys = key_list if profile in _EMBEDDING_PROFILES else []
        unscored_keys = [key for key in key_list if key not in scored_keys]
        # cell-eval writes its DE tables to a folder (./cell-eval-outdir unless told otherwise):
        # output_dir, or a temporary folder that is removed afterwards.
        with contextlib.ExitStack() as stack:
            if output_dir:
                report_dir = os.path.abspath(output_dir)
                os.makedirs(report_dir, exist_ok=True)
            else:
                import tempfile

                report_dir = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="embpy-cell-eval-", ignore_cleanup_errors=True)
                )
            df_results, scored_on, evaluated_keys = _score_perturbations(
                adata_pred, adata_real, control_pert, pert_col, profile, scored_keys, outdir=report_dir
            )
        report_files = _write_benchmark_reports(df_results, report_dir) if output_dir else []

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

        response: dict[str, Any] = {
            "status": "success",
            "path": str(resolved_real),
            "control_pert": control_pert,
            "pert_col": pert_col,
            "profile": profile,
            "n_perturbations": pert_count,
            "perturbation_count": pert_count,
            "evaluated_keys": evaluated_keys,
            "unscored_keys": unscored_keys,
            "scored_on": scored_on,
            "markdown_table": markdown_table,
            "results": summary_records,
            "summary_metrics": summary_records,
            "output_csv": saved_csv,
            "output_dir": report_dir if output_dir else None,
            "report_files": report_files,
        }
        if unscored_keys:
            response["warning"] = (
                f"Profile '{profile}' scores gene expression (.X) only, so {', '.join(unscored_keys)} "
                f"{'was' if len(unscored_keys) == 1 else 'were'} not scored. An embedding is scored by "
                f"{' and '.join(_EMBEDDING_METRICS)}, which run with "
                f"{' or '.join(f'profile={name!r}' for name in _EMBEDDING_PROFILES)}."
            )
        return response

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
