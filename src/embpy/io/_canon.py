"""Shared id-canonicalization for the io layer.

One place that turns raw row identifiers into the canonical scheme for an
entity type, reusing the existing resolvers (which own all id-mapping
logic). Used by both :mod:`embpy.io.legacy` (normalizing old artifacts)
and :meth:`embpy.embedder.BioEmbedder.embed` (the standardized output
API), so neither duplicates the mapping/dedup/alias bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

EntityType = Literal["gene", "molecule", "protein", "sequence", "text"]

# Canonical id scheme per entity type (the internal key, always emitted).
SCHEME: dict[str, str] = {
    "gene": "ensembl_gene_id",
    "molecule": "canonical_smiles",
    "protein": "uniprot",
    "sequence": "sequence",
    "text": "input_text_id",
}

_ENSEMBL_PREFIXES = ("ENSG", "ENSMUSG", "ENS")


def canonicalize(
    raw_ids: list[str],
    entity_type: str,
    organism: str,
    *,
    id_type: str | None = None,
    gene_resolver: Any | None = None,
    protein_resolver: Any | None = None,
    molecule_resolver: Any | None = None,
) -> tuple[list[str | None], np.ndarray]:
    """Map raw ids to the canonical scheme; returns (canonical_or_None, keep_mask).

    ``keep_mask[i]`` is False when ``raw_ids[i]`` could not be resolved.
    Logs a counted WARNING (with examples) for anything dropped.
    """
    if entity_type == "sequence":
        if id_type in ("sequence", "raw_sequence", "input_sequence"):
            canon: list[str | None] = [_stable_input_id("seq", i, x) for i, x in enumerate(raw_ids)]
        else:
            canon = [str(x) for x in raw_ids]
    elif entity_type == "text":
        canon = [_stable_input_id("text", i, x) for i, x in enumerate(raw_ids)]
    elif entity_type == "molecule":
        resolver = molecule_resolver
        if resolver is None:
            from embpy.resources.molecule.resolver import DrugResolver

            resolver = DrugResolver()
        canon = [resolver.canonicalize_smiles(s) for s in raw_ids]
    elif entity_type == "gene":
        canon = _canonicalize_genes(raw_ids, organism, id_type=id_type, resolver=gene_resolver)
    elif entity_type == "protein":
        canon = _canonicalize_proteins(
            raw_ids,
            organism,
            id_type=id_type,
            resolver=protein_resolver,
        )
    else:
        raise ValueError(f"entity_type must be one of {sorted(SCHEME)}, got {entity_type!r}.")

    keep = np.array([c is not None and c != "" for c in canon], dtype=bool)
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = [raw_ids[i] for i in np.where(~keep)[0][:10]]
        logger.warning(
            "Dropping %d/%d %s ids that did not canonicalize. First: %s",
            n_drop,
            len(raw_ids),
            entity_type,
            dropped,
        )
    return canon, keep


def _canonicalize_genes(
    raw_ids: list[str],
    organism: str,
    *,
    id_type: str | None,
    resolver: Any | None,
) -> list[str | None]:
    """Already-Ensembl ids pass through; symbols are resolved via GeneResolver."""
    if id_type in ("ensembl_id", "ensembl_gene_id") or (
        raw_ids and all(str(x).upper().startswith(_ENSEMBL_PREFIXES) for x in raw_ids)
    ):
        logger.info("Gene ids already look like Ensembl; no resolution needed.")
        return list(raw_ids)
    if id_type not in (None, "symbol"):
        raise ValueError(
            f"canonicalization: gene id_type must be 'symbol' or 'ensembl_id' for standardized output, got {id_type!r}."
        )
    if resolver is None:
        from embpy.resources.gene.resolver import GeneResolver

        resolver = GeneResolver(species=organism)
    mapping = resolver.symbols_to_ensembl_batch(list(raw_ids), organism=organism)
    return [mapping.get(x) for x in raw_ids]


def _canonicalize_proteins(
    raw_ids: list[str],
    organism: str,
    *,
    id_type: str | None,
    resolver: Any | None,
) -> list[str | None]:
    it = id_type or "symbol"
    if it in ("uniprot", "uniprot_id", "uniprot_accession"):
        return list(raw_ids)
    if it not in ("symbol", "ensembl_id"):
        raise ValueError(
            f"canonicalization: protein id_type must be 'symbol', 'ensembl_id', or 'uniprot_id', got {id_type!r}."
        )
    if resolver is None:
        from embpy.resources.protein.resolver import ProteinResolver

        resolver = ProteinResolver(organism=organism)
    return [resolver.resolve_uniprot_id(x, id_type=it, organism=organism) for x in raw_ids]


def _stable_input_id(prefix: str, index: int, value: str) -> str:
    import hashlib

    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{index:06d}_{digest}"


def drop_and_dedup(
    canon: list[str | None],
    matrix: np.ndarray,
    keep: np.ndarray,
    raw_ids: list[str],
    alias_cols: dict[str, list[str]],
) -> tuple[list[str], np.ndarray, dict[str, list[str]], list[str]]:
    """Drop unresolved rows, then collapse duplicate canonical ids (keep first).

    Keeps ``matrix``, ``raw_ids`` and ``alias_cols`` row-aligned to the
    surviving canonical ids throughout.
    """
    idx = [i for i in range(len(canon)) if keep[i]]
    seen: set[str] = set()
    final_idx: list[int] = []
    n_dup = 0
    for i in idx:
        cid = canon[i]
        if cid in seen:
            n_dup += 1
            continue
        seen.add(cid)  # type: ignore[arg-type]
        final_idx.append(i)
    if n_dup:
        logger.warning("Dropped %d rows that collapsed to a duplicate canonical id.", n_dup)

    out_ids = [str(canon[i]) for i in final_idx]
    out_mat = matrix[final_idx]
    out_raw = [raw_ids[i] for i in final_idx]
    out_alias = {k: [v[i] for i in final_idx] for k, v in alias_cols.items()}
    return out_ids, out_mat, out_alias, out_raw


def build_aliases(
    entity_type: str,
    canon: list[str],
    raw: list[str],
    alias_cols: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """Attach display cross-refs (never the key): name / original symbol."""
    aliases: dict[str, dict[str, str]] = {}

    if "name" in alias_cols:
        for cid, nm in zip(canon, alias_cols["name"], strict=False):
            if nm and nm != "nan":
                aliases.setdefault(cid, {})["name"] = nm

    if entity_type in ("gene", "protein"):
        for cid, r in zip(canon, raw, strict=False):
            if r and r != cid:
                aliases.setdefault(cid, {})["gene_symbol"] = r
    elif entity_type == "molecule":
        # A non-canonical SMILES is a different *string* for the same molecule,
        # so the form the caller passed has to stay addressable. `to_anndata`
        # re-indexes a target AnnData by that target's own `.obs_names`, and
        # without this alias anyone who passed, say, Kekule caffeine gets
        # "target .obs_names have no embedding" for a molecule that embedded
        # perfectly well -- the canonical id simply is not the string they
        # indexed by.
        for cid, r in zip(canon, raw, strict=False):
            if r and r != cid:
                aliases.setdefault(cid, {})["input_smiles"] = r
    elif entity_type in ("sequence", "text"):
        scheme = "original_sequence" if entity_type == "sequence" else "original_text"
        for cid, r in zip(canon, raw, strict=False):
            if r and r != cid:
                aliases.setdefault(cid, {})[scheme] = r

    return aliases


__all__ = ["SCHEME", "EntityType", "canonicalize", "drop_and_dedup", "build_aliases"]
