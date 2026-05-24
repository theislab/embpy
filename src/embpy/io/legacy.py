"""Normalize embpy's historical, un-harmonized embedding artifacts.

Pre-standardization, each model/cache invented its own row id and column
convention (Ensembl-indexed csv, SMILES-indexed csv with offset dim
names, ``name``+``canonical_smiles`` csv, ``{ids, embeddings}`` npz, ...).
:func:`load_legacy_embedding` reads any of them and emits a clean
:class:`EmbeddingResult` whose rows are canonical ids and whose columns
are ``dim_*``.

The caller must declare ``entity_type`` -- we never guess what the rows
*are*. We do auto-detect the id *column* and log it at INFO, and we
canonicalize ids to the scheme for that entity type, reusing the existing
resolvers (no duplicated id-mapping logic here).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._canon import SCHEME, EntityType, build_aliases, canonicalize, drop_and_dedup
from .result import EmbeddingProvenance, EmbeddingResult

logger = logging.getLogger(__name__)

# CSV id-column detection priority (highest first). Anything not matched
# falls back to the first (unnamed/"Unnamed: 0") column.
_ID_CANDIDATES = ("canonical_smiles", "smiles", "name")
# NPZ 1-D id arrays we recognise, in priority order.
_NPZ_ID_KEYS = ("gene_ids", "symbols", "ids", "proteins", "names")
_NPZ_MATRIX_KEYS = ("embeddings", "X", "matrix")


def load_legacy_embedding(
    path: str | Path,
    *,
    entity_type: EntityType,
    organism: str = "human",
    max_rows: int | None = None,
) -> EmbeddingResult:
    """Read a heterogeneous legacy embedding file into an EmbeddingResult.

    Parameters
    ----------
    path
        ``.csv`` or ``.npz`` artifact.
    entity_type
        Required -- what the rows are (``"gene"``, ``"molecule"``,
        ``"protein"``, ``"sequence"``). Never inferred.
    organism
        Passed to gene/protein resolvers.
    max_rows
        If given, only read the first ``max_rows`` rows (preview / fast
        tests on the 100k-row drug files). ``None`` reads everything.

    Returns
    -------
    EmbeddingResult
        Rows = canonical ids for ``entity_type``; columns = ``dim_*``.
        Ids that fail to canonicalize, and rows that collapse to a
        duplicate canonical id, are dropped with a counted WARNING.
    """
    if entity_type not in SCHEME:
        raise ValueError(
            f"entity_type must be one of {sorted(SCHEME)}, got {entity_type!r}."
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Legacy embedding not found: {p}")

    if p.suffix == ".npz":
        raw_ids, matrix, alias_cols = _read_npz(p)
    elif p.suffix in (".csv", ".tsv"):
        raw_ids, matrix, alias_cols = _read_csv(p, max_rows=max_rows)
    else:
        raise ValueError(f"Unsupported legacy format {p.suffix!r} (need .csv/.npz).")
    if max_rows is not None and p.suffix == ".npz":
        raw_ids, matrix = raw_ids[:max_rows], matrix[:max_rows]
        alias_cols = {k: v[:max_rows] for k, v in alias_cols.items()}

    canon, keep = canonicalize(raw_ids, entity_type, organism)
    canon, matrix, alias_cols, kept_raw = drop_and_dedup(
        canon, matrix, keep, raw_ids, alias_cols,
    )

    aliases = build_aliases(entity_type, canon, kept_raw, alias_cols)
    prov = EmbeddingProvenance.create(model=p.stem, extra={"source_file": str(p)})
    logger.info(
        "Loaded legacy %s embedding from %s: %d entities x %d dims [%s].",
        entity_type, p.name, len(canon), matrix.shape[1], SCHEME[entity_type],
    )
    return EmbeddingResult(
        matrix=matrix,
        entity_ids=tuple(canon),
        entity_type=entity_type,
        id_scheme=SCHEME[entity_type],
        provenance=prov,
        aliases=aliases or None,
    )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_csv(
    path: Path, *, max_rows: int | None,
) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    df = pd.read_csv(path, nrows=max_rows)
    cols = list(df.columns)
    if not cols:
        raise ValueError(f"{path.name} has no columns.")

    id_col = next((c for c in _ID_CANDIDATES if c in cols), None)
    if id_col is None:
        # Unnamed first column = the row id (Ensembl / raw SMILES / ...).
        id_col = cols[0]
        logger.info("Detected id column in %s: first/unnamed column %r.", path.name, id_col)
    else:
        logger.info("Detected id column in %s: %r.", path.name, id_col)

    alias_cols = {
        c: [str(x) for x in df[c].tolist()]
        for c in _ID_CANDIDATES
        if c in cols and c != id_col
    }
    dim_cols = [c for c in cols if c != id_col and c not in alias_cols]
    if not dim_cols:
        raise ValueError(f"{path.name}: no embedding-dimension columns after id/alias detection.")

    ids = [str(x) for x in df[id_col].tolist()]
    matrix = df[dim_cols].to_numpy(dtype=np.float32)
    return ids, matrix, alias_cols


def _read_npz(path: Path) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)

    mat_key = next((k for k in _NPZ_MATRIX_KEYS if k in keys), None)
    id_key = next((k for k in _NPZ_ID_KEYS if k in keys), None)
    if mat_key is not None and id_key is not None:
        logger.info("Detected npz layout in %s: ids=%r, matrix=%r.", path.name, id_key, mat_key)
        ids = [str(x) for x in np.asarray(z[id_key]).tolist()]
        matrix = np.asarray(z[mat_key], dtype=np.float32)
        if matrix.shape[0] != len(ids):
            raise ValueError(
                f"{path.name}: '{mat_key}' has {matrix.shape[0]} rows but "
                f"'{id_key}' has {len(ids)} ids."
            )
        return ids, matrix, {}

    # Fallback: one array per id (per-key npz).
    logger.info("Detected npz layout in %s: one array per key (%d keys).", path.name, len(keys))
    ids = list(keys)
    matrix = np.stack([np.asarray(z[k], dtype=np.float32).ravel() for k in keys], axis=0)
    return ids, matrix, {}


__all__ = ["load_legacy_embedding"]
