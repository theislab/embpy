r"""One-shot migration of legacy gene-embedding tables into ``.emstore``.

The legacy world-model code path read gene embeddings from a CSV (first column
= gene symbol, remaining columns = embedding entries) or an ``.npz`` archive
carrying ``"symbols"`` and ``"embeddings"`` arrays. This module converts either
format into a canonical :class:`~embpy.store.EmbeddingStore` (and, optionally,
an on-disk ``.emstore`` directory) so those vectors -- gene2vec, GenePT,
ESM-derived tables, etc. -- remain usable once the CSV route is retired.

It is a faithful 1:1 conversion: duplicate ids or NaN/Inf entries surface as a
loud :class:`ValueError` (via :class:`~embpy.io.result.EmbeddingResult`
validation) rather than being silently dropped.

CLI
---
::

    python -m embpy.store.migrate embeddings_3072.csv gene_genept.emstore \\
        --model genept --id-scheme symbol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.store.core import EmbeddingStore

__all__ = ["gene_store_from_table", "migrate_table_to_emstore", "read_embedding_table"]


def read_embedding_table(source: str | Path | pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Read ``(ids, matrix)`` from a CSV / NPZ path or an in-memory DataFrame.

    * CSV  -- first column is the id, remaining columns are the embedding.
    * NPZ  -- arrays ``"symbols"`` (ids) and ``"embeddings"`` (matrix).
    * DataFrame -- index is the id, columns are the embedding.
    """
    if isinstance(source, pd.DataFrame):
        ids = [str(s) for s in source.index.tolist()]
        matrix = source.to_numpy(dtype=np.float32)
        return ids, matrix

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"embpy.store.migrate: embedding table not found: {path}.")

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        if "symbols" not in archive or "embeddings" not in archive:
            raise KeyError(
                f"embpy.store.migrate: {path} must contain 'symbols' and 'embeddings' arrays; "
                f"found {list(archive.keys())}."
            )
        ids = [str(s) for s in archive["symbols"]]
        matrix = np.asarray(archive["embeddings"], dtype=np.float32)
    else:
        frame = pd.read_csv(path, index_col=0)
        ids = [str(s) for s in frame.index.tolist()]
        matrix = frame.to_numpy(dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"embpy.store.migrate: embedding table from {path} must be 2D, got {matrix.shape!r}.")
    return ids, matrix


def gene_store_from_table(
    source: str | Path | pd.DataFrame,
    *,
    model: str,
    entity_type: str = "gene",
    id_scheme: str = "symbol",
    key: str | None = None,
    pooling: str | None = None,
    organism: str | None = None,
) -> EmbeddingStore:
    """Build an :class:`EmbeddingStore` from a legacy embedding table.

    ``model`` names the source embedding (e.g. ``"genept"``) and is stamped
    into the block's :class:`~embpy.io.result.EmbeddingProvenance` so a
    migrated store self-documents where its vectors came from.
    """
    ids, matrix = read_embedding_table(source)
    src_repr = "<dataframe>" if isinstance(source, pd.DataFrame) else str(source)
    provenance = EmbeddingProvenance.create(
        model=str(model),
        pooling=pooling,
        extra={"migrated_from": src_repr, "organism": organism},
    )
    result = EmbeddingResult(
        matrix=matrix,
        entity_ids=tuple(ids),
        entity_type=str(entity_type),
        id_scheme=str(id_scheme),
        provenance=provenance,
    )
    store = EmbeddingStore()
    store.add_result(result, key=key)
    return store


def migrate_table_to_emstore(
    source: str | Path | pd.DataFrame,
    dest: str | Path,
    *,
    model: str,
    entity_type: str = "gene",
    id_scheme: str = "symbol",
    key: str | None = None,
    pooling: str | None = None,
    organism: str | None = None,
) -> Path:
    """Convert a CSV / NPZ / DataFrame embedding table into a ``.emstore`` dir.

    Returns the written ``.emstore`` path. ``dest`` must end with ``.emstore``.
    """
    store = gene_store_from_table(
        source,
        model=model,
        entity_type=entity_type,
        id_scheme=id_scheme,
        key=key,
        pooling=pooling,
        organism=organism,
    )
    return store.write(dest)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m embpy.store.migrate",
        description="Convert a legacy CSV/NPZ gene-embedding table into a .emstore directory.",
    )
    parser.add_argument("source", help="Path to the CSV or NPZ embedding table.")
    parser.add_argument("dest", help="Output path ending in .emstore.")
    parser.add_argument("--model", required=True, help="Source embedding name (e.g. genept, gene2vec).")
    parser.add_argument("--entity-type", default="gene")
    parser.add_argument("--id-scheme", default="symbol")
    parser.add_argument("--key", default=None, help="Embedding key in the store (default derived from model).")
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--organism", default=None)
    args = parser.parse_args(argv)

    path = migrate_table_to_emstore(
        args.source,
        args.dest,
        model=args.model,
        entity_type=args.entity_type,
        id_scheme=args.id_scheme,
        key=args.key,
        pooling=args.pooling,
        organism=args.organism,
    )
    store = EmbeddingStore.read(path)
    block = store.embedding(store.keys()[0])
    print(  # user-facing CLI summary
        f"Wrote {path} :: key={store.keys()[0]!r} n_entities={block.n_entities} n_dims={block.n_dims}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
