from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embpy.io.result import EmbeddingProvenance, EmbeddingResult


@dataclass(slots=True)
class EmbeddingBlock:
    """A validated reusable embedding matrix plus semantic metadata."""

    key: str
    matrix: np.ndarray
    entity_ids: tuple[str, ...]
    entity_type: str
    id_scheme: str
    aliases: Mapping[str, Mapping[str, str]] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("embedding store: embedding key must be a non-empty string.")

        matrix = self.matrix if isinstance(self.matrix, np.memmap) else np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise ValueError(
                f"embedding store: matrix for {self.key!r} must be 2D (n_entities, n_dims), got shape {matrix.shape!r}."
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"embedding store: matrix for {self.key!r} must be numeric, got {matrix.dtype}.")
        if matrix.dtype != np.float32:
            matrix = matrix.astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f"embedding store: matrix for {self.key!r} contains NaN/Inf values.")
        self.matrix = matrix

        ids = tuple(str(x) for x in self.entity_ids)
        if len(ids) != matrix.shape[0]:
            raise ValueError(
                f"embedding store: entity_ids for {self.key!r} has length {len(ids)} "
                f"but matrix has {matrix.shape[0]} rows."
            )
        duplicate = _first_duplicate(ids)
        if duplicate is not None:
            raise ValueError(f"embedding store: entity_ids for {self.key!r} are not unique; duplicate={duplicate!r}.")
        self.entity_ids = ids

        aliases = _normalize_aliases(self.aliases)
        missing_alias_keys = sorted(set(aliases) - set(ids))
        if missing_alias_keys:
            raise ValueError(
                f"embedding store: aliases for {self.key!r} contain ids not present in entity_ids "
                f"(first={missing_alias_keys[0]!r})."
            )
        self.aliases = aliases
        self.provenance = dict(self.provenance)

    @property
    def n_entities(self) -> int:
        """Number of rows in the embedding matrix."""
        return int(self.matrix.shape[0])

    @property
    def n_dims(self) -> int:
        """Number of embedding dimensions."""
        return int(self.matrix.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """Return a table with canonical ids, aliases, and dimension columns."""
        index = pd.Index(self.entity_ids, name=self.id_scheme)
        dim_cols = [f"dim_{i}" for i in range(self.n_dims)]
        frame = pd.DataFrame(self.matrix, index=index, columns=dim_cols)
        alias_frame = _aliases_to_frame(self.entity_ids, self.aliases, index_name=self.id_scheme)
        if alias_frame.empty:
            return frame
        return pd.concat([alias_frame, frame], axis=1)


@dataclass(slots=True)
class RelationTable:
    """A typed edge table connecting two canonical entity sets."""

    name: str
    frame: pd.DataFrame
    source_type: str
    target_type: str
    source_col: str = "source_id"
    target_col: str = "target_id"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("embedding store: relation name must be a non-empty string.")
        missing = [c for c in (self.source_col, self.target_col) if c not in self.frame.columns]
        if missing:
            raise ValueError(
                f"embedding store: relation {self.name!r} is missing required column(s) {missing}; "
                f"available columns: {list(self.frame.columns)}."
            )
        frame = self.frame.copy()
        if frame[[self.source_col, self.target_col]].isna().any().any():
            raise ValueError(f"embedding store: relation {self.name!r} contains null source/target ids.")
        frame[self.source_col] = frame[self.source_col].astype(str)
        frame[self.target_col] = frame[self.target_col].astype(str)
        self.frame = frame.reset_index(drop=True)

    @property
    def n_edges(self) -> int:
        """Number of edges in the relation table."""
        return int(len(self.frame))

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the relation DataFrame."""
        return self.frame.copy()


class EmbeddingStore:
    """Reusable biological embedding universe with typed entity relations.

    The store deliberately does not run resolvers or model inference. It
    accepts canonicalized :class:`~embpy.io.result.EmbeddingResult` objects
    or explicit matrices and records enough metadata to connect those
    embeddings to AnnData experiments through ``adata.embpy``.
    """

    def __init__(self) -> None:
        self.embeddings: dict[str, EmbeddingBlock] = {}
        self.entities: dict[str, pd.DataFrame] = {}
        self.entity_id_schemes: dict[str, str] = {}
        self.relations: dict[str, RelationTable] = {}

    @classmethod
    def from_results(cls, results: EmbeddingResult | Sequence[EmbeddingResult]) -> EmbeddingStore:
        """Build a store from one or more canonical embedding results."""
        store = cls()
        if isinstance(results, EmbeddingResult):
            results = [results]
        for result in results:
            store.add_result(result)
        return store

    @classmethod
    def read(cls, path: str | Path, *, backed: bool = False) -> EmbeddingStore:
        """Read a ``.emstore`` directory from disk.

        Parameters
        ----------
        path
            Directory created by :meth:`write`.
        backed
            If ``True``, embedding matrices are opened with NumPy memory
            mapping so large universes can be inspected without eagerly
            copying the matrix into memory.
        """
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"embedding store: missing manifest.json under {root}.")
        manifest = json.loads(manifest_path.read_text())

        store = cls()
        for entity_type, meta in manifest.get("entities", {}).items():
            frame = pd.read_parquet(root / "entities" / f"{entity_type}.parquet")
            id_column = str(meta["id_column"])
            store.add_entities(
                entity_type,
                frame,
                id_column=id_column,
                id_scheme=str(meta.get("id_scheme") or id_column),
            )

        for key, meta in manifest.get("embeddings", {}).items():
            emb_dir = root / "embeddings" / str(meta["dir"])
            matrix = np.load(emb_dir / "matrix.npy", mmap_mode="r" if backed else None)
            index = pd.read_parquet(emb_dir / "index.parquet")
            id_column = str(meta["id_scheme"])
            entity_ids = tuple(index[id_column].astype(str))
            aliases = _frame_to_aliases(index, id_column=id_column)
            block_meta = json.loads((emb_dir / "metadata.json").read_text())
            store.add_embedding(
                key,
                matrix,
                entity_ids,
                entity_type=str(block_meta["entity_type"]),
                id_scheme=id_column,
                aliases=aliases,
                provenance=block_meta.get("provenance", {}),
            )

        for name, meta in manifest.get("relations", {}).items():
            frame = pd.read_parquet(root / "relations" / f"{name}.parquet")
            store.add_relation(
                name,
                frame,
                source_type=str(meta["source_type"]),
                target_type=str(meta["target_type"]),
                source_col=str(meta.get("source_col", "source_id")),
                target_col=str(meta.get("target_col", "target_id")),
            )
        return store

    def add_result(self, result: EmbeddingResult, key: str | None = None) -> EmbeddingBlock:
        """Add a canonical :class:`EmbeddingResult` to the store."""
        if not isinstance(result, EmbeddingResult):
            raise TypeError(f"embedding store: expected EmbeddingResult, got {type(result).__name__}.")
        block_key = key or _default_result_key(result)
        return self.add_embedding(
            block_key,
            result.matrix,
            result.entity_ids,
            entity_type=result.entity_type,
            id_scheme=result.id_scheme,
            aliases=result.aliases,
            provenance=result.provenance.to_dict(),
        )

    def add_embedding(
        self,
        key: str,
        matrix: Any,
        entity_ids: Sequence[Any],
        entity_type: str,
        id_scheme: str,
        aliases: Mapping[str, Mapping[str, str]] | None = None,
        provenance: Mapping[str, Any] | EmbeddingProvenance | None = None,
    ) -> EmbeddingBlock:
        """Add a validated embedding block and ensure its entity table exists."""
        if isinstance(provenance, EmbeddingProvenance):
            prov: Mapping[str, Any] = provenance.to_dict()
        else:
            prov = dict(provenance or {})
        block = EmbeddingBlock(
            key=str(key),
            matrix=matrix,
            entity_ids=tuple(str(x) for x in entity_ids),
            entity_type=str(entity_type),
            id_scheme=str(id_scheme),
            aliases=aliases,
            provenance=prov,
        )
        if block.key in self.embeddings:
            raise ValueError(f"embedding store: embedding key {block.key!r} already exists.")
        self.embeddings[block.key] = block
        self._ensure_entity_rows(block.entity_type, block.entity_ids, block.id_scheme, block.aliases)
        return block

    def add_entities(
        self,
        entity_type: str,
        frame: pd.DataFrame,
        id_column: str | None = None,
        id_scheme: str | None = None,
    ) -> pd.DataFrame:
        """Register canonical entities and optional aliases/annotations."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"embedding store: entities must be a pandas DataFrame, got {type(frame).__name__}.")
        ids, out, column = _entity_ids_from_frame(frame, id_column=id_column)
        duplicate = _first_duplicate(tuple(ids))
        if duplicate is not None:
            raise ValueError(f"embedding store: duplicate entity id {duplicate!r} for entity_type={entity_type!r}.")
        out.index = pd.Index(ids, name=id_scheme or column)
        if column in out.columns:
            out = out.drop(columns=[column])
        out = out.copy()

        et = str(entity_type)
        if et in self.entities:
            old = self.entities[et]
            combined = pd.concat([old, out[~out.index.isin(old.index)]], axis=0)
            self.entities[et] = combined
        else:
            self.entities[et] = out
        self.entity_id_schemes[et] = str(id_scheme or column)
        return self.entities[et].copy()

    def add_relation(
        self,
        name: str,
        edges: pd.DataFrame | RelationTable,
        source_type: str | None = None,
        target_type: str | None = None,
        *,
        source_col: str = "source_id",
        target_col: str = "target_id",
    ) -> RelationTable:
        """Register a typed relation table such as perturbation -> gene."""
        if isinstance(edges, RelationTable):
            relation = edges
        else:
            if source_type is None or target_type is None:
                raise ValueError("embedding store: source_type and target_type are required for relation DataFrames.")
            relation = RelationTable(
                name=str(name),
                frame=edges,
                source_type=str(source_type),
                target_type=str(target_type),
                source_col=source_col,
                target_col=target_col,
            )
        if relation.name in self.relations:
            raise ValueError(f"embedding store: relation {relation.name!r} already exists.")
        self.relations[relation.name] = relation
        return relation

    def embedding(self, key: str) -> EmbeddingBlock:
        """Return an embedding block by key."""
        try:
            return self.embeddings[key]
        except KeyError as exc:
            raise KeyError(f"embedding store: embedding {key!r} not found. Available: {self.keys()}.") from exc

    def entity_table(self, entity_type: str) -> pd.DataFrame:
        """Return registered entities for one entity type."""
        try:
            return self.entities[entity_type].copy()
        except KeyError as exc:
            raise KeyError(
                f"embedding store: entity_type {entity_type!r} not found. Available: {sorted(self.entities)}."
            ) from exc

    def relation(self, name: str) -> RelationTable:
        """Return a typed relation table by name."""
        try:
            return self.relations[name]
        except KeyError as exc:
            raise KeyError(
                f"embedding store: relation {name!r} not found. Available: {sorted(self.relations)}."
            ) from exc

    def keys(self) -> list[str]:
        """Return embedding keys in deterministic order."""
        return sorted(self.embeddings)

    def describe(self) -> pd.DataFrame:
        """Return a compact summary of entities, embeddings, and relations."""
        rows: list[dict[str, Any]] = []
        for entity_type, frame in sorted(self.entities.items()):
            rows.append(
                {
                    "kind": "entity",
                    "name": entity_type,
                    "entity_type": entity_type,
                    "n": int(len(frame)),
                    "n_dims": None,
                    "id_scheme": self.entity_id_schemes.get(entity_type),
                }
            )
        for key, block in sorted(self.embeddings.items()):
            rows.append(
                {
                    "kind": "embedding",
                    "name": key,
                    "entity_type": block.entity_type,
                    "n": block.n_entities,
                    "n_dims": block.n_dims,
                    "id_scheme": block.id_scheme,
                }
            )
        for name, relation in sorted(self.relations.items()):
            rows.append(
                {
                    "kind": "relation",
                    "name": name,
                    "entity_type": f"{relation.source_type}->{relation.target_type}",
                    "n": relation.n_edges,
                    "n_dims": None,
                    "id_scheme": None,
                }
            )
        return pd.DataFrame(rows)

    def audit(self) -> pd.DataFrame:
        """Return validation issues such as relation edges pointing to missing ids."""
        rows: list[dict[str, Any]] = []
        for key, block in sorted(self.embeddings.items()):
            entity_ids = self._known_entity_ids(block.entity_type)
            missing = sorted(set(block.entity_ids) - entity_ids)
            if missing:
                rows.append(_audit_row("embedding", key, "embedding_ids_missing_from_entity_table", missing))

        for name, relation in sorted(self.relations.items()):
            source_known = self._known_entity_ids(relation.source_type)
            target_known = self._known_entity_ids(relation.target_type)
            source_missing = sorted(set(relation.frame[relation.source_col].astype(str)) - source_known)
            target_missing = sorted(set(relation.frame[relation.target_col].astype(str)) - target_known)
            if source_missing:
                rows.append(_audit_row("relation", name, "missing_source_ids", source_missing))
            if target_missing:
                rows.append(_audit_row("relation", name, "missing_target_ids", target_missing))
        return pd.DataFrame(rows, columns=["kind", "name", "issue", "count", "examples"])

    def write(self, path: str | Path) -> Path:
        """Write this store to a ``.emstore`` directory."""
        root = Path(path)
        if root.suffix != ".emstore":
            raise ValueError(f"embedding store: path must end with '.emstore', got {root}.")
        if root.exists() and not root.is_dir():
            raise ValueError(f"embedding store: path exists and is not a directory: {root}.")
        (root / "entities").mkdir(parents=True, exist_ok=True)
        (root / "embeddings").mkdir(parents=True, exist_ok=True)
        (root / "relations").mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "format": "embpy.emstore",
            "version": 1,
            "entities": {},
            "embeddings": {},
            "relations": {},
        }

        for entity_type, frame in sorted(self.entities.items()):
            id_scheme = self.entity_id_schemes.get(entity_type) or str(frame.index.name or "entity_id")
            out = frame.reset_index(names=id_scheme)
            out.to_parquet(root / "entities" / f"{entity_type}.parquet", index=False)
            manifest["entities"][entity_type] = {"id_column": id_scheme, "id_scheme": id_scheme, "n": int(len(frame))}

        used_dirs: set[str] = set()
        for key, block in sorted(self.embeddings.items()):
            dirname = _unique_name(_safe_name(key), used_dirs)
            used_dirs.add(dirname)
            emb_dir = root / "embeddings" / dirname
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_dir / "matrix.npy", np.asarray(block.matrix, dtype=np.float32))
            index = _aliases_to_frame(block.entity_ids, block.aliases, index_name=block.id_scheme).reset_index()
            index.to_parquet(emb_dir / "index.parquet", index=False)
            metadata = {
                "key": key,
                "entity_type": block.entity_type,
                "id_scheme": block.id_scheme,
                "n_entities": block.n_entities,
                "n_dims": block.n_dims,
                "provenance": dict(block.provenance),
            }
            (emb_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
            manifest["embeddings"][key] = {
                "dir": dirname,
                "entity_type": block.entity_type,
                "id_scheme": block.id_scheme,
                "n_entities": block.n_entities,
                "n_dims": block.n_dims,
            }

        for name, relation in sorted(self.relations.items()):
            relation.frame.to_parquet(root / "relations" / f"{name}.parquet", index=False)
            manifest["relations"][name] = {
                "source_type": relation.source_type,
                "target_type": relation.target_type,
                "source_col": relation.source_col,
                "target_col": relation.target_col,
                "n_edges": relation.n_edges,
            }

        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
        return root

    def _ensure_entity_rows(
        self,
        entity_type: str,
        entity_ids: Sequence[str],
        id_scheme: str,
        aliases: Mapping[str, Mapping[str, str]] | None,
    ) -> None:
        alias_frame = _aliases_to_frame(tuple(entity_ids), aliases, index_name=id_scheme)
        self.add_entities(entity_type, alias_frame.reset_index(), id_column=id_scheme, id_scheme=id_scheme)

    def _known_entity_ids(self, entity_type: str) -> set[str]:
        ids: set[str] = set()
        if entity_type in self.entities:
            ids.update(str(x) for x in self.entities[entity_type].index)
        for block in self.embeddings.values():
            if block.entity_type == entity_type:
                ids.update(block.entity_ids)
        return ids


def _aliases_to_frame(
    entity_ids: Sequence[str],
    aliases: Mapping[str, Mapping[str, str]] | None,
    *,
    index_name: str,
) -> pd.DataFrame:
    index = pd.Index(entity_ids, name=index_name)
    aliases = _normalize_aliases(aliases)
    schemes = sorted({scheme for mapping in aliases.values() for scheme in mapping})
    if not schemes:
        return pd.DataFrame(index=index)
    data = {scheme: [aliases.get(eid, {}).get(scheme) for eid in entity_ids] for scheme in schemes}
    return pd.DataFrame(data, index=index)


def _frame_to_aliases(frame: pd.DataFrame, *, id_column: str) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    alias_cols = [c for c in frame.columns if c != id_column]
    if not alias_cols:
        return aliases
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        eid = str(row_dict[id_column])
        mapping = {str(col): str(row_dict[col]) for col in alias_cols if pd.notna(row_dict[col])}
        if mapping:
            aliases[eid] = mapping
    return aliases


def _normalize_aliases(aliases: Mapping[str, Mapping[str, str]] | None) -> dict[str, dict[str, str]]:
    if aliases is None:
        return {}
    return {str(entity_id): {str(k): str(v) for k, v in mapping.items()} for entity_id, mapping in aliases.items()}


def _entity_ids_from_frame(frame: pd.DataFrame, id_column: str | None) -> tuple[list[str], pd.DataFrame, str]:
    out = frame.copy()
    if id_column is not None:
        if id_column not in out.columns:
            raise ValueError(
                f"embedding store: id_column {id_column!r} not found in entity table; "
                f"available columns: {list(out.columns)}."
            )
        return out[id_column].astype(str).tolist(), out, id_column
    if out.index.name:
        column = str(out.index.name)
        out = out.reset_index()
        return out[column].astype(str).tolist(), out, column
    if "id" in out.columns:
        return out["id"].astype(str).tolist(), out, "id"
    if len(out.columns) == 1:
        column = str(out.columns[0])
        return out[column].astype(str).tolist(), out, column
    raise ValueError(
        f"embedding store: entity table has multiple columns and no id_column. Available columns: {list(out.columns)}."
    )


def _default_result_key(result: EmbeddingResult) -> str:
    parts = [result.entity_type, result.provenance.model]
    if result.provenance.pooling:
        parts.append(f"pool_{result.provenance.pooling}")
    if result.provenance.layer is not None:
        parts.append(f"layer_{result.provenance.layer}")
    return ":".join(str(p) for p in parts)


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return clean or "embedding"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def _first_duplicate(ids: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            return item
        seen.add(item)
    return None


def _audit_row(kind: str, name: str, issue: str, values: Sequence[str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "issue": issue,
        "count": len(values),
        "examples": list(values[:5]),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


__all__ = ["EmbeddingBlock", "EmbeddingStore", "RelationTable"]
