"""Static embedding package layout and local query helpers.

This module owns the on-disk layout used for static lookup embeddings:

    manifest.json
    embeddings/<model_key>/
      values.zarr/
      metadata/
        index.parquet
        index.csv
        metadata.json
        uns.json

Only the dense embedding matrix lives in ``values.zarr``. Row identifiers,
source provenance, and AnnData-like ``uns`` metadata live under
``metadata/`` so the structure is easy to inspect and can be uploaded as a
plain Hugging Face dataset folder.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "embpy.static_embedding_package.v1"
VALUES_STORE_NAME = "values.zarr"
METADATA_DIR_NAME = "metadata"
MATRIX_KEY = "matrix"

DuplicatePolicy = Literal["error", "first"]
MissingPolicy = Literal["raise", "drop", "nan"]

_TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".parquet"})
_GENE_SYMBOL_WITH_ENTREZ = re.compile(r"^(.+?)\s*\(\d+\)\s*$")


@dataclass(frozen=True, slots=True)
class StaticEmbeddingSource:
    """A local source table that can be converted into a static package."""

    key: str
    path: Path
    entity_type: str = "gene"
    id_type: str | None = None
    sep: str | None = None
    transpose: bool = False
    id_regex: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StaticEmbeddingTable:
    """In-memory representation of one static embedding matrix."""

    key: str
    matrix: np.ndarray
    entity_ids: tuple[str, ...]
    entity_type: str
    id_type: str
    source_path: Path
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    n_duplicate_input_ids: int = 0
    n_missing_input_ids: int = 0

    @property
    def n_entities(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.matrix.shape[1])


@dataclass(frozen=True, slots=True)
class StaticEmbeddingValidation:
    """Summary returned by package validation."""

    key: str
    path: Path
    n_entities: int
    n_dims: int
    id_type: str


_DEFAULT_SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "genept/embeddings_3072.csv": {
        "key": "genept",
        "id_type": "ensembl_id",
        "description": "GenePT GPT-3.5 text embedding, 3072d, Ensembl-keyed.",
    },
    "genept/scaled/embeddings_3072.csv": {
        "key": "genept_scaled",
        "id_type": "symbol",
        "description": "GenePT GPT-3.5 text embedding, z-scored, 3072d.",
    },
    "gene2vec/embeddings_d200.csv": {
        "key": "gene2vec",
        "id_type": "ensembl_id",
        "description": "Gene2Vec co-expression embedding, 200d.",
    },
    "wikicrow/scaled/embeddings_4096.csv": {
        "key": "wikicrow",
        "id_type": "symbol",
        "description": "WikiCrow text embedding, scaled, 4096d.",
    },
    "omics/embeddings_d256.tsv": {
        "key": "omics",
        "id_type": "ensembl_id",
        "description": "Omics 256d static gene embedding, Ensembl-keyed.",
    },
    "pops/features_d256.tsv": {
        "key": "pops",
        "id_type": "ensembl_id",
        "description": "PoPS 256d gene features, Ensembl-keyed.",
    },
    "crispr_gene_effect/gene_effect.csv": {
        "key": "crispr_gene_effect",
        "id_type": "symbol",
        "transpose": True,
        "id_regex": _GENE_SYMBOL_WITH_ENTREZ.pattern,
        "description": "DepMap CRISPR gene effect matrix, genes as rows after transposition.",
    },
    "crispr_gene_effect/scaled/gene_effect_1178.csv": {
        "key": "crispr_gene_effect_1178",
        "id_type": "symbol",
        "description": "DepMap CRISPR gene effect embedding, scaled, 1178d.",
    },
    "crispr_gene_effect/scaled/gene_effect_205.csv": {
        "key": "crispr_gene_effect_205",
        "id_type": "symbol",
        "description": "DepMap CRISPR gene effect embedding, scaled, 205d.",
    },
}


def discover_static_embedding_sources(
    input_dir: str | Path,
    *,
    include_unknown: bool = True,
) -> list[StaticEmbeddingSource]:
    """Discover supported local static embedding tables under ``input_dir``.

    Known files get stable model keys matching ``BioEmbedder`` static model
    names. Additional ``.csv``, ``.tsv``, and ``.parquet`` files are included
    with a path-derived key when ``include_unknown=True``.
    """

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Static embedding input directory not found: {root}")

    sources: list[StaticEmbeddingSource] = []
    known_paths: set[Path] = set()
    for rel, spec in sorted(_DEFAULT_SOURCE_SPECS.items()):
        path = root / rel
        known_paths.add(path.resolve())
        if path.is_file():
            sources.append(_source_from_spec(path, spec))

    if include_unknown:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TABULAR_SUFFIXES:
                continue
            if path.resolve() in known_paths:
                continue
            if _has_emstore_parent(path) or path.name.endswith(".meta.json"):
                continue
            key = _sanitize_key(path.relative_to(root).with_suffix("").as_posix())
            sources.append(
                StaticEmbeddingSource(
                    key=key,
                    path=path,
                    metadata={"discovery": "path-derived"},
                )
            )

    return sorted(sources, key=lambda source: source.key)


def read_static_embedding_table(
    source: StaticEmbeddingSource,
    *,
    duplicate_policy: DuplicatePolicy = "error",
    drop_missing_ids: bool = False,
) -> StaticEmbeddingTable:
    """Read and validate one source table as ``float32`` embeddings."""

    duplicate_policy = _validate_duplicate_policy(duplicate_policy)
    path = Path(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"Static embedding source file not found for {source.key!r}: {path}")

    logger.info("Reading static embedding source key=%s path=%s", source.key, path)
    frame = _read_table_frame(path, sep=source.sep)
    if frame.empty:
        raise ValueError(f"Static embedding source {path} is empty.")
    if source.transpose:
        frame = frame.T

    ids, keep_mask, n_missing_ids = _clean_identifiers(
        frame.index.tolist(),
        regex=source.id_regex,
        drop_missing=drop_missing_ids,
        key=source.key,
    )
    if n_missing_ids:
        frame = frame.iloc[keep_mask]

    matrix_frame = frame.copy()
    try:
        matrix = matrix_frame.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Static embedding source {path} contains non-numeric embedding values. "
            "Expected one identifier column/index and numeric dimensions only."
        ) from exc

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"Static embedding source {path} must be a 2D non-empty matrix, got {matrix.shape!r}.")
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"Static embedding source {path} has {matrix.shape[0]} rows but {len(ids)} identifiers."
        )
    if not np.isfinite(matrix).all():
        bad = np.where(~np.isfinite(matrix).all(axis=1))[0]
        first = int(bad[0])
        raise ValueError(
            f"Static embedding source {path} contains NaN/Inf values; "
            f"first bad row={first} id={ids[first]!r}."
        )

    ids, matrix, n_duplicates = _handle_duplicate_ids(ids, matrix, policy=duplicate_policy, key=source.key)
    id_type = source.id_type or _infer_gene_id_type(ids)

    return StaticEmbeddingTable(
        key=source.key,
        matrix=np.ascontiguousarray(matrix, dtype=np.float32),
        entity_ids=tuple(ids),
        entity_type=source.entity_type,
        id_type=id_type,
        source_path=path,
        source_metadata=dict(source.metadata),
        description=source.description,
        n_duplicate_input_ids=n_duplicates,
        n_missing_input_ids=n_missing_ids,
    )


def prepare_static_embedding_package(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    keys: Sequence[str] | None = None,
    include_unknown: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    duplicate_policy: DuplicatePolicy = "error",
    drop_missing_ids: bool = False,
) -> dict[str, Any]:
    """Prepare a local static embedding package.

    ``dry_run=True`` performs discovery only and returns the manifest that
    would be written.
    """

    sources = discover_static_embedding_sources(input_dir, include_unknown=include_unknown)
    if keys is not None:
        wanted = set(keys)
        sources = [source for source in sources if source.key in wanted]
        missing = sorted(wanted - {source.key for source in sources})
        if missing:
            raise KeyError(f"Requested static embedding key(s) not found under {input_dir}: {missing}")

    package_root = Path(output_dir)
    manifest = _base_manifest(package_root=package_root, input_dir=Path(input_dir), dry_run=dry_run)
    manifest["planned_embeddings"] = [_source_manifest_entry(source) for source in sources]

    unsupported = _discover_unsupported_sources(Path(input_dir))
    if unsupported:
        manifest["unsupported_sources"] = unsupported
        logger.warning("Found %d unsupported source artifact(s); they are listed in manifest.", len(unsupported))

    if dry_run:
        logger.info("Dry run: discovered %d static embedding source(s); no files written.", len(sources))
        return manifest

    package_root.mkdir(parents=True, exist_ok=True)
    manifest_path = package_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Pass overwrite=True to replace it.")

    embeddings: dict[str, Any] = {}
    for source in sources:
        table = read_static_embedding_table(
            source,
            duplicate_policy=duplicate_policy,
            drop_missing_ids=drop_missing_ids,
        )
        entry = write_static_embedding_package(table, package_root, overwrite=overwrite)
        validate_static_embedding_dir(package_root / "embeddings" / table.key)
        embeddings[table.key] = entry
        logger.info(
            "Packaged static embedding key=%s n_entities=%d n_dims=%d",
            table.key,
            table.n_entities,
            table.n_dims,
        )

    manifest["embeddings"] = embeddings
    manifest["n_embeddings"] = len(embeddings)
    manifest["generated_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    logger.info("Wrote static embedding package manifest: %s", manifest_path)
    return manifest


def write_static_embedding_package(
    table: StaticEmbeddingTable,
    package_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one table into ``package_root/embeddings/<key>/``."""

    root = Path(package_root)
    model_dir = root / "embeddings" / table.key
    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Static embedding package already exists: {model_dir}. Pass overwrite=True.")
        shutil.rmtree(model_dir)

    values_path = model_dir / VALUES_STORE_NAME
    metadata_dir = model_dir / METADATA_DIR_NAME
    metadata_dir.mkdir(parents=True, exist_ok=True)

    _write_values_zarr(values_path, table.matrix)

    index = pd.DataFrame({"entity_id": list(table.entity_ids)})
    index.to_parquet(metadata_dir / "index.parquet", index=False)
    index.to_csv(metadata_dir / "index.csv", index=False)

    metadata = _metadata_for_table(table, model_dir=model_dir)
    uns = _uns_for_table(table, metadata=metadata)
    _write_json(metadata_dir / "metadata.json", metadata)
    _write_json(metadata_dir / "uns.json", uns)

    return _manifest_entry_for_table(table, model_dir=model_dir)


def validate_static_embedding_package(path: str | Path) -> list[StaticEmbeddingValidation]:
    """Validate a package root or one ``embeddings/<key>`` directory."""

    root = Path(path)
    if _is_model_dir(root):
        return [validate_static_embedding_dir(root)]

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        keys = sorted((manifest.get("embeddings") or {}).keys())
    else:
        embeddings_dir = root / "embeddings"
        keys = sorted(p.name for p in embeddings_dir.iterdir() if _is_model_dir(p)) if embeddings_dir.is_dir() else []

    if not keys:
        raise FileNotFoundError(f"No static embedding package entries found under {root}.")

    return [validate_static_embedding_dir(root / "embeddings" / key) for key in keys]


def validate_static_embedding_dir(path: str | Path) -> StaticEmbeddingValidation:
    """Validate one ``embeddings/<key>`` package directory."""

    store = StaticEmbeddingStore.open(path)
    shape = tuple(int(x) for x in store.matrix_array.shape)
    expected_shape = tuple(int(x) for x in store.metadata.get("shape", []))
    if expected_shape and shape != expected_shape:
        raise ValueError(f"Zarr shape mismatch for {store.path}: values.zarr={shape}, metadata={expected_shape}.")
    if shape[0] != len(store.entity_ids):
        raise ValueError(
            f"Index length mismatch for {store.path}: values.zarr has {shape[0]} rows, "
            f"metadata index has {len(store.entity_ids)} rows."
        )
    if len(set(store.entity_ids)) != len(store.entity_ids):
        raise ValueError(f"Metadata index has duplicate entity_id values: {store.path}")
    if shape[0] and shape[1]:
        first = np.asarray(store.matrix_array[0, :], dtype=np.float32)
        if first.shape[0] != shape[1] or not np.isfinite(first).all():
            raise ValueError(f"Could not read a finite first row from {store.path / VALUES_STORE_NAME}.")
    return StaticEmbeddingValidation(
        key=store.key,
        path=store.path,
        n_entities=shape[0],
        n_dims=shape[1],
        id_type=store.id_type,
    )


def load_static_embedding_package(path: str | Path, *, key: str | None = None) -> "StaticEmbeddingStore":
    """Open a packaged static embedding for exact identifier lookup."""

    root = Path(path)
    if key is not None:
        model_dir = root / "embeddings" / key
        if not model_dir.is_dir():
            model_dir = root / key
        return StaticEmbeddingStore.open(model_dir)
    if _is_model_dir(root):
        return StaticEmbeddingStore.open(root)

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected a model package directory or manifest.json under {root}.")
    manifest = json.loads(manifest_path.read_text())
    keys = sorted((manifest.get("embeddings") or {}).keys())
    if len(keys) != 1:
        raise ValueError(f"Package {root} contains {len(keys)} embeddings; pass key=... to select one.")
    return StaticEmbeddingStore.open(root / "embeddings" / keys[0])


class StaticEmbeddingStore:
    """Local exact-lookup reader for a packaged static embedding."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any],
        uns: Mapping[str, Any],
        index: pd.DataFrame,
    ) -> None:
        self.path = path
        self.metadata = dict(metadata)
        self.uns = dict(uns)
        self.index = pd.DataFrame(index)
        if "entity_id" not in self.index:
            raise KeyError(f"Static embedding index is missing required column 'entity_id': {path}")
        self.entity_ids = tuple(self.index["entity_id"].astype(str).tolist())
        duplicate = _first_duplicate(self.entity_ids)
        if duplicate is not None:
            raise ValueError(f"Static embedding index contains duplicate entity_id={duplicate!r}: {path}")
        self._row_by_id = {entity_id: i for i, entity_id in enumerate(self.entity_ids)}
        self._matrix_array: Any | None = None

    @classmethod
    def open(cls, path: str | Path) -> "StaticEmbeddingStore":
        model_dir = Path(path)
        metadata_dir = model_dir / METADATA_DIR_NAME
        metadata_path = metadata_dir / "metadata.json"
        uns_path = metadata_dir / "uns.json"
        index_path = metadata_dir / "index.parquet"

        if not (model_dir / VALUES_STORE_NAME).is_dir():
            raise FileNotFoundError(f"Missing Zarr values store: {model_dir / VALUES_STORE_NAME}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        if not uns_path.is_file():
            raise FileNotFoundError(f"Missing uns metadata file: {uns_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing metadata index file: {index_path}")

        return cls(
            model_dir,
            metadata=json.loads(metadata_path.read_text()),
            uns=json.loads(uns_path.read_text()),
            index=pd.read_parquet(index_path),
        )

    @property
    def key(self) -> str:
        return str(self.metadata.get("key") or self.path.name)

    @property
    def entity_type(self) -> str:
        return str(self.metadata.get("entity_type", "gene"))

    @property
    def id_type(self) -> str:
        return str(self.metadata.get("id_type", "symbol"))

    @property
    def n_entities(self) -> int:
        return int(self.metadata.get("n_entities", len(self.entity_ids)))

    @property
    def n_dims(self) -> int:
        shape = self.metadata.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return int(shape[1])
        return int(self.matrix_array.shape[1])

    @property
    def matrix_array(self) -> Any:
        if self._matrix_array is None:
            try:
                import zarr
            except ImportError as exc:  # pragma: no cover
                raise ImportError("zarr is required to read static embedding packages.") from exc
            root = zarr.open_group(str(self.path / VALUES_STORE_NAME), mode="r")
            matrix_key = str(self.metadata.get("matrix_key", MATRIX_KEY))
            self._matrix_array = root[matrix_key]
        return self._matrix_array

    def get(self, identifiers: str | Sequence[str], *, missing: MissingPolicy = "raise") -> np.ndarray:
        """Return embeddings for one or more exact row identifiers."""

        wanted, scalar = _normalize_identifier_query(identifiers)
        rows: list[int | None] = [self._row_by_id.get(identifier) for identifier in wanted]
        missing_ids = [identifier for identifier, row in zip(wanted, rows, strict=True) if row is None]
        missing = _validate_missing_policy(missing)
        if missing_ids and missing == "raise":
            preview = missing_ids[:10]
            raise KeyError(
                f"{len(missing_ids)} identifier(s) are not present in static embedding {self.key!r}; "
                f"first missing: {preview}."
            )

        arrays: list[np.ndarray] = []
        for row in rows:
            if row is None:
                if missing == "drop":
                    continue
                arrays.append(np.full((self.n_dims,), np.nan, dtype=np.float32))
                continue
            arrays.append(np.asarray(self.matrix_array[int(row), :], dtype=np.float32))

        if not arrays:
            out = np.zeros((0, self.n_dims), dtype=np.float32)
        else:
            out = np.vstack(arrays).astype(np.float32, copy=False)
        return out[0] if scalar and out.shape[0] == 1 else out

    def query(
        self,
        identifiers: str | Sequence[str],
        *,
        missing: MissingPolicy = "raise",
        as_dataframe: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        """Query identifiers and return a DataFrame by default."""

        wanted, scalar = _normalize_identifier_query(identifiers)
        matrix = self.get(wanted, missing=missing)
        if scalar and matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if not as_dataframe:
            return matrix

        if missing == "drop":
            ids = [identifier for identifier in wanted if identifier in self._row_by_id]
        else:
            ids = wanted
        return pd.DataFrame(
            np.asarray(matrix, dtype=np.float32),
            index=pd.Index(ids, name="entity_id"),
            columns=[f"dim_{i}" for i in range(np.asarray(matrix).shape[1])],
        )

    def to_hf_dict(self) -> dict[str, Any]:
        """Return the dict shape expected by ``HFHandler.download_embedding``."""

        matrix = np.asarray(self.matrix_array[:, :], dtype=np.float32)
        ids = np.asarray(self.entity_ids, dtype=str)
        return {
            "embeddings": matrix,
            "ids": ids,
            "entity_ids": ids,
            "index": self.index.copy(),
            "metadata": dict(self.metadata),
            "uns": dict(self.uns),
            "id_key": "entity_id",
            "id_type": self.id_type,
            "format": "embpy_static_zarr",
        }


def _source_from_spec(path: Path, spec: Mapping[str, Any]) -> StaticEmbeddingSource:
    return StaticEmbeddingSource(
        key=str(spec["key"]),
        path=path,
        entity_type=str(spec.get("entity_type", "gene")),
        id_type=spec.get("id_type"),
        sep=spec.get("sep"),
        transpose=bool(spec.get("transpose", False)),
        id_regex=spec.get("id_regex"),
        description=spec.get("description"),
        metadata={k: v for k, v in spec.items() if k not in {"key", "entity_type", "id_type", "sep"}},
    )


def _read_table_frame(path: Path, *, sep: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.RangeIndex):
            first = str(df.columns[0])
            df = df.set_index(first)
        return pd.DataFrame(df)
    if suffix not in {".csv", ".tsv"}:
        raise ValueError(f"Unsupported static embedding source extension {suffix!r}: {path}")
    resolved_sep = sep if sep is not None else ("\t" if suffix == ".tsv" else ",")
    return pd.read_csv(path, sep=resolved_sep, index_col=0)


def _write_values_zarr(path: Path, matrix: np.ndarray) -> None:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("zarr is required to write static embedding packages.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    chunks = (min(max(matrix.shape[0], 1), 1024), matrix.shape[1])
    root.create_array(MATRIX_KEY, data=np.asarray(matrix, dtype=np.float32), chunks=chunks)
    root.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "matrix_key": MATRIX_KEY,
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "dtype": "float32",
        }
    )


def _metadata_for_table(table: StaticEmbeddingTable, *, model_dir: Path) -> dict[str, Any]:
    source = table.source_path
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "key": table.key,
            "entity_type": table.entity_type,
            "id_type": table.id_type,
            "id_key": "entity_id",
            "n_entities": table.n_entities,
            "n_dims": table.n_dims,
            "shape": [table.n_entities, table.n_dims],
            "dtype": "float32",
            "matrix_key": MATRIX_KEY,
            "values_path": VALUES_STORE_NAME,
            "metadata_path": METADATA_DIR_NAME,
            "index_path": f"{METADATA_DIR_NAME}/index.parquet",
            "uns_path": f"{METADATA_DIR_NAME}/uns.json",
            "description": table.description,
            "source": {
                "path": str(source),
                "name": source.name,
                "suffix": source.suffix,
                "size_bytes": source.stat().st_size if source.exists() else None,
            },
            "source_metadata": dict(table.source_metadata),
            "n_duplicate_input_ids": int(table.n_duplicate_input_ids),
            "n_missing_input_ids": int(table.n_missing_input_ids),
            "created_at": _utc_now(),
            "package_dir": model_dir.as_posix(),
        }
    )


def _uns_for_table(table: StaticEmbeddingTable, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "embpy_static_embedding": {
                "schema_version": SCHEMA_VERSION,
                "model_key": table.key,
                "storage": "zarr",
                "values_path": VALUES_STORE_NAME,
                "matrix_key": MATRIX_KEY,
                "entity_type": table.entity_type,
                "id_type": table.id_type,
                "id_key": "entity_id",
                "n_entities": table.n_entities,
                "n_dims": table.n_dims,
                "description": table.description,
                "source": metadata.get("source", {}),
            }
        }
    )


def _manifest_entry_for_table(table: StaticEmbeddingTable, *, model_dir: Path) -> dict[str, Any]:
    return _json_safe(
        {
            "key": table.key,
            "entity_type": table.entity_type,
            "id_type": table.id_type,
            "n_entities": table.n_entities,
            "n_dims": table.n_dims,
            "shape": [table.n_entities, table.n_dims],
            "values_path": f"embeddings/{table.key}/{VALUES_STORE_NAME}",
            "metadata_path": f"embeddings/{table.key}/{METADATA_DIR_NAME}/metadata.json",
            "index_path": f"embeddings/{table.key}/{METADATA_DIR_NAME}/index.parquet",
            "uns_path": f"embeddings/{table.key}/{METADATA_DIR_NAME}/uns.json",
            "description": table.description,
            "source_path": str(table.source_path),
            "n_missing_input_ids": int(table.n_missing_input_ids),
            "n_duplicate_input_ids": int(table.n_duplicate_input_ids),
            "package_dir": model_dir.as_posix(),
        }
    )


def _base_manifest(*, package_root: Path, input_dir: Path, dry_run: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layout": "single_repo_embeddings_folder_v1",
        "package_root": str(package_root),
        "input_dir": str(input_dir),
        "dry_run": bool(dry_run),
        "generated_at": _utc_now(),
        "embeddings": {},
        "n_embeddings": 0,
        "unsupported_sources": [],
    }


def _source_manifest_entry(source: StaticEmbeddingSource) -> dict[str, Any]:
    return _json_safe(
        {
            "key": source.key,
            "path": str(source.path),
            "entity_type": source.entity_type,
            "id_type": source.id_type,
            "transpose": source.transpose,
            "description": source.description,
        }
    )


def _discover_unsupported_sources(input_dir: Path) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TABULAR_SUFFIXES or _has_emstore_parent(path):
            continue
        if suffix in {".zip", ".h5", ".hdf5"}:
            unsupported.append(
                {
                    "path": str(path),
                    "suffix": suffix,
                    "reason": "not a single gene-by-dimension table; use an explicit converter before packaging",
                }
            )
    return _summarize_unsupported_sources(unsupported)


def _handle_duplicate_ids(
    ids: list[str],
    matrix: np.ndarray,
    *,
    policy: DuplicatePolicy,
    key: str,
) -> tuple[list[str], np.ndarray, int]:
    seen: set[str] = set()
    keep: list[bool] = []
    duplicates = 0
    first_duplicate: str | None = None
    for entity_id in ids:
        is_duplicate = entity_id in seen
        if is_duplicate:
            duplicates += 1
            first_duplicate = first_duplicate or entity_id
        keep.append(not is_duplicate)
        seen.add(entity_id)

    if duplicates == 0:
        return ids, matrix, 0
    if policy == "error":
        raise ValueError(
            f"Static embedding source for key={key!r} contains duplicate identifiers; "
            f"first duplicate={first_duplicate!r}. Pass duplicate_policy='first' to keep first occurrence."
        )

    mask = np.asarray(keep, dtype=bool)
    deduped_ids = [entity_id for entity_id, should_keep in zip(ids, keep, strict=True) if should_keep]
    logger.warning("Dropped %d duplicate identifier row(s) for key=%s.", duplicates, key)
    return deduped_ids, np.asarray(matrix[mask], dtype=np.float32), duplicates


def _clean_identifiers(
    values: Sequence[object],
    *,
    regex: str | None,
    drop_missing: bool,
    key: str,
) -> tuple[list[str], list[int], int]:
    ids: list[str] = []
    keep: list[int] = []
    missing = 0
    first_missing: int | None = None
    for i, value in enumerate(values):
        if _is_missing_identifier(value):
            missing += 1
            first_missing = first_missing if first_missing is not None else i
            if drop_missing:
                continue
        ids.append(_clean_identifier(value, regex=regex))
        keep.append(i)
    if missing and not drop_missing:
        raise ValueError(
            f"Static embedding source for key={key!r} contains {missing} missing/blank identifier row(s); "
            f"first missing row={first_missing}. Pass drop_missing_ids=True to discard those rows."
        )
    if missing:
        logger.warning("Dropped %d missing/blank identifier row(s) for key=%s.", missing, key)
    return ids, keep, missing


def _clean_identifier(value: object, *, regex: str | None) -> str:
    text = str(value).strip()
    if regex:
        match = re.match(regex, text)
        if match:
            return match.group(1).strip()
    return text


def _is_missing_identifier(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _infer_gene_id_type(ids: Sequence[str]) -> str:
    sample = [str(x).upper() for x in ids[:50]]
    if sample and all(x.startswith(("ENSG", "ENSMUSG", "ENS")) for x in sample):
        return "ensembl_id"
    return "symbol"


def _normalize_identifier_query(value: str | Sequence[str]) -> tuple[list[str], bool]:
    if isinstance(value, str):
        return [value], True
    identifiers = [str(x) for x in value]
    if not identifiers:
        raise ValueError("Static embedding query identifiers cannot be empty.")
    return identifiers, False


def _validate_duplicate_policy(value: str) -> DuplicatePolicy:
    if value not in ("error", "first"):
        raise ValueError(f"duplicate_policy must be 'error' or 'first', got {value!r}.")
    return value  # type: ignore[return-value]


def _validate_missing_policy(value: str) -> MissingPolicy:
    if value not in ("raise", "drop", "nan"):
        raise ValueError(f"missing must be 'raise', 'drop', or 'nan', got {value!r}.")
    return value  # type: ignore[return-value]


def _is_model_dir(path: Path) -> bool:
    return (path / VALUES_STORE_NAME).is_dir() and (path / METADATA_DIR_NAME / "metadata.json").is_file()


def _has_emstore_parent(path: Path) -> bool:
    return any(part.endswith(".emstore") for part in path.parts)


def _summarize_unsupported_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h5_by_parent: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for item in items:
        path = Path(str(item["path"]))
        if item.get("suffix") in {".h5", ".hdf5"}:
            h5_by_parent[str(path.parent)] = h5_by_parent.get(str(path.parent), 0) + 1
            continue
        out.append(item)
    for parent, count in sorted(h5_by_parent.items()):
        out.append(
            {
                "path": parent,
                "suffix": ".h5",
                "count": count,
                "reason": "directory of HDF5 files; use an explicit converter before packaging",
            }
        )
    return out


def _sanitize_key(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    return clean or "static_embedding"


def _first_duplicate(ids: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for entity_id in ids:
        if entity_id in seen:
            return entity_id
        seen.add(entity_id)
    return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True))


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, default=str))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "SCHEMA_VERSION",
    "StaticEmbeddingSource",
    "StaticEmbeddingStore",
    "StaticEmbeddingTable",
    "StaticEmbeddingValidation",
    "discover_static_embedding_sources",
    "load_static_embedding_package",
    "prepare_static_embedding_package",
    "read_static_embedding_table",
    "validate_static_embedding_dir",
    "validate_static_embedding_package",
    "write_static_embedding_package",
]
