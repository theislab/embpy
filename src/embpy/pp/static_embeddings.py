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
DATASET_CARD_NAME = "README.md"
DEFAULT_SPECIES = "human"
DEFAULT_TAXONOMY_ID = "9606"

DuplicatePolicy = Literal["error", "first"]
NanPolicy = Literal["error", "drop-rows", "fill-zero"]
MissingPolicy = Literal["raise", "drop", "nan"]
TargetIdType = Literal["source", "ensembl_id"]
UnresolvedIdPolicy = Literal["drop", "error"]

_TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".parquet"})
_COMPRESSED_TABULAR_SUFFIXES = frozenset({".csv.gz", ".tsv.gz", ".txt.gz"})
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5"})
_GENE_SYMBOL_WITH_ENTREZ = re.compile(r"^(.+?)\s*\(\d+\)\s*$")
_TAXONOMY_SPECIES = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "drosophila",
    "6239": "c_elegans",
    "4932": "yeast",
    "3702": "arabidopsis",
}
_SPECIES_TAXONOMY = {
    "human": "9606",
    "homo_sapiens": "9606",
    "mouse": "10090",
    "mus_musculus": "10090",
    "rat": "10116",
    "rattus_norvegicus": "10116",
    "zebrafish": "7955",
    "danio_rerio": "7955",
    "drosophila": "7227",
    "drosophila_melanogaster": "7227",
    "c_elegans": "6239",
    "caenorhabditis_elegans": "6239",
    "yeast": "4932",
    "saccharomyces_cerevisiae": "4932",
    "arabidopsis": "3702",
    "arabidopsis_thaliana": "3702",
}


@dataclass(frozen=True, slots=True)
class StaticEmbeddingSource:
    """A local source table that can be converted into a static package."""

    key: str
    path: Path
    entity_type: str = "gene"
    id_type: str | None = None
    species: str | None = None
    taxonomy_id: str | int | None = None
    sep: str | None = None
    transpose: bool = False
    id_column: str | int | None = None
    embedding_columns: Sequence[str | int] | None = None
    metadata_columns: Sequence[str | int] = field(default_factory=tuple)
    skip_rows: int = 0
    header: int | None = 0
    comment: str | None = None
    na_values: Sequence[str] = field(default_factory=tuple)
    h5_embedding_dataset: str = "embeddings"
    h5_id_dataset: str = "proteins"
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
    source_id_type: str | None = None
    target_id_type: str | None = None
    organism: str | None = None
    species: str = DEFAULT_SPECIES
    taxonomy_id: str | None = DEFAULT_TAXONOMY_ID
    species_key: str = f"{DEFAULT_SPECIES}_{DEFAULT_TAXONOMY_ID}"
    id_harmonized: bool = False
    alias_columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    n_duplicate_input_ids: int = 0
    n_missing_input_ids: int = 0
    n_nan_input_values: int = 0
    n_nan_input_rows: int = 0
    n_nan_input_columns: int = 0
    nan_policy: str = "error"
    n_unresolved_harmonization_ids: int = 0
    n_duplicate_harmonized_ids: int = 0
    unresolved_id_policy: str = "drop"

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
    "precomputed_embeddings_string/functional_embeddings/functional_emb/9606.h5": {
        "key": "string_functional_9606",
        "entity_type": "protein",
        "id_type": "string_protein_id",
        "species": "human",
        "taxonomy_id": "9606",
        "h5_embedding_dataset": "embeddings",
        "h5_id_dataset": "proteins",
        "description": "STRING/SPACE functional PPI embedding for human proteins, species 9606, 512d.",
    },
    "precomputed_embeddings_string/node2vec/node2vec/9606.h5": {
        "key": "string_node2vec_9606",
        "entity_type": "protein",
        "id_type": "string_protein_id",
        "species": "human",
        "taxonomy_id": "9606",
        "h5_embedding_dataset": "embeddings",
        "h5_id_dataset": "proteins",
        "description": "STRING node2vec PPI embedding for human proteins, species 9606, 128d.",
    },
}

_STRING_SOURCE_TEMPLATES: dict[str, dict[str, Any]] = {
    "functional": {
        "rel_template": "precomputed_embeddings_string/functional_embeddings/functional_emb/{species}.h5",
        "key_template": "string_functional_{species}",
        "description_template": "STRING/SPACE functional PPI embedding for proteins, species {species}, 512d.",
    },
    "node2vec": {
        "rel_template": "precomputed_embeddings_string/node2vec/node2vec/{species}.h5",
        "key_template": "string_node2vec_{species}",
        "description_template": "STRING node2vec PPI embedding for proteins, species {species}, 128d.",
    },
}


def static_embedding_keys(entity_type: str = "gene") -> frozenset[str]:
    """Static-embedding keys for one entity type, derived from the specs.

    Taken from :data:`_DEFAULT_SOURCE_SPECS` so the advertised roster and the
    download specification cannot drift apart. ``embpy.embedder`` uses this to
    build ``DEFAULT_STATIC_EMBEDDING_MODELS``, which previously listed ``ccle``
    and ``ccle_ensembl`` -- keys with no specification behind them, so ``embed``
    raised ``FileNotFoundError`` for anything the catalogue happily advertised.

    A spec with no ``entity_type`` is a gene table; the two STRING tables declare
    ``entity_type="protein"`` and are keyed by STRING protein ids, so they must
    *not* be offered as gene lookups. There is currently no protein-side static
    routing, which is why asking for them at all is still a dead end -- but a dead
    end is better than silently resolving 19,000 ``9606.ENSP...`` ids into gene
    symbols one HTTP call at a time.

    A key being listed here means embpy knows which file to ask for, not that the
    configured repository serves it: ``crispr_gene_effect`` needs the raw DepMap
    matrix, which the public ``theislab/Embpy_Data`` repository does not ship.
    Missing files surface as a ``FileNotFoundError`` naming what *is* available.
    """
    return frozenset(
        spec["key"]
        for spec in _DEFAULT_SOURCE_SPECS.values()
        if spec.get("entity_type", "gene") == entity_type
    )



def discover_static_embedding_sources(
    input_dir: str | Path,
    *,
    include_unknown: bool = True,
    string_species: Sequence[str | int] | None = None,
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

    for source in _string_sources_for_species(root, string_species or ()):
        resolved = source.path.resolve()
        if resolved in known_paths:
            continue
        known_paths.add(resolved)
        sources.append(source)

    if include_unknown:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not _is_auto_discoverable_source_table(path):
                continue
            if path.resolve() in known_paths:
                continue
            if _has_emstore_parent(path) or _has_static_package_parent(path) or path.name.endswith(".meta.json"):
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


def load_static_embedding_source_config(
    config_path: str | Path,
    *,
    input_dir: str | Path | None = None,
) -> list[StaticEmbeddingSource]:
    """Load explicit source descriptions from a JSON config file.

    The config may be either ``{"sources": [...]}``, a list of source objects,
    or one source object. Relative source paths are resolved relative to
    ``input_dir`` when provided, otherwise relative to the config file.
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Static embedding source config not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Source config {path} is not valid JSON: {exc}") from exc

    items = payload.get("sources") if isinstance(payload, dict) and "sources" in payload else payload
    if isinstance(items, Mapping):
        items = [items]
    if not isinstance(items, list) or not items:
        raise ValueError(
            f"Source config {path} must contain a non-empty source object, source list, or "
            "an object with a non-empty 'sources' list."
        )

    base = Path(input_dir) if input_dir is not None else path.parent
    sources = [_source_from_config_item(item, base_dir=base, config_path=path) for item in items]
    keys = [source.key for source in sources]
    duplicate = _first_duplicate(keys)
    if duplicate is not None:
        raise ValueError(f"Source config {path} contains duplicate source key={duplicate!r}.")
    return sorted(sources, key=lambda source: source.key)


def read_static_embedding_table(
    source: StaticEmbeddingSource,
    *,
    duplicate_policy: DuplicatePolicy = "error",
    drop_missing_ids: bool = False,
    nan_policy: NanPolicy = "error",
    harmonize_ids: bool = True,
    target_id_type: TargetIdType = "ensembl_id",
    organism: str | None = None,
    unresolved_id_policy: UnresolvedIdPolicy = "drop",
    gene_resolver: Any | None = None,
) -> StaticEmbeddingTable:
    """Read and validate one source table as ``float32`` embeddings."""

    duplicate_policy = _validate_duplicate_policy(duplicate_policy)
    nan_policy = _validate_nan_policy(nan_policy)
    target_id_type = _validate_target_id_type(target_id_type)
    unresolved_id_policy = _validate_unresolved_id_policy(unresolved_id_policy)
    path = Path(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"Static embedding source file not found for {source.key!r}: {path}")

    species, taxonomy_id, species_key = _source_species_info(source)
    resolver_organism = _canonical_species_name(organism) if organism is not None else species
    logger.info("Reading static embedding source key=%s path=%s", source.key, path)
    frame, index_columns = _read_source_frames(source)
    if frame.empty:
        raise ValueError(f"Static embedding source {path} is empty.")

    ids, keep_mask, n_missing_ids = _clean_identifiers(
        frame.index.tolist(),
        regex=source.id_regex,
        drop_missing=drop_missing_ids,
        key=source.key,
    )
    if n_missing_ids:
        frame = frame.iloc[keep_mask]
        index_columns = _subset_index_columns(index_columns, keep_mask)

    matrix_frame = frame.copy()
    matrix = _coerce_embedding_matrix(matrix_frame, ids, source)

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"Static embedding source {path} must be a 2D non-empty matrix, got {matrix.shape!r}.")
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"Static embedding source {path} has {matrix.shape[0]} rows but {len(ids)} identifiers."
        )
    matrix, ids, index_columns, nan_counts = _handle_nan_values(
        matrix,
        ids,
        index_columns=index_columns,
        policy=nan_policy,
        path=path,
        column_names=[str(column) for column in matrix_frame.columns],
    )

    ids, matrix, index_columns, n_duplicates = _handle_duplicate_ids(
        ids,
        matrix,
        index_columns=index_columns,
        policy=duplicate_policy,
        key=source.key,
    )
    id_type = _normalize_source_id_type(source.id_type) or _infer_gene_id_type(ids)
    source_id_type = id_type
    alias_columns: Mapping[str, tuple[str, ...]] = {
        column: tuple(values) for column, values in index_columns.items()
    }
    n_unresolved_harmonization_ids = 0
    n_duplicate_harmonized_ids = 0
    id_harmonized = False

    if harmonize_ids and source.entity_type == "gene" and target_id_type == "ensembl_id":
        (
            ids,
            matrix,
            alias_columns,
            harmonization_counts,
            id_harmonized,
        ) = _harmonize_gene_ids_to_ensembl(
            ids,
            matrix,
            source_id_type=source_id_type,
            organism=resolver_organism,
            unresolved_policy=unresolved_id_policy,
            gene_resolver=gene_resolver,
            index_columns=index_columns,
            key=source.key,
        )
        id_type = "ensembl_id"
        n_unresolved_harmonization_ids = harmonization_counts["unresolved"]
        n_duplicate_harmonized_ids = harmonization_counts["duplicates"]
    elif harmonize_ids and source.entity_type != "gene":
        logger.info(
            "Skipping Ensembl ID harmonization for key=%s because entity_type=%s is not gene.",
            source.key,
            source.entity_type,
        )
        target_id_type = "source"

    return StaticEmbeddingTable(
        key=source.key,
        matrix=np.ascontiguousarray(matrix, dtype=np.float32),
        entity_ids=tuple(ids),
        entity_type=source.entity_type,
        id_type=id_type,
        source_path=path,
        source_id_type=source_id_type,
        target_id_type=target_id_type,
        organism=resolver_organism,
        species=species,
        taxonomy_id=taxonomy_id,
        species_key=species_key,
        id_harmonized=id_harmonized,
        alias_columns=alias_columns,
        source_metadata=dict(source.metadata),
        description=source.description,
        n_duplicate_input_ids=n_duplicates,
        n_missing_input_ids=n_missing_ids,
        n_nan_input_values=nan_counts["values"],
        n_nan_input_rows=nan_counts["rows"],
        n_nan_input_columns=nan_counts["columns"],
        nan_policy=nan_policy,
        n_unresolved_harmonization_ids=n_unresolved_harmonization_ids,
        n_duplicate_harmonized_ids=n_duplicate_harmonized_ids,
        unresolved_id_policy=unresolved_id_policy,
    )


def prepare_static_embedding_package(
    input_dir: str | Path | None,
    output_dir: str | Path,
    *,
    keys: Sequence[str] | None = None,
    sources: Sequence[StaticEmbeddingSource] | None = None,
    source_config: str | Path | None = None,
    string_species: Sequence[str | int] | None = None,
    include_unknown: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    duplicate_policy: DuplicatePolicy = "error",
    drop_missing_ids: bool = False,
    nan_policy: NanPolicy = "error",
    harmonize_ids: bool = True,
    target_id_type: TargetIdType = "ensembl_id",
    organism: str | None = None,
    unresolved_id_policy: UnresolvedIdPolicy = "drop",
    gene_resolver: Any | None = None,
) -> dict[str, Any]:
    """Prepare a local static embedding package.

    ``dry_run=True`` performs discovery only and returns the manifest that
    would be written.
    """

    target_id_type = _validate_target_id_type(target_id_type)
    unresolved_id_policy = _validate_unresolved_id_policy(unresolved_id_policy)
    input_path = Path(input_dir) if input_dir is not None else None
    if sources is not None and source_config is not None:
        raise ValueError("Pass either explicit sources or source_config, not both.")
    if sources is not None:
        discovered_sources = list(sources)
    elif source_config is not None:
        discovered_sources = load_static_embedding_source_config(source_config, input_dir=input_path)
    elif input_path is not None:
        discovered_sources = discover_static_embedding_sources(
            input_path,
            include_unknown=include_unknown,
            string_species=string_species,
        )
    else:
        raise ValueError("prepare_static_embedding_package requires input_dir, sources, or source_config.")

    if keys is not None:
        wanted = set(keys)
        discovered_sources = [source for source in discovered_sources if source.key in wanted]
        missing = sorted(wanted - {source.key for source in discovered_sources})
        if missing:
            raise KeyError(f"Requested static embedding key(s) not found under {input_dir}: {missing}")

    package_root = Path(output_dir)
    manifest = _base_manifest(package_root=package_root, input_dir=input_path, dry_run=dry_run)
    species_entries = [_source_species_info(source) for source in discovered_sources]
    manifest["id_harmonization"] = {
        "enabled": bool(harmonize_ids),
        "target_id_type": target_id_type,
        "organism": _canonical_species_name(organism) if organism is not None else "source_species",
        "unresolved_id_policy": unresolved_id_policy,
    }
    manifest["species"] = sorted({species for species, _, _ in species_entries})
    manifest["taxonomy_ids"] = sorted({taxonomy_id for _, taxonomy_id, _ in species_entries if taxonomy_id is not None})
    manifest["species_keys"] = sorted({species_key for _, _, species_key in species_entries})
    manifest["string_species"] = [str(species) for species in (string_species or [])]
    if source_config is not None:
        manifest["source_config"] = str(source_config)
    manifest["planned_embeddings"] = [_source_manifest_entry(source) for source in discovered_sources]

    packaged_paths = {Path(source.path).resolve() for source in discovered_sources}
    unsupported = _discover_unsupported_sources(input_path, packaged_paths=packaged_paths) if input_path is not None else []
    if unsupported:
        manifest["unsupported_sources"] = unsupported
        logger.warning("Found %d unsupported source artifact(s); they are listed in manifest.", len(unsupported))
    skipped_collections = (
        _discover_skipped_source_collections(input_path, packaged_paths=packaged_paths) if input_path is not None else []
    )
    if skipped_collections:
        manifest["skipped_source_collections"] = skipped_collections
        logger.info("Found %d skipped source collection(s); they are listed in manifest.", len(skipped_collections))

    if dry_run:
        logger.info("Dry run: discovered %d static embedding source(s); no files written.", len(discovered_sources))
        return manifest

    package_root.mkdir(parents=True, exist_ok=True)
    manifest_path = package_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Pass overwrite=True to replace it.")

    resolver = _prepare_gene_resolver(
        discovered_sources,
        harmonize_ids=harmonize_ids,
        target_id_type=target_id_type,
        organism=organism,
        gene_resolver=gene_resolver,
    )
    embeddings: dict[str, Any] = {}
    for source in discovered_sources:
        table = read_static_embedding_table(
            source,
            duplicate_policy=duplicate_policy,
            drop_missing_ids=drop_missing_ids,
            nan_policy=nan_policy,
            harmonize_ids=harmonize_ids,
            target_id_type=target_id_type,
            organism=_source_resolver_organism(source, override=organism),
            unresolved_id_policy=unresolved_id_policy,
            gene_resolver=resolver,
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

    index = _index_for_table(table)
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


def render_static_embedding_dataset_card(
    manifest: Mapping[str, Any],
    *,
    repo_id: str | None = None,
) -> str:
    """Render a Hugging Face dataset card from a static embedding manifest."""

    repo = repo_id or "your-org/Embpy_Data"
    embeddings = dict(manifest.get("embeddings") or {})
    planned = list(manifest.get("planned_embeddings") or [])
    entries = embeddings if embeddings else {str(item.get("key")): item for item in planned if item.get("key")}
    n_embeddings = len(entries)
    example_key = sorted(entries)[0] if entries else "genept_scaled"
    total_entities = sum(
        int(entry.get("n_entities", 0))
        for entry in entries.values()
        if _looks_like_int(entry.get("n_entities"))
    )
    species_keys = ", ".join(str(item) for item in manifest.get("species_keys", []) or ["human_9606"])
    taxonomy_ids = ", ".join(str(item) for item in manifest.get("taxonomy_ids", []) or ["9606"])
    default_entry_species_key = (
        str((manifest.get("species_keys") or ["human_9606"])[0])
        if len(manifest.get("species_keys") or ["human_9606"]) == 1
        else ""
    )
    id_harmonization = dict(manifest.get("id_harmonization") or {})
    target_id_type = id_harmonization.get("target_id_type", "ensembl_id")
    unresolved_policy = id_harmonization.get("unresolved_id_policy", "drop")
    generated_at = manifest.get("generated_at", "unknown")
    schema_version = manifest.get("schema_version", SCHEMA_VERSION)
    has_per_embedding_species = bool(entries) and all(
        bool(entry.get("species_key")) for entry in entries.values()
    )
    species_metadata_note = (
        "Each embedding has `species`, `taxonomy_id`, and `species_key` fields in both the package "
        "manifest and `metadata/metadata.json`. If source metadata did not provide species information, "
        "embpy defaults to human (`taxonomy_id=9606`, `species_key=human_9606`)."
        if has_per_embedding_species
        else "This card reports species using package-level defaults. Regenerate the package with a current "
        "embpy build to also write per-embedding `species`, `taxonomy_id`, and `species_key` fields into "
        "`metadata/metadata.json`."
    )

    rows = [
        [
            key,
            entry.get("entity_type", ""),
            entry.get("species_key") or default_entry_species_key,
            entry.get("id_type", ""),
            _format_card_value(entry.get("n_entities", "")),
            _format_card_value(entry.get("n_dims", "")),
            entry.get("description", ""),
        ]
        for key, entry in sorted(entries.items())
    ]
    table = _markdown_table(
        ["key", "entity", "species", "id type", "rows", "dims", "description"],
        rows,
    )

    skipped = manifest.get("skipped_source_collections") or []
    skipped_note = ""
    if skipped:
        n_skipped_files = sum(int(item.get("count", 0)) for item in skipped if _looks_like_int(item.get("count")))
        skipped_note = (
            "\n\nThis package intentionally skips "
            f"{len(skipped)} source collection(s) containing {n_skipped_files} file(s). "
            "Those collections usually contain per-species artifacts and should be packaged only when a "
            "species/taxonomy ID is selected explicitly."
        )

    total_text = f"{total_entities:,}" if total_entities else "available in manifest entries"
    return "\n".join(
        [
            "---",
            "pretty_name: Embpy Static Embeddings",
            "license: other",
            "tags:",
            "- embpy",
            "- biology",
            "- genomics",
            "- gene-embeddings",
            "- protein-embeddings",
            "- zarr",
            "---",
            "",
            "# Embpy Static Embeddings",
            "",
            "This dataset contains static gene and protein embeddings packaged with embpy. "
            "Embedding values are stored in Zarr arrays, while row identifiers, provenance, "
            "species metadata, and AnnData-like `.uns` metadata are stored in sidecar metadata files.",
            "",
            "## Summary",
            "",
            f"- Repository: `{repo}`",
            f"- Schema version: `{schema_version}`",
            f"- Generated at: `{generated_at}`",
            f"- Number of embeddings: `{n_embeddings}`",
            f"- Total indexed entities: `{total_text}`",
            f"- Species keys: `{species_keys}`",
            f"- NCBI taxonomy IDs: `{taxonomy_ids}`",
            f"- Default gene identifier policy: `{target_id_type}`",
            f"- Unresolved gene identifier policy: `{unresolved_policy}`",
            "",
            "## Available Embeddings",
            "",
            table,
            "",
            "## File Layout",
            "",
            "```text",
            "manifest.json",
            "embeddings/<model_key>/",
            "  values.zarr/",
            "  metadata/",
            "    index.parquet",
            "    index.csv",
            "    metadata.json",
            "    uns.json",
            "```",
            "",
            "The dense matrix is stored under `values.zarr`. The `metadata/index.parquet` file maps "
            "row positions to `entity_id` values and any preserved aliases such as source IDs or gene symbols.",
            "",
            "## Loading With embpy",
            "",
            "```python",
            "from embpy.pp import HFHandler",
            "",
            f"hf = HFHandler(\"{repo}\")",
            f"embedding = hf.download_embedding(\"{example_key}\")",
            "matrix = embedding[\"embeddings\"]",
            "ids = embedding[\"ids\"]",
            "```",
            "",
            "For a local checkout or downloaded snapshot:",
            "",
            "```python",
            "from embpy import load_static_embedding_package",
            "",
            f"store = load_static_embedding_package(\"/path/to/package\", key=\"{example_key}\")",
            "tp53 = store.get(\"ENSG00000141510\")",
            "tp53_by_symbol = store.get(\"TP53\", id_type=\"symbol\")",
            "```",
            "",
            "Missing identifiers raise by default. Use `missing=\"drop\"` or `missing=\"nan\"` when a "
            "partial result is acceptable.",
            "",
            "## Metadata And Species",
            "",
            species_metadata_note,
            "",
            "## Validation",
            "",
            "The package was designed to be validated locally before upload:",
            "",
            "```bash",
            "python -m embpy.scripts.package_static_embeddings validate --package /path/to/package",
            "```",
            skipped_note,
            "",
            "## License And Attribution",
            "",
            "This repository aggregates embeddings derived from multiple upstream resources. Please check "
            "the per-embedding metadata and upstream sources for the applicable licenses and citation terms.",
            "",
        ]
    ).replace("\n\n\n", "\n\n")


def write_static_embedding_dataset_card(
    package_root: str | Path,
    *,
    repo_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``README.md`` for a prepared static embedding package."""

    root = Path(package_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    card_path = root / DATASET_CARD_NAME
    if card_path.exists() and not overwrite:
        raise FileExistsError(f"Dataset card already exists: {card_path}. Pass overwrite=True to replace it.")
    manifest = json.loads(manifest_path.read_text())
    card_path.write_text(render_static_embedding_dataset_card(manifest, repo_id=repo_id), encoding="utf-8")
    logger.info("Wrote static embedding dataset card: %s", card_path)
    return card_path


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
        self._row_by_column = self._build_row_lookups()
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
    def species(self) -> str:
        return str(self.metadata.get("species", DEFAULT_SPECIES))

    @property
    def taxonomy_id(self) -> str | None:
        value = self.metadata.get("taxonomy_id")
        return None if value is None else str(value)

    @property
    def species_key(self) -> str:
        return str(self.metadata.get("species_key") or _species_key(self.species, self.taxonomy_id))

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

    def get(
        self,
        identifiers: str | Sequence[str],
        *,
        missing: MissingPolicy = "raise",
        id_type: str | None = None,
    ) -> np.ndarray:
        """Return embeddings for row identifiers or preserved aliases."""

        wanted, scalar = _normalize_identifier_query(identifiers)
        lookup, lookup_name = self._lookup_for_query_id_type(id_type)
        rows: list[int | None] = [lookup.get(identifier) for identifier in wanted]
        missing_ids = [identifier for identifier, row in zip(wanted, rows, strict=True) if row is None]
        missing = _validate_missing_policy(missing)
        if missing_ids and missing == "raise":
            preview = missing_ids[:10]
            raise KeyError(
                f"{len(missing_ids)} identifier(s) are not present in static embedding {self.key!r} "
                f"for id_type={lookup_name!r}; "
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
        id_type: str | None = None,
        as_dataframe: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        """Query identifiers and return a DataFrame by default."""

        wanted, scalar = _normalize_identifier_query(identifiers)
        lookup, lookup_name = self._lookup_for_query_id_type(id_type)
        matrix = self.get(wanted, missing=missing, id_type=id_type)
        if scalar and matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if not as_dataframe:
            return matrix

        if missing == "drop":
            ids = [identifier for identifier in wanted if identifier in lookup]
        else:
            ids = wanted
        return pd.DataFrame(
            np.asarray(matrix, dtype=np.float32),
            index=pd.Index(ids, name=lookup_name),
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

    def _build_row_lookups(self) -> dict[str, dict[str, int]]:
        lookups: dict[str, dict[str, int]] = {"entity_id": dict(self._row_by_id)}
        for column in self.index.columns:
            if column == "entity_id":
                continue
            values = self.index[column].astype(str).tolist()
            mapping: dict[str, int] = {}
            for row, value in enumerate(values):
                if _is_missing_identifier(value):
                    continue
                mapping.setdefault(value, row)
            if mapping:
                lookups[column] = mapping
        return lookups

    def _lookup_for_query_id_type(self, id_type: str | None) -> tuple[dict[str, int], str]:
        if id_type is None:
            return self._row_by_column["entity_id"], "entity_id"

        normalized = _normalize_query_id_type(id_type)
        if normalized in {"entity_id", _normalize_query_id_type(self.id_type)}:
            return self._row_by_column["entity_id"], "entity_id"

        source_id_type = self.metadata.get("source_id_type")
        if source_id_type is not None and normalized == _normalize_query_id_type(str(source_id_type)):
            if "source_id" in self._row_by_column:
                return self._row_by_column["source_id"], str(source_id_type)
            if "gene_symbol" in self._row_by_column and normalized == "symbol":
                return self._row_by_column["gene_symbol"], "gene_symbol"

        alias_column = {
            "source": "source_id",
            "source_id": "source_id",
            "symbol": "gene_symbol",
            "gene_symbol": "gene_symbol",
            "gene_symbols": "gene_symbol",
        }.get(normalized, normalized)
        if alias_column in self._row_by_column:
            return self._row_by_column[alias_column], alias_column

        available = sorted(self._row_by_column)
        raise KeyError(
            f"Static embedding {self.key!r} cannot query id_type={id_type!r}; "
            f"available index columns are {available}."
        )


def _source_from_spec(path: Path, spec: Mapping[str, Any]) -> StaticEmbeddingSource:
    return StaticEmbeddingSource(
        key=str(spec["key"]),
        path=path,
        entity_type=str(spec.get("entity_type", "gene")),
        id_type=spec.get("id_type"),
        species=spec.get("species"),
        taxonomy_id=spec.get("taxonomy_id"),
        sep=spec.get("sep"),
        transpose=bool(spec.get("transpose", False)),
        id_column=spec.get("id_column"),
        embedding_columns=_optional_tuple(spec.get("embedding_columns")),
        metadata_columns=_tuple(spec.get("metadata_columns")),
        skip_rows=int(spec.get("skip_rows", 0)),
        header=_parse_header_value(spec.get("header", 0)),
        comment=spec.get("comment"),
        na_values=_tuple(spec.get("na_values")),
        h5_embedding_dataset=str(spec.get("h5_embedding_dataset", "embeddings")),
        h5_id_dataset=str(spec.get("h5_id_dataset", "proteins")),
        id_regex=spec.get("id_regex"),
        description=spec.get("description"),
        metadata={
            k: v
            for k, v in spec.items()
            if k
            not in {
                "key",
                "path",
                "entity_type",
                "id_type",
                "species",
                "taxonomy_id",
                "taxon_id",
                "species_code",
                "ncbi_taxon_id",
                "ncbi_taxonomy_id",
                "sep",
                "transpose",
                "id_column",
                "embedding_columns",
                "metadata_columns",
                "skip_rows",
                "header",
                "comment",
                "na_values",
                "h5_embedding_dataset",
                "h5_id_dataset",
                "id_regex",
                "description",
            }
        },
    )


def _source_from_config_item(
    item: object,
    *,
    base_dir: Path,
    config_path: Path,
) -> StaticEmbeddingSource:
    if not isinstance(item, Mapping):
        raise ValueError(f"Every source in {config_path} must be an object, got {type(item).__name__}.")
    allowed = {
        "key",
        "path",
        "entity_type",
        "id_type",
        "species",
        "taxonomy_id",
        "taxon_id",
        "species_code",
        "ncbi_taxon_id",
        "ncbi_taxonomy_id",
        "sep",
        "transpose",
        "id_column",
        "embedding_columns",
        "metadata_columns",
        "skip_rows",
        "header",
        "comment",
        "na_values",
        "h5_embedding_dataset",
        "h5_id_dataset",
        "id_regex",
        "description",
        "metadata",
    }
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ValueError(
            f"Source config {config_path} has unsupported field(s) for source "
            f"{item.get('key', '<missing-key>')!r}: {unknown}. "
            f"Supported fields are {sorted(allowed)}."
        )
    if "key" not in item or "path" not in item:
        raise ValueError(f"Every source in {config_path} must define 'key' and 'path'.")

    source_path = Path(str(item["path"]))
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata for source {item['key']!r} in {config_path} must be an object.")
    species, taxonomy_id, _ = _normalize_species_info(
        species=item.get("species"),
        taxonomy_id=_first_present(item, ("taxonomy_id", "taxon_id", "species_code", "ncbi_taxon_id", "ncbi_taxonomy_id")),
        metadata=metadata,
    )
    return StaticEmbeddingSource(
        key=str(item["key"]),
        path=source_path,
        entity_type=str(item.get("entity_type", "gene")),
        id_type=item.get("id_type"),
        species=species,
        taxonomy_id=taxonomy_id,
        sep=item.get("sep"),
        transpose=bool(item.get("transpose", False)),
        id_column=item.get("id_column"),
        embedding_columns=_optional_tuple(item.get("embedding_columns")),
        metadata_columns=_tuple(item.get("metadata_columns")),
        skip_rows=int(item.get("skip_rows", 0)),
        header=_parse_header_value(item.get("header", 0)),
        comment=item.get("comment"),
        na_values=_tuple(item.get("na_values")),
        h5_embedding_dataset=str(item.get("h5_embedding_dataset", "embeddings")),
        h5_id_dataset=str(item.get("h5_id_dataset", "proteins")),
        id_regex=item.get("id_regex"),
        description=item.get("description"),
        metadata=dict(metadata),
    )


def _string_sources_for_species(root: Path, species_codes: Sequence[str | int]) -> list[StaticEmbeddingSource]:
    sources: list[StaticEmbeddingSource] = []
    missing: list[str] = []
    for raw_species in species_codes:
        species = str(raw_species).strip()
        if not species or not species.isdigit():
            raise ValueError(
                f"STRING species code must be an NCBI taxonomy integer, got {raw_species!r}."
            )
        n_found = 0
        for spec in _STRING_SOURCE_TEMPLATES.values():
            rel = str(spec["rel_template"]).format(species=species)
            path = root / rel
            if not path.is_file():
                continue
            n_found += 1
            species_name, taxonomy_id, _ = _normalize_species_info(taxonomy_id=species)
            sources.append(
                StaticEmbeddingSource(
                    key=str(spec["key_template"]).format(species=species),
                    path=path,
                    entity_type="protein",
                    id_type="string_protein_id",
                    species=species_name,
                    taxonomy_id=taxonomy_id,
                    h5_embedding_dataset="embeddings",
                    h5_id_dataset="proteins",
                    description=str(spec["description_template"]).format(species=species),
                    metadata={"discovery": "explicit-string-species", "string_species": species},
                )
            )
        if n_found == 0:
            missing.extend(str(spec["rel_template"]).format(species=species) for spec in _STRING_SOURCE_TEMPLATES.values())
    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            "Requested STRING species file(s) were not found under the static embedding input directory. "
            f"First missing: {preview}."
        )
    return sources


def _source_species_info(source: StaticEmbeddingSource) -> tuple[str, str | None, str]:
    return _normalize_species_info(
        species=source.species,
        taxonomy_id=source.taxonomy_id,
        metadata=source.metadata,
    )


def _source_resolver_organism(source: StaticEmbeddingSource, *, override: str | None) -> str:
    if override is not None:
        return _canonical_species_name(override)
    species, _, _ = _source_species_info(source)
    return species


def _default_resolver_organism(
    sources: Sequence[StaticEmbeddingSource],
    *,
    override: str | None,
) -> str:
    if override is not None:
        return _canonical_species_name(override)
    gene_species = [
        _source_species_info(source)[0]
        for source in sources
        if source.entity_type == "gene"
    ]
    return gene_species[0] if gene_species else DEFAULT_SPECIES


def _normalize_species_info(
    *,
    species: object | None = None,
    taxonomy_id: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, str | None, str]:
    metadata = metadata or {}
    metadata_species = _first_present(metadata, ("species", "organism"))
    metadata_taxonomy_id = _first_present(
        metadata,
        ("taxonomy_id", "taxon_id", "ncbi_taxon_id", "ncbi_taxonomy_id", "species_code", "string_species"),
    )
    resolved_taxonomy_id = _normalize_taxonomy_id(taxonomy_id if taxonomy_id is not None else metadata_taxonomy_id)
    raw_species = species if species is not None else metadata_species

    if raw_species is None:
        if resolved_taxonomy_id is None:
            resolved_species = DEFAULT_SPECIES
        else:
            resolved_species = _TAXONOMY_SPECIES.get(resolved_taxonomy_id, f"taxon_{resolved_taxonomy_id}")
    else:
        resolved_species = _canonical_species_name(raw_species)

    if resolved_taxonomy_id is None:
        resolved_taxonomy_id = _SPECIES_TAXONOMY.get(resolved_species)
    if resolved_taxonomy_id is None and resolved_species == DEFAULT_SPECIES:
        resolved_taxonomy_id = DEFAULT_TAXONOMY_ID
    expected_taxonomy_id = _SPECIES_TAXONOMY.get(resolved_species)
    if (
        expected_taxonomy_id is not None
        and resolved_taxonomy_id is not None
        and expected_taxonomy_id != resolved_taxonomy_id
    ):
        raise ValueError(
            "species and taxonomy_id/species_code do not agree: "
            f"species={resolved_species!r} maps to taxonomy_id={expected_taxonomy_id!r}, "
            f"but got taxonomy_id={resolved_taxonomy_id!r}."
        )

    return resolved_species, resolved_taxonomy_id, _species_key(resolved_species, resolved_taxonomy_id)


def _canonical_species_name(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return DEFAULT_SPECIES
    aliases = {
        "homo_sapiens": "human",
        "9606": "human",
        "mus_musculus": "mouse",
        "10090": "mouse",
        "rattus_norvegicus": "rat",
        "10116": "rat",
        "danio_rerio": "zebrafish",
        "7955": "zebrafish",
        "drosophila_melanogaster": "drosophila",
        "7227": "drosophila",
        "caenorhabditis_elegans": "c_elegans",
        "6239": "c_elegans",
        "saccharomyces_cerevisiae": "yeast",
        "4932": "yeast",
        "arabidopsis_thaliana": "arabidopsis",
        "3702": "arabidopsis",
    }
    return aliases.get(text, text)


def _normalize_taxonomy_id(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"taxonomy_id/species_code must be an integer NCBI taxonomy ID, got {value!r}.")
        text = str(int(value))
    else:
        text = str(value).strip()
    text = re.sub(r"^NCBITaxon:", "", text, flags=re.IGNORECASE)
    if not text:
        return None
    if not text.isdigit():
        raise ValueError(
            "taxonomy_id/species_code must be an integer NCBI taxonomy ID such as '9606' "
            f"for human or '10090' for mouse, got {value!r}."
        )
    return text


def _species_key(species: str, taxonomy_id: str | None) -> str:
    species_part = _sanitize_key(species or DEFAULT_SPECIES)
    return f"{species_part}_{taxonomy_id}" if taxonomy_id else species_part


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _prepare_gene_resolver(
    sources: Sequence[StaticEmbeddingSource],
    *,
    harmonize_ids: bool,
    target_id_type: TargetIdType,
    organism: str | None,
    gene_resolver: Any | None,
) -> Any | None:
    if not harmonize_ids or target_id_type != "ensembl_id":
        return gene_resolver
    if not any(source.entity_type == "gene" for source in sources):
        return gene_resolver

    resolver = gene_resolver
    if resolver is None:
        from embpy.resources.gene.resolver import GeneResolver

        resolver = GeneResolver(species=_default_resolver_organism(sources, override=organism))
    return _CachedGeneResolver(resolver)


class _CachedGeneResolver:
    """Small per-prepare cache around the existing GeneResolver batch API."""

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver
        self._symbol_to_ensembl: dict[tuple[str, str], str | None] = {}

    def symbols_to_ensembl_batch(self, symbols: list[str], organism: str = "human") -> dict[str, str | None]:
        unique = list(dict.fromkeys(str(symbol) for symbol in symbols))
        missing = [
            symbol
            for symbol in unique
            if (organism, symbol) not in self._symbol_to_ensembl
        ]
        if missing:
            mapping = self._resolver.symbols_to_ensembl_batch(missing, organism=organism)
            for symbol in missing:
                self._symbol_to_ensembl[(organism, symbol)] = mapping.get(symbol)
        return {symbol: self._symbol_to_ensembl.get((organism, str(symbol))) for symbol in symbols}


def _read_source_frames(source: StaticEmbeddingSource) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    path = Path(source.path)
    if path.suffix.lower() in _HDF5_SUFFIXES:
        return _read_hdf5_source_frames(source)

    frame = _read_raw_table_frame(source)
    if frame.empty:
        return frame, {}

    ids, value_frame = _extract_identifier_index(frame, source)
    if source.transpose:
        if source.embedding_columns or source.metadata_columns:
            raise ValueError(
                f"Static embedding source {path} uses transpose=True, so embedding_columns and "
                "metadata_columns cannot be selected before transposition. Remove those fields or "
                "pre-convert the file so genes are rows."
            )
        value_frame = value_frame.copy()
        value_frame.index = pd.Index(ids)
        transposed = value_frame.T
        if transposed.empty:
            raise ValueError(f"Static embedding source {path} is empty after transposition.")
        return transposed, {}

    value_frame = value_frame.copy()
    value_frame.index = pd.Index(ids)
    metadata_columns = _resolve_columns(value_frame, source.metadata_columns, field_name="metadata_columns")
    index_columns: dict[str, list[str]] = {}
    if metadata_columns:
        for column in metadata_columns:
            index_columns[_stringify_column(column)] = value_frame[column].astype(str).tolist()
        value_frame = value_frame.drop(columns=metadata_columns)

    if source.embedding_columns is not None:
        embedding_columns = _resolve_columns(value_frame, source.embedding_columns, field_name="embedding_columns")
        value_frame = value_frame.loc[:, embedding_columns]

    if value_frame.shape[1] == 0:
        raise ValueError(
            f"Static embedding source {path} has no embedding columns after parsing. "
            "Check id_column, metadata_columns, embedding_columns, and transpose."
        )
    return value_frame, index_columns


def _read_raw_table_frame(source: StaticEmbeddingSource) -> pd.DataFrame:
    path = Path(source.path)
    suffix = path.suffix.lower()
    compound_suffix = "".join(path.suffixes[-2:]).lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return pd.DataFrame(df)
    if suffix not in {".csv", ".tsv", ".txt"} and compound_suffix not in _COMPRESSED_TABULAR_SUFFIXES:
        raise ValueError(f"Unsupported static embedding source extension {suffix!r}: {path}")
    resolved_sep = source.sep
    if resolved_sep is None:
        resolved_sep = "\t" if suffix in {".tsv", ".txt"} or compound_suffix in {".tsv.gz", ".txt.gz"} else ","
    return pd.read_csv(
        path,
        sep=resolved_sep,
        header=source.header,
        skiprows=int(source.skip_rows),
        comment=source.comment,
        na_values=list(source.na_values) if source.na_values else None,
    )


def _read_hdf5_source_frames(source: StaticEmbeddingSource) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    path = Path(source.path)
    if source.transpose:
        raise ValueError(
            f"Static embedding HDF5 source {path} does not support transpose=True. "
            "HDF5 sources must contain one 2D embedding dataset with rows aligned to one ID dataset."
        )
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to package HDF5 static embedding sources.") from exc

    with h5py.File(path, "r") as handle:
        if source.h5_embedding_dataset not in handle:
            available = sorted(handle.keys())
            raise KeyError(
                f"HDF5 source {path} is missing embedding dataset {source.h5_embedding_dataset!r}. "
                f"Available top-level keys: {available}. Pass h5_embedding_dataset in the source config "
                "or --h5-embedding-dataset on the CLI."
            )
        if source.h5_id_dataset not in handle:
            available = sorted(handle.keys())
            raise KeyError(
                f"HDF5 source {path} is missing ID dataset {source.h5_id_dataset!r}. "
                f"Available top-level keys: {available}. Pass h5_id_dataset in the source config "
                "or --h5-id-dataset on the CLI."
            )
        matrix = np.asarray(handle[source.h5_embedding_dataset], dtype=np.float32)
        ids = _decode_hdf5_ids(np.asarray(handle[source.h5_id_dataset]))
        h5_attrs = dict(handle.attrs)
        if "metadata" in handle and hasattr(handle["metadata"], "attrs"):
            h5_attrs.update({f"metadata.{key}": value for key, value in dict(handle["metadata"].attrs).items()})

    if matrix.ndim != 2:
        raise ValueError(
            f"HDF5 source {path} dataset {source.h5_embedding_dataset!r} must be 2D, got shape={matrix.shape!r}."
        )
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"HDF5 source {path} has {matrix.shape[0]} embedding rows but {len(ids)} IDs in "
            f"dataset {source.h5_id_dataset!r}."
        )
    columns = [f"dim_{i}" for i in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, index=pd.Index(ids), columns=columns)
    index_columns: dict[str, list[str]] = {}
    if source.id_type == "string_protein_id":
        stripped = [_strip_string_taxon_prefix(identifier) for identifier in ids]
        if any(value != identifier for value, identifier in zip(stripped, ids, strict=False)):
            index_columns["ensembl_protein_id"] = stripped
    if h5_attrs:
        logger.info("Read HDF5 metadata attributes for key=%s: %s", source.key, sorted(h5_attrs))
    return frame, index_columns


def _decode_hdf5_ids(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _strip_string_taxon_prefix(identifier: str) -> str:
    text = str(identifier)
    if "." not in text:
        return text
    taxon, accession = text.split(".", 1)
    return accession if taxon.isdigit() else text


def _extract_identifier_index(
    frame: pd.DataFrame,
    source: StaticEmbeddingSource,
) -> tuple[list[object], pd.DataFrame]:
    path = Path(source.path)
    if source.id_column is not None:
        id_column = _resolve_column(frame, source.id_column, field_name="id_column")
        ids = frame[id_column].tolist()
        values = frame.drop(columns=[id_column])
        return ids, values

    if not isinstance(frame.index, pd.RangeIndex):
        return frame.index.tolist(), frame
    if frame.shape[1] == 0:
        raise ValueError(f"Static embedding source {path} has no columns; cannot infer identifier column.")

    first_column = frame.columns[0]
    ids = frame[first_column].tolist()
    values = frame.drop(columns=[first_column])
    logger.info(
        "Using first column %r as row identifiers for key=%s. "
        "Pass id_column explicitly for messy files.",
        first_column,
        source.key,
    )
    return ids, values


def _coerce_embedding_matrix(
    frame: pd.DataFrame,
    ids: Sequence[str],
    source: StaticEmbeddingSource,
) -> np.ndarray:
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        details = _first_nonnumeric_values(frame, ids)
        detail_text = "; ".join(details) if details else str(exc)
        raise ValueError(
            f"Static embedding source {source.path} contains non-numeric values in embedding columns. "
            f"First offending values: {detail_text}. "
            "If these columns are metadata, pass metadata_columns/--metadata-columns. "
            "If only some columns are embedding dimensions, pass embedding_columns/--embedding-columns. "
            "If the file has genes as columns instead of rows, pass transpose=True/--transpose."
        ) from exc
    return numeric.to_numpy(dtype=np.float32)


def _first_nonnumeric_values(frame: pd.DataFrame, ids: Sequence[str], *, limit: int = 5) -> list[str]:
    details: list[str] = []
    for column in frame.columns:
        series = frame[column]
        converted = pd.to_numeric(series, errors="coerce")
        bad = converted.isna() & ~series.isna()
        if not bool(bad.any()):
            continue
        first_pos = int(np.where(bad.to_numpy())[0][0])
        row_id = ids[first_pos] if first_pos < len(ids) else first_pos
        value = series.iloc[first_pos]
        details.append(f"column={_stringify_column(column)!r} row_id={str(row_id)!r} value={str(value)!r}")
        if len(details) >= limit:
            break
    return details


def _resolve_columns(
    frame: pd.DataFrame,
    selectors: Sequence[str | int],
    *,
    field_name: str,
) -> list[Any]:
    return [_resolve_column(frame, selector, field_name=field_name) for selector in selectors]


def _resolve_column(frame: pd.DataFrame, selector: str | int, *, field_name: str) -> Any:
    if selector in frame.columns:
        return selector
    if isinstance(selector, str):
        for column in frame.columns:
            if str(column) == selector:
                return column
        try:
            position = int(selector)
        except ValueError:
            position = None
    else:
        position = int(selector)

    if position is not None and -len(frame.columns) <= position < len(frame.columns):
        return frame.columns[position]
    preview = [_stringify_column(column) for column in frame.columns[:12]]
    raise KeyError(
        f"Could not resolve {field_name}={selector!r}. Available columns begin with {preview}. "
        "Use an exact column name or a zero-based column position."
    )


def _tuple(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _optional_tuple(value: object) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return _tuple(value)


def _parse_header_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "no", "false"}:
        return None
    return int(value)


def _stringify_column(column: object) -> str:
    return str(column)


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


def _index_for_table(table: StaticEmbeddingTable) -> pd.DataFrame:
    data: dict[str, list[str]] = {"entity_id": list(table.entity_ids)}
    for column, values in table.alias_columns.items():
        if len(values) != table.n_entities:
            raise ValueError(
                f"Alias column {column!r} for key={table.key!r} has {len(values)} rows; "
                f"expected {table.n_entities}."
            )
        if column == "entity_id":
            raise ValueError("Alias columns cannot replace required column 'entity_id'.")
        data[column] = [str(value) for value in values]
    return pd.DataFrame(data)


def _metadata_for_table(table: StaticEmbeddingTable, *, model_dir: Path) -> dict[str, Any]:
    source = table.source_path
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "key": table.key,
            "entity_type": table.entity_type,
            "id_type": table.id_type,
            "id_key": "entity_id",
            "source_id_type": table.source_id_type,
            "target_id_type": table.target_id_type or table.id_type,
            "organism": table.organism,
            "species": table.species,
            "taxonomy_id": table.taxonomy_id,
            "species_key": table.species_key,
            "id_harmonized": bool(table.id_harmonized),
            "index_columns": ["entity_id", *table.alias_columns.keys()],
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
                "species": table.species,
                "taxonomy_id": table.taxonomy_id,
                "species_key": table.species_key,
            },
            "source_metadata": dict(table.source_metadata),
            "n_duplicate_input_ids": int(table.n_duplicate_input_ids),
            "n_missing_input_ids": int(table.n_missing_input_ids),
            "n_nan_input_values": int(table.n_nan_input_values),
            "n_nan_input_rows": int(table.n_nan_input_rows),
            "n_nan_input_columns": int(table.n_nan_input_columns),
            "nan_policy": table.nan_policy,
            "id_harmonization": {
                "enabled": bool(table.target_id_type == "ensembl_id"),
                "source_id_type": table.source_id_type,
                "target_id_type": table.target_id_type or table.id_type,
                "organism": table.organism,
                "unresolved_id_policy": table.unresolved_id_policy,
                "n_unresolved_ids": int(table.n_unresolved_harmonization_ids),
                "n_duplicate_canonical_ids": int(table.n_duplicate_harmonized_ids),
                "alias_columns": sorted(table.alias_columns),
            },
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
                "source_id_type": table.source_id_type,
                "target_id_type": table.target_id_type or table.id_type,
                "organism": table.organism,
                "species": table.species,
                "taxonomy_id": table.taxonomy_id,
                "species_key": table.species_key,
                "index_columns": ["entity_id", *table.alias_columns.keys()],
                "n_entities": table.n_entities,
                "n_dims": table.n_dims,
                "description": table.description,
                "source": metadata.get("source", {}),
                "id_harmonization": metadata.get("id_harmonization", {}),
            }
        }
    )


def _manifest_entry_for_table(table: StaticEmbeddingTable, *, model_dir: Path) -> dict[str, Any]:
    return _json_safe(
        {
            "key": table.key,
            "entity_type": table.entity_type,
            "id_type": table.id_type,
            "source_id_type": table.source_id_type,
            "target_id_type": table.target_id_type or table.id_type,
            "organism": table.organism,
            "species": table.species,
            "taxonomy_id": table.taxonomy_id,
            "species_key": table.species_key,
            "id_harmonized": bool(table.id_harmonized),
            "index_columns": ["entity_id", *table.alias_columns.keys()],
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
            "n_nan_input_values": int(table.n_nan_input_values),
            "n_nan_input_rows": int(table.n_nan_input_rows),
            "n_nan_input_columns": int(table.n_nan_input_columns),
            "nan_policy": table.nan_policy,
            "n_unresolved_harmonization_ids": int(table.n_unresolved_harmonization_ids),
            "n_duplicate_harmonized_ids": int(table.n_duplicate_harmonized_ids),
            "unresolved_id_policy": table.unresolved_id_policy,
            "package_dir": model_dir.as_posix(),
        }
    )


def _base_manifest(*, package_root: Path, input_dir: Path | None, dry_run: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layout": "single_repo_embeddings_folder_v1",
        "package_root": str(package_root),
        "input_dir": str(input_dir) if input_dir is not None else None,
        "dry_run": bool(dry_run),
        "generated_at": _utc_now(),
        "embeddings": {},
        "n_embeddings": 0,
        "unsupported_sources": [],
        "skipped_source_collections": [],
    }


def _source_manifest_entry(source: StaticEmbeddingSource) -> dict[str, Any]:
    species, taxonomy_id, species_key = _source_species_info(source)
    return _json_safe(
        {
            "key": source.key,
            "path": str(source.path),
            "entity_type": source.entity_type,
            "id_type": source.id_type,
            "species": species,
            "taxonomy_id": taxonomy_id,
            "species_key": species_key,
            "transpose": source.transpose,
            "id_column": source.id_column,
            "embedding_columns": list(source.embedding_columns) if source.embedding_columns is not None else None,
            "metadata_columns": list(source.metadata_columns),
            "sep": source.sep,
            "skip_rows": source.skip_rows,
            "header": source.header,
            "h5_embedding_dataset": source.h5_embedding_dataset,
            "h5_id_dataset": source.h5_id_dataset,
            "description": source.description,
            "source_metadata": dict(source.metadata),
        }
    )


def _discover_unsupported_sources(input_dir: Path, *, packaged_paths: set[Path]) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in packaged_paths:
            continue
        suffix = path.suffix.lower()
        if (
            _is_auto_discoverable_source_table(path)
            or suffix in _HDF5_SUFFIXES
            or _has_emstore_parent(path)
            or _has_static_package_parent(path)
        ):
            continue
        if suffix == ".zip":
            unsupported.append(
                {
                    "path": str(path),
                    "suffix": suffix,
                    "reason": (
                        "compressed collection archive; extract or select one source file before packaging. "
                        "Known extracted human STRING 9606 HDF5 files are auto-packaged when present."
                    ),
                }
            )
    return unsupported


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row)[: len(headers)]
        padded.extend("" for _ in range(len(headers) - len(padded)))
        lines.append("| " + " | ".join(_markdown_cell(value) for value in padded) + " |")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text.strip()


def _format_card_value(value: Any) -> str:
    if _looks_like_int(value):
        return f"{int(value):,}"
    return "" if value is None else str(value)


def _looks_like_int(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return str(value).strip() == str(int(value))


def _discover_skipped_source_collections(input_dir: Path, *, packaged_paths: set[Path]) -> list[dict[str, Any]]:
    h5_by_parent: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _HDF5_SUFFIXES:
            continue
        if path.resolve() in packaged_paths:
            continue
        if _has_emstore_parent(path) or _has_static_package_parent(path):
            continue
        parent = str(path.parent)
        entry = h5_by_parent.setdefault(
            parent,
            {
                "path": parent,
                "suffix": ".h5",
                "count": 0,
                "reason": (
                    "per-species HDF5 collection skipped by default. Package a chosen species with "
                    "--source-file/--h5-id-dataset/--h5-embedding-dataset, or add it to a source config. "
                    "Known human STRING 9606 files are already included as planned embeddings when present."
                ),
            },
        )
        entry["count"] += 1
    return sorted(h5_by_parent.values(), key=lambda item: str(item["path"]))


def _harmonize_gene_ids_to_ensembl(
    ids: list[str],
    matrix: np.ndarray,
    *,
    source_id_type: str,
    organism: str,
    unresolved_policy: UnresolvedIdPolicy,
    gene_resolver: Any | None,
    index_columns: Mapping[str, list[str]],
    key: str,
) -> tuple[list[str], np.ndarray, Mapping[str, tuple[str, ...]], dict[str, int], bool]:
    source_id_type_norm = _normalize_query_id_type(source_id_type)
    if source_id_type_norm in {"ensembl_id", "ensembl_gene_id"}:
        return (
            ids,
            matrix,
            {column: tuple(values) for column, values in index_columns.items()},
            {"unresolved": 0, "duplicates": 0},
            False,
        )

    from embpy.io._canon import canonicalize, drop_and_dedup

    raw_ids = list(ids)
    logger.info(
        "Harmonizing %d gene identifier(s) for key=%s from id_type=%s to ensembl_id.",
        len(raw_ids),
        key,
        source_id_type,
    )
    canon, keep = canonicalize(
        raw_ids,
        "gene",
        organism,
        id_type=source_id_type,
        gene_resolver=gene_resolver,
    )
    unresolved_idx = np.where(~keep)[0]
    n_unresolved = int(len(unresolved_idx))
    if n_unresolved and unresolved_policy == "error":
        preview = [raw_ids[int(i)] for i in unresolved_idx[:10]]
        raise ValueError(
            f"Static embedding source for key={key!r} has {n_unresolved} gene identifier(s) "
            "that could not be harmonized to Ensembl IDs. "
            f"First unresolved: {preview}. Pass unresolved_id_policy='drop' to discard them."
        )

    alias_cols: dict[str, list[str]] = {column: list(values) for column, values in index_columns.items()}
    source_column = _dedup_index_column_name("source_id", alias_cols)
    alias_cols[source_column] = raw_ids
    if source_id_type_norm in {"symbol", "gene_symbol", "gene_symbols"}:
        alias_cols[_dedup_index_column_name("gene_symbol", alias_cols)] = raw_ids

    out_ids, out_matrix, out_alias, _out_raw = drop_and_dedup(
        canon,
        matrix,
        keep,
        raw_ids,
        alias_cols,
    )
    if not out_ids:
        raise ValueError(
            f"Static embedding source for key={key!r} has no rows left after Ensembl ID harmonization."
        )

    n_duplicate_canonical = int(keep.sum()) - len(out_ids)
    if n_unresolved:
        logger.warning(
            "Dropped %d unresolved gene identifier row(s) while harmonizing key=%s to Ensembl IDs.",
            n_unresolved,
            key,
        )

    id_harmonized = any(raw != canon_id for raw, canon_id in zip(out_alias[source_column], out_ids, strict=False))
    return (
        out_ids,
        np.asarray(out_matrix, dtype=np.float32),
        {column: tuple(values) for column, values in out_alias.items()},
        {"unresolved": n_unresolved, "duplicates": n_duplicate_canonical},
        id_harmonized,
    )


def _handle_duplicate_ids(
    ids: list[str],
    matrix: np.ndarray,
    *,
    index_columns: Mapping[str, list[str]],
    policy: DuplicatePolicy,
    key: str,
) -> tuple[list[str], np.ndarray, dict[str, list[str]], int]:
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
        return ids, matrix, {column: list(values) for column, values in index_columns.items()}, 0
    if policy == "error":
        raise ValueError(
            f"Static embedding source for key={key!r} contains duplicate identifiers; "
            f"first duplicate={first_duplicate!r}; duplicate rows={duplicates}. "
            "Pass duplicate_policy='first' / --duplicates first to keep first occurrence, "
            "or fix the source file before packaging."
        )

    mask = np.asarray(keep, dtype=bool)
    deduped_ids = [entity_id for entity_id, should_keep in zip(ids, keep, strict=True) if should_keep]
    deduped_index_columns = _subset_index_columns(index_columns, mask)
    logger.warning("Dropped %d duplicate identifier row(s) for key=%s.", duplicates, key)
    return deduped_ids, np.asarray(matrix[mask], dtype=np.float32), deduped_index_columns, duplicates


def _handle_nan_values(
    matrix: np.ndarray,
    ids: list[str],
    *,
    index_columns: Mapping[str, list[str]],
    policy: NanPolicy,
    path: Path,
    column_names: Sequence[str],
) -> tuple[np.ndarray, list[str], dict[str, list[str]], dict[str, int]]:
    finite_by_row = np.isfinite(matrix).all(axis=1)
    if finite_by_row.all():
        return (
            matrix,
            ids,
            {column: list(values) for column, values in index_columns.items()},
            {"values": 0, "rows": 0, "columns": 0},
        )

    bad_rows = np.where(~finite_by_row)[0]
    n_bad_values = int((~np.isfinite(matrix)).sum())
    bad_cols = np.where(~np.isfinite(matrix).all(axis=0))[0]
    first = int(bad_rows[0])
    bad_positions = np.argwhere(~np.isfinite(matrix))
    first_bad_row = int(bad_positions[0, 0])
    first_bad_col = int(bad_positions[0, 1])
    first_bad_col_name = column_names[first_bad_col] if first_bad_col < len(column_names) else str(first_bad_col)
    counts = {"values": n_bad_values, "rows": int(len(bad_rows)), "columns": int(len(bad_cols))}
    if policy == "error":
        raise ValueError(
            f"Static embedding source {path} contains NaN/Inf values; "
            f"{counts['values']} bad value(s) across {counts['rows']}/{matrix.shape[0]} row(s) "
            f"and {counts['columns']}/{matrix.shape[1]} column(s). "
            f"First bad value row={first_bad_row} id={ids[first_bad_row]!r} "
            f"column={first_bad_col_name!r}. "
            "Pass nan_policy='drop-rows' / --nan-policy drop-rows to discard affected identifiers, "
            "or nan_policy='fill-zero' / --nan-policy fill-zero only when a zero fill is scientifically valid."
        )
    if policy == "drop-rows":
        logger.warning(
            "Dropping %d row(s) with %d NaN/Inf value(s) from %s. First bad row=%d id=%r.",
            counts["rows"],
            counts["values"],
            path,
            first,
            ids[first],
        )
        keep = finite_by_row
        kept_ids = [entity_id for entity_id, ok in zip(ids, keep, strict=True) if bool(ok)]
        if not kept_ids:
            raise ValueError(
                f"Static embedding source {path} has no fully finite rows after applying nan_policy='drop-rows'. "
                "Fix the source values or use nan_policy='fill-zero' only if zero-imputation is appropriate."
            )
        return (
            np.asarray(matrix[keep], dtype=np.float32),
            kept_ids,
            _subset_index_columns(index_columns, keep),
            counts,
        )

    logger.warning(
        "Filling %d NaN/Inf value(s) across %d row(s) with 0 in %s.",
        counts["values"],
        counts["rows"],
        path,
    )
    return (
        np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        ids,
        {column: list(values) for column, values in index_columns.items()},
        counts,
    )


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


def _validate_nan_policy(value: str) -> NanPolicy:
    if value not in ("error", "drop-rows", "fill-zero"):
        raise ValueError(
            f"nan_policy must be 'error', 'drop-rows', or 'fill-zero', got {value!r}."
        )
    return value  # type: ignore[return-value]


def _validate_missing_policy(value: str) -> MissingPolicy:
    if value not in ("raise", "drop", "nan"):
        raise ValueError(f"missing must be 'raise', 'drop', or 'nan', got {value!r}.")
    return value  # type: ignore[return-value]


def _validate_target_id_type(value: str) -> TargetIdType:
    normalized = _normalize_query_id_type(value)
    if normalized in {"source", "original", "none", "keep"}:
        return "source"
    if normalized in {"ensembl_id", "ensembl_ids", "ensembl_gene_id", "ensembl_gene_ids"}:
        return "ensembl_id"
    raise ValueError(
        "target_id_type must be 'ensembl_id' or 'source', "
        f"got {value!r}."
    )


def _validate_unresolved_id_policy(value: str) -> UnresolvedIdPolicy:
    if value not in ("drop", "error"):
        raise ValueError(f"unresolved_id_policy must be 'drop' or 'error', got {value!r}.")
    return value  # type: ignore[return-value]


def _normalize_source_id_type(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_query_id_type(value)


def _normalize_query_id_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"ensembl_ids", "ensembl_gene_id", "ensembl_gene_ids"}:
        return "ensembl_id"
    if normalized in {"symbols", "gene_symbols"}:
        return "symbol"
    return normalized


def _is_model_dir(path: Path) -> bool:
    return (path / VALUES_STORE_NAME).is_dir() and (path / METADATA_DIR_NAME / "metadata.json").is_file()


def _has_emstore_parent(path: Path) -> bool:
    return any(part.endswith(".emstore") for part in path.parts)


def _has_static_package_parent(path: Path) -> bool:
    parts = path.parts
    if VALUES_STORE_NAME in parts:
        return True
    for i, part in enumerate(parts):
        if part == "embeddings" and i + 2 < len(parts) and parts[i + 2] == METADATA_DIR_NAME:
            return True
    return False


def _is_supported_source_table(path: Path) -> bool:
    suffix = path.suffix.lower()
    compound_suffix = "".join(path.suffixes[-2:]).lower()
    return (
        suffix in _TABULAR_SUFFIXES
        or suffix == ".txt"
        or suffix in _HDF5_SUFFIXES
        or compound_suffix in _COMPRESSED_TABULAR_SUFFIXES
    )


def _is_auto_discoverable_source_table(path: Path) -> bool:
    suffix = path.suffix.lower()
    compound_suffix = "".join(path.suffixes[-2:]).lower()
    return suffix in _TABULAR_SUFFIXES or suffix == ".txt" or compound_suffix in _COMPRESSED_TABULAR_SUFFIXES


def _subset_index_columns(
    columns: Mapping[str, Sequence[str]],
    keep: Sequence[int] | np.ndarray,
) -> dict[str, list[str]]:
    if isinstance(keep, np.ndarray) and keep.dtype == bool:
        indices = [int(i) for i in np.where(keep)[0]]
    else:
        indices = [int(i) for i in keep]
    return {name: [str(values[i]) for i in indices] for name, values in columns.items()}


def _dedup_index_column_name(name: str, columns: Mapping[str, Any]) -> str:
    if name not in columns:
        return name
    i = 2
    while f"{name}_{i}" in columns:
        i += 1
    return f"{name}_{i}"


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
    "load_static_embedding_source_config",
    "load_static_embedding_package",
    "prepare_static_embedding_package",
    "read_static_embedding_table",
    "validate_static_embedding_dir",
    "validate_static_embedding_package",
    "write_static_embedding_package",
]
