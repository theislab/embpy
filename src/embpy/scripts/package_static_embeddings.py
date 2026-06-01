"""Prepare, validate, and upload embpy static embedding packages.

Examples
--------
Local dry run:

    python -m embpy.scripts.package_static_embeddings prepare \
        --input data/static_embeddings \
        --output data/static_embedding_package \
        --dry-run

Prepare and validate locally:

    python -m embpy.scripts.package_static_embeddings prepare \
        --input data/static_embeddings \
        --output data/static_embedding_package

Upload only after explicit confirmation:

    python -m embpy.scripts.package_static_embeddings upload \
        --package data/static_embedding_package \
        --repo-id your-org/Embpy_Data \
        --execute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from embpy.pp.hf_handler import HFHandler
from embpy.pp.static_embeddings import (
    DATASET_CARD_NAME,
    StaticEmbeddingSource,
    prepare_static_embedding_package,
    render_static_embedding_dataset_card,
    validate_static_embedding_package,
    write_static_embedding_dataset_card,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package static embeddings as zarr plus metadata sidecars.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Discover local tables and write the package layout.")
    prepare.add_argument(
        "--input",
        default=None,
        help="Directory containing local static embeddings. Optional when --source-config or --source-file is used.",
    )
    prepare.add_argument("--output", required=True, help="Output package directory.")
    prepare.add_argument("--keys", nargs="*", default=None, help="Optional subset of model keys to package.")
    prepare.add_argument(
        "--source-config",
        default=None,
        help=(
            "JSON file describing one or more custom sources. Relative paths are resolved "
            "against --input when provided, otherwise against the config file."
        ),
    )
    prepare.add_argument(
        "--string-species",
        nargs="*",
        default=None,
        help=(
            "Additional STRING NCBI taxonomy IDs to package from precomputed_embeddings_string. "
            "Default packages only known human 9606 files when present."
        ),
    )
    prepare.add_argument(
        "--source-file",
        default=None,
        help="One custom source table to package. Use the source parsing flags below to describe its layout.",
    )
    prepare.add_argument("--source-key", default=None, help="Model key for --source-file; defaults to file stem.")
    prepare.add_argument(
        "--source-id-type",
        default=None,
        help="Identifier type present in --source-file before harmonization.",
    )
    prepare.add_argument("--entity-type", default="gene", help="Entity type for --source-file.")
    prepare.add_argument(
        "--species",
        default=None,
        help="Species/organism for --source-file metadata and gene ID harmonization. Default: human.",
    )
    prepare.add_argument(
        "--taxonomy-id",
        default=None,
        help="NCBI taxonomy ID for --source-file, e.g. 9606 for human or 10090 for mouse.",
    )
    prepare.add_argument(
        "--id-column",
        default=None,
        help="Column name or zero-based position containing row identifiers for --source-file.",
    )
    prepare.add_argument(
        "--embedding-columns",
        nargs="*",
        default=None,
        help="Column names or positions to use as embedding dimensions for --source-file.",
    )
    prepare.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Non-embedding columns to preserve in metadata/index.* for --source-file.",
    )
    prepare.add_argument("--sep", default=None, help="Delimiter for --source-file; inferred from extension by default.")
    prepare.add_argument("--transpose", action="store_true", help="Transpose --source-file before packaging.")
    prepare.add_argument("--id-regex", default=None, help="Optional regex used to clean source row identifiers.")
    prepare.add_argument("--description", default=None, help="Description stored in metadata for --source-file.")
    prepare.add_argument("--skip-rows", type=int, default=0, help="Rows to skip before reading --source-file.")
    prepare.add_argument(
        "--header",
        default="0",
        help="Header row for --source-file, or 'none' for files without a header. Default: 0.",
    )
    prepare.add_argument(
        "--na-values",
        nargs="*",
        default=None,
        help="Additional strings to parse as missing numeric values for --source-file.",
    )
    prepare.add_argument(
        "--h5-embedding-dataset",
        default="embeddings",
        help="HDF5 dataset containing the 2D embedding matrix for --source-file.",
    )
    prepare.add_argument(
        "--h5-id-dataset",
        default="proteins",
        help="HDF5 dataset containing row identifiers for --source-file.",
    )
    prepare.add_argument(
        "--known-only",
        action="store_true",
        help="Package only embpy's known static embedding paths; ignore other tabular files.",
    )
    prepare.add_argument("--dry-run", action="store_true", help="Discover and print the plan without writing files.")
    prepare.add_argument("--overwrite", action="store_true", help="Replace existing local package entries.")
    prepare.add_argument(
        "--duplicates",
        choices=["error", "first"],
        default="error",
        help="How to handle duplicate source identifiers.",
    )
    prepare.add_argument(
        "--drop-missing-ids",
        action="store_true",
        help="Discard source rows with blank/missing identifiers after reporting their count.",
    )
    prepare.add_argument(
        "--nan-policy",
        choices=["error", "drop-rows", "fill-zero"],
        default="error",
        help=(
            "How to handle NaN/Inf embedding values. Default fails. "
            "'drop-rows' discards affected identifiers; 'fill-zero' replaces missing values with 0."
        ),
    )
    prepare.add_argument(
        "--target-id-type",
        default="ensembl_id",
        choices=["ensembl_id", "ensembl_gene_id", "source"],
        help="Canonical row identifier type to write. Default harmonizes gene identifiers to Ensembl IDs.",
    )
    prepare.add_argument(
        "--organism",
        default=None,
        help=(
            "Override organism name passed to embpy's gene resolver. "
            "By default each source uses its species metadata, falling back to human."
        ),
    )
    prepare.add_argument(
        "--on-unresolved-ids",
        choices=["drop", "error"],
        default="drop",
        help="How to handle source gene IDs that cannot be harmonized to the target ID type.",
    )
    prepare.add_argument(
        "--no-harmonize-ids",
        action="store_true",
        help="Keep source row identifiers instead of harmonizing gene IDs.",
    )

    validate = sub.add_parser("validate", help="Validate a package root or one embeddings/<key> directory.")
    validate.add_argument("--package", required=True, help="Package root or model package directory.")

    card = sub.add_parser("card", help="Generate a Hugging Face README.md dataset card from a package manifest.")
    card.add_argument("--package", required=True, help="Prepared package root.")
    card.add_argument("--repo-id", default=None, help="HF dataset repo id to show in examples.")
    card.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local README.md dataset card.",
    )
    card.add_argument(
        "--print",
        dest="print_card",
        action="store_true",
        help="Print the generated markdown instead of writing README.md.",
    )

    upload = sub.add_parser("upload", help="Upload a validated package to a Hugging Face dataset repo.")
    upload.add_argument("--package", required=True, help="Prepared package root.")
    upload.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. your-org/Embpy_Data.")
    upload.add_argument("--token", default=None, help="Optional Hugging Face token.")
    upload.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    upload.add_argument(
        "--execute",
        action="store_true",
        help="Actually create/upload. Without this flag, upload is a local dry run.",
    )
    upload.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing remote manifest or embeddings/<key>/ prefixes.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "prepare":
            explicit_sources = _explicit_sources_from_args(args)
            manifest = prepare_static_embedding_package(
                args.input,
                args.output,
                keys=args.keys,
                sources=explicit_sources,
                source_config=args.source_config,
                string_species=args.string_species,
                include_unknown=not args.known_only,
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                duplicate_policy=args.duplicates,
                drop_missing_ids=bool(args.drop_missing_ids),
                nan_policy=args.nan_policy,
                harmonize_ids=not bool(args.no_harmonize_ids),
                target_id_type=args.target_id_type,
                organism=args.organism,
                unresolved_id_policy=args.on_unresolved_ids,
            )
            print(_summary_json(manifest))
            return 0

        if args.command == "validate":
            summaries = validate_static_embedding_package(args.package)
            print(
                json.dumps(
                    [
                        {
                            "key": item.key,
                            "path": str(item.path),
                            "n_entities": item.n_entities,
                            "n_dims": item.n_dims,
                            "id_type": item.id_type,
                        }
                        for item in summaries
                    ],
                    indent=2,
                )
            )
            return 0

        if args.command == "upload":
            return _upload(args)
        if args.command == "card":
            return _card(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


def _upload(args: argparse.Namespace) -> int:
    package_root = Path(args.package)
    validate_static_embedding_package(package_root)
    manifest = _read_manifest(package_root)
    planned = _planned_remote_prefixes(
        manifest,
        include_dataset_card=(package_root / DATASET_CARD_NAME).is_file(),
    )

    if not args.execute:
        logger.info("Upload dry run only. Pass --execute to upload to Hugging Face.")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "repo_id": args.repo_id,
                    "package": str(package_root),
                    "remote_prefixes": planned,
                    "n_embeddings": len((manifest.get("embeddings") or {})),
                },
                indent=2,
            )
        )
        return 0

    handler = HFHandler(args.repo_id, token=args.token)
    handler.create_repo(private=bool(args.private))
    remote_files = handler.list_files()
    conflicts = _remote_conflicts(remote_files, planned)
    if conflicts and not args.allow_overwrite:
        preview = conflicts[:20]
        raise FileExistsError(
            "Remote package paths already exist and --allow-overwrite was not set. "
            f"First conflicts: {preview}"
        )

    handler.upload_folder(
        package_root,
        "",
        commit_message=f"Upload embpy static embedding package ({len(planned)} prefixes)",
    )
    logger.info("Uploaded static embedding package to %s", args.repo_id)
    return 0


def _card(args: argparse.Namespace) -> int:
    package_root = Path(args.package)
    if args.print_card:
        manifest = _read_manifest(package_root)
        print(render_static_embedding_dataset_card(manifest, repo_id=args.repo_id))
        return 0

    card_path = write_static_embedding_dataset_card(
        package_root,
        repo_id=args.repo_id,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"dataset_card": str(card_path)}, indent=2))
    return 0


def _explicit_sources_from_args(args: argparse.Namespace) -> list[StaticEmbeddingSource] | None:
    if not args.source_file:
        return None
    path = Path(args.source_file)
    key = args.source_key or _sanitize_key(path.stem)
    return [
        StaticEmbeddingSource(
            key=key,
            path=path,
            entity_type=args.entity_type,
            id_type=_normalize_source_id_type(args.source_id_type),
            species=args.species,
            taxonomy_id=args.taxonomy_id,
            sep=args.sep,
            transpose=bool(args.transpose),
            id_column=args.id_column,
            embedding_columns=tuple(args.embedding_columns) if args.embedding_columns is not None else None,
            metadata_columns=tuple(args.metadata_columns or ()),
            skip_rows=int(args.skip_rows),
            header=_parse_header(args.header),
            na_values=tuple(args.na_values or ()),
            h5_embedding_dataset=args.h5_embedding_dataset,
            h5_id_dataset=args.h5_id_dataset,
            id_regex=args.id_regex,
            description=args.description,
            metadata={"discovery": "explicit-cli"},
        )
    ]


def _parse_header(value: str | None) -> int | None:
    if value is None:
        return 0
    if str(value).strip().lower() in {"none", "no", "false"}:
        return None
    return int(value)


def _normalize_source_id_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"gene_symbol", "gene_symbols", "symbols"}:
        return "symbol"
    if normalized in {"ensembl_gene_id", "ensembl_gene_ids", "ensembl_ids"}:
        return "ensembl_id"
    return normalized


def _sanitize_key(value: str) -> str:
    import re

    clean = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    return clean or "static_embedding"


def _read_manifest(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _planned_remote_prefixes(manifest: dict[str, Any], *, include_dataset_card: bool = False) -> list[str]:
    keys = sorted((manifest.get("embeddings") or {}).keys())
    prefixes = ["manifest.json", *[f"embeddings/{key}/" for key in keys]]
    if include_dataset_card:
        prefixes.insert(0, DATASET_CARD_NAME)
    return prefixes


def _remote_conflicts(remote_files: list[str], planned_prefixes: list[str]) -> list[str]:
    conflicts: list[str] = []
    for remote in remote_files:
        for prefix in planned_prefixes:
            if prefix.endswith("/"):
                if remote.startswith(prefix):
                    conflicts.append(remote)
                    break
            elif remote == prefix:
                conflicts.append(remote)
                break
    return sorted(conflicts)


def _summary_json(manifest: dict[str, Any]) -> str:
    embeddings = manifest.get("embeddings") or {}
    planned = manifest.get("planned_embeddings") or []
    payload = {
        "dry_run": bool(manifest.get("dry_run")),
        "n_embeddings": len(embeddings),
        "n_planned_embeddings": len(planned),
        "embeddings": sorted(embeddings),
        "planned_embeddings": [item.get("key") for item in planned],
        "species_keys": manifest.get("species_keys", []),
        "taxonomy_ids": manifest.get("taxonomy_ids", []),
        "unsupported_sources": manifest.get("unsupported_sources", []),
        "skipped_source_collections": manifest.get("skipped_source_collections", []),
        "package_root": manifest.get("package_root"),
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    sys.exit(main())
