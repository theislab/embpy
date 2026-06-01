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
    prepare_static_embedding_package,
    validate_static_embedding_package,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package static embeddings as zarr plus metadata sidecars.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Discover local tables and write the package layout.")
    prepare.add_argument("--input", required=True, help="Directory containing local static embeddings.")
    prepare.add_argument("--output", required=True, help="Output package directory.")
    prepare.add_argument("--keys", nargs="*", default=None, help="Optional subset of model keys to package.")
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

    validate = sub.add_parser("validate", help="Validate a package root or one embeddings/<key> directory.")
    validate.add_argument("--package", required=True, help="Package root or model package directory.")

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
            manifest = prepare_static_embedding_package(
                args.input,
                args.output,
                keys=args.keys,
                include_unknown=not args.known_only,
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                duplicate_policy=args.duplicates,
                drop_missing_ids=bool(args.drop_missing_ids),
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
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


def _upload(args: argparse.Namespace) -> int:
    package_root = Path(args.package)
    validate_static_embedding_package(package_root)
    manifest = _read_manifest(package_root)
    planned = _planned_remote_prefixes(manifest)

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


def _read_manifest(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _planned_remote_prefixes(manifest: dict[str, Any]) -> list[str]:
    keys = sorted((manifest.get("embeddings") or {}).keys())
    return ["manifest.json", *[f"embeddings/{key}/" for key in keys]]


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
        "unsupported_sources": manifest.get("unsupported_sources", []),
        "package_root": manifest.get("package_root"),
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    sys.exit(main())
