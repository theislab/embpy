"""Upload the data/embeddings folder to a Hugging Face dataset repo.

Usage
-----
    # Upload everything
    python upload_embeddings_to_hf.py --repo theislab/embpy-embeddings

    # Upload only drug embeddings
    python upload_embeddings_to_hf.py --repo theislab/embpy-embeddings --subfolder drug_embeddings

    # Dry run (list files without uploading)
    python upload_embeddings_to_hf.py --repo theislab/embpy-embeddings --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"


def list_files(directory: Path) -> list[Path]:
    """Recursively list all files in a directory."""
    return sorted(f for f in directory.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload embeddings to a Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HF dataset repo ID (e.g. 'theislab/embpy-embeddings').",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
        help="Local embeddings directory (default: data/embeddings).",
    )
    parser.add_argument(
        "--subfolder", type=str, default=None,
        help="Upload only a specific subfolder (e.g. 'drug_embeddings').",
    )
    parser.add_argument(
        "--path-in-repo", type=str, default="embeddings",
        help="Target path prefix in the HF repo (default: 'embeddings').",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HF API token (uses cached token from `huggingface-cli login` if omitted).",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create repo as private if it does not exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be uploaded without uploading.",
    )
    parser.add_argument(
        "--commit-message", type=str, default=None,
        help="Custom commit message for the upload.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.subfolder:
        data_dir = data_dir / args.subfolder

    if not data_dir.is_dir():
        logger.error("Directory not found: %s", data_dir)
        raise SystemExit(1)

    files = list_files(data_dir)
    total_size = sum(f.stat().st_size for f in files)
    logger.info(
        "Found %d files (%.2f GB) in %s",
        len(files), total_size / 1e9, data_dir,
    )

    if args.dry_run:
        for f in files:
            rel = f.relative_to(data_dir)
            size_mb = f.stat().st_size / 1e6
            print(f"  {rel}  ({size_mb:.1f} MB)")
        logger.info("Dry run complete. No files uploaded.")
        return

    from embpy.pp.hf_handler import HFHandler

    hf = HFHandler(repo_name=args.repo, token=args.token)
    hf.create_repo(private=args.private)

    path_in_repo = args.path_in_repo
    if args.subfolder:
        path_in_repo = f"{path_in_repo}/{args.subfolder}"

    commit_msg = args.commit_message or f"Upload {data_dir.name} ({len(files)} files, {total_size / 1e9:.1f} GB)"

    logger.info("Uploading %s -> %s/%s ...", data_dir, args.repo, path_in_repo)
    hf.upload_folder(
        local_dir=data_dir,
        path_in_repo=path_in_repo,
        commit_message=commit_msg,
    )
    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
