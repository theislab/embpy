#!/usr/bin/env python3
"""Upload perturbation datasets, metadata, and molecule embeddings to Hugging Face.

Usage
-----
    python scripts/upload_to_hf.py \
        --repo-name "your-username/perturbation-embeddings" \
        --data-dir ./data

The ``--data-dir`` should follow this layout::

    data/
      raw/            # .h5ad files
      metadata/       # perturbations.parquet
      embeddings/     # one .parquet per model

Files that don't exist are silently skipped, so you can upload incrementally.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Model catalogue (keep in sync with embpy) ──────────────────────────────
MOLECULE_MODELS = [
    "chemberta2MTR",
    "chemberta2MLM",
    "molformer_base",
    "rdkit_fp",
    "morgan_fp",
    "morgan_count_fp",
    "maccs_fp",
    "atom_pair_fp",
    "atom_pair_count_fp",
    "torsion_fp",
    "torsion_count_fp",
    "minimol",
    "mhg_gnn",
    "mole",
]

DATASET_CARD = textwrap.dedent("""\
    ---
    license: cc-by-4.0
    task_categories:
      - feature-extraction
    tags:
      - biology
      - perturbation
      - small-molecule
      - embeddings
    ---

    # Perturbation Embeddings

    Pre-computed small-molecule embeddings for perturbation biology datasets.

    ## Datasets included

    | Dataset | Description |
    |---------|-------------|
    | TAHOE | Large-scale perturbation atlas |
    | LINCS | Library of Integrated Network-based Cellular Signatures |
    | GDSC | Genomics of Drug Sensitivity in Cancer |
    | SciPlex | Single-cell chemical perturbation screen |
    | ComboSciPlex | Combination perturbation screen |

    ## Embedding models

    {model_table}

    ## Repo layout

    ```
    raw/              # Raw .h5ad perturbation data
    metadata/         # perturbations.parquet (drug names, resolved SMILES, dataset)
    embeddings/       # One .parquet per model (columns: smiles, embedding)
    ```

    ## Usage

    ```python
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="{repo_name}",
        filename="embeddings/chemberta2MTR.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    # df.columns: ["smiles", "embedding"]
    ```

    Generated with [embpy](https://github.com/grpinto/embpy).
""")


def _build_model_table() -> str:
    lines = ["| Model key | Type |", "|-----------|------|"]
    for m in MOLECULE_MODELS:
        if m.startswith(("chemberta", "molformer")):
            mtype = "Transformer"
        elif m in ("minimol", "mhg_gnn", "mole"):
            mtype = "GNN"
        else:
            mtype = "RDKit fingerprint"
        lines.append(f"| `{m}` | {mtype} |")
    return "\n".join(lines)


def upload(repo_name: str, data_dir: Path, *, create_repo: bool = False) -> None:
    """Upload all files from *data_dir* to the HF dataset repo."""
    api = HfApi()

    if create_repo:
        api.create_repo(repo_name, repo_type="dataset", exist_ok=True)
        log.info("Ensured repo %s exists", repo_name)

    readme_text = DATASET_CARD.format(
        model_table=_build_model_table(),
        repo_name=repo_name,
    )
    readme_path = data_dir / "README.md"
    readme_path.write_text(readme_text)
    log.info("Wrote dataset card to %s", readme_path)

    files_uploaded = 0

    for subdir in ("raw", "metadata", "embeddings"):
        local_dir = data_dir / subdir
        if not local_dir.is_dir():
            log.warning("Skipping %s (not found)", local_dir)
            continue
        for fpath in sorted(local_dir.iterdir()):
            if fpath.is_file():
                path_in_repo = f"{subdir}/{fpath.name}"
                log.info("Uploading %s → %s", fpath, path_in_repo)
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=path_in_repo,
                    repo_id=repo_name,
                    repo_type="dataset",
                )
                files_uploaded += 1

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    files_uploaded += 1

    log.info("Done — uploaded %d files to %s", files_uploaded, repo_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload perturbation data to Hugging Face")
    parser.add_argument("--repo-name", required=True, help='HF dataset repo, e.g. "user/repo"')
    parser.add_argument("--data-dir", required=True, type=Path, help="Local directory with raw/, metadata/, embeddings/")
    parser.add_argument("--create-repo", action="store_true", help="Create the HF repo if it doesn't exist")
    args = parser.parse_args()

    upload(args.repo_name, args.data_dir, create_repo=args.create_repo)


if __name__ == "__main__":
    main()
