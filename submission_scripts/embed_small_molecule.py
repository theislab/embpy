#!/usr/bin/env python3
"""Batch-generate molecule embeddings for all (or selected) embpy models.

Usage
-----
    python scripts/small_molecule_embedding_generation.py \
        --smiles-file data/unique_smiles.parquet \
        --output-dir  data/embeddings \
        --batch-size  64

The input file must contain a column called ``smiles`` (CSV or Parquet).
One ``.parquet`` file per model is written to *output-dir* with columns
``smiles`` and ``embedding``.  Models whose output file already exists are
skipped, so the script is safe to re-run for resume.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

ALL_MOLECULE_MODELS = [
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


def _load_smiles(path: Path) -> list[str]:
    """Load unique SMILES from a CSV or Parquet file."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "smiles" not in df.columns:
        raise ValueError(f"Input file must have a 'smiles' column; found {list(df.columns)}")

    smiles = df["smiles"].dropna().unique().tolist()
    log.info("Loaded %d unique SMILES from %s", len(smiles), path)
    return smiles


def _embed_model(
    smiles: list[str],
    model_key: str,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    """Generate embeddings for one model and save to parquet."""
    out_path = output_dir / f"{model_key}.parquet"
    if out_path.exists():
        log.info("Skipping %s — output already exists at %s", model_key, out_path)
        return

    from embpy import BioEmbedder

    log.info("═══ Model: %s ═══", model_key)
    t0 = time.perf_counter()

    try:
        embedder = BioEmbedder(device=device)
    except Exception:
        log.exception("Failed to initialise BioEmbedder for model %s", model_key)
        return

    all_embeddings: list[np.ndarray | None] = []
    n_chunks = (len(smiles) + batch_size - 1) // batch_size

    for i in range(0, len(smiles), batch_size):
        chunk = smiles[i : i + batch_size]
        chunk_idx = i // batch_size + 1
        log.info("  chunk %d/%d  (%d SMILES)", chunk_idx, n_chunks, len(chunk))
        try:
            embs = embedder.embed_molecules_batch(chunk, model=model_key)
            all_embeddings.extend(embs)
        except Exception:
            log.exception("  Error in chunk %d for model %s — filling with None", chunk_idx, model_key)
            all_embeddings.extend([None] * len(chunk))

    valid = sum(1 for e in all_embeddings if e is not None)
    elapsed = time.perf_counter() - t0
    log.info("  %s done — %d/%d valid embeddings in %.1fs", model_key, valid, len(smiles), elapsed)

    records = []
    for smi, emb in zip(smiles, all_embeddings, strict=True):
        if emb is not None:
            records.append({"smiles": smi, "embedding": emb.tolist()})

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    log.info("  Saved %s (%d rows)", out_path, len(df))


def main() -> None:
    """CLI entry-point: parse arguments and run embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate molecule embeddings for all embpy models",
    )
    parser.add_argument(
        "--smiles-file",
        required=True,
        type=Path,
        help="CSV or Parquet file with a 'smiles' column",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write per-model .parquet files",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model keys to embed (default: all molecule models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of SMILES per batch (default: 64)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto | cuda | cpu (default: auto)",
    )
    args = parser.parse_args()

    models = args.models if args.models else ALL_MOLECULE_MODELS
    unknown = set(models) - set(ALL_MOLECULE_MODELS)
    if unknown:
        parser.error(f"Unknown model(s): {unknown}.  Choose from {ALL_MOLECULE_MODELS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    smiles = _load_smiles(args.smiles_file)
    if not smiles:
        log.warning("No SMILES found — exiting.")
        return

    log.info("Generating embeddings for %d models, %d SMILES, batch_size=%d", len(models), len(smiles), args.batch_size)

    for model_key in models:
        _embed_model(smiles, model_key, args.output_dir, args.batch_size, args.device)

    log.info("All models complete.")


if __name__ == "__main__":
    main()
