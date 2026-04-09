#!/usr/bin/env python3
"""Batch-generate molecule embeddings for a single model + pooling combo.

Designed to be called from a SLURM array job where each task runs one
(model, pooling) configuration.

Usage
-----
    python embed_small_molecule.py \
        --smiles-file data/embeddings/rdkit_200.csv \
        --output-dir  data/embeddings/drug_embeddings \
        --model chemberta2MTR \
        --pooling mean \
        --batch-size 64 \
        --device auto

Output is saved as a CSV to:
    {output_dir}/{model}/embeddings_{pooling}.csv
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

TRANSFORMER_MODELS = ["chemberta2MTR", "chemberta2MLM", "molformer_base"]
FINGERPRINT_MODELS = [
    "rdkit_fp", "morgan_fp", "morgan_count_fp", "maccs_fp",
    "atom_pair_fp", "atom_pair_count_fp", "torsion_fp", "torsion_count_fp",
]
GNN_MODELS = ["minimol", "mhg_gnn", "mole"]

ALL_MOLECULE_MODELS = TRANSFORMER_MODELS + FINGERPRINT_MODELS + GNN_MODELS

POOLING_BY_MODEL_TYPE = {
    "transformer": ["mean", "max", "cls"],
    "fingerprint": ["flat"],
    "gnn": ["graph"],
}


def _model_type(model_key: str) -> str:
    if model_key in TRANSFORMER_MODELS:
        return "transformer"
    if model_key in FINGERPRINT_MODELS:
        return "fingerprint"
    return "gnn"


def _valid_poolings(model_key: str) -> list[str]:
    return POOLING_BY_MODEL_TYPE[_model_type(model_key)]


def _load_smiles(path: Path) -> list[str]:
    """Load unique SMILES from a CSV or Parquet file."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, usecols=["smiles"])
    if "smiles" not in df.columns:
        raise ValueError(f"Input file must have a 'smiles' column; found {list(df.columns)}")
    smiles = df["smiles"].dropna().unique().tolist()
    log.info("Loaded %d unique SMILES from %s", len(smiles), path)
    return smiles


def _embed_and_save(
    smiles: list[str],
    model_key: str,
    pooling: str,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> None:
    """Generate embeddings for one (model, pooling) and save to CSV."""
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"embeddings_{pooling}.csv"

    if out_path.exists():
        log.info("SKIP %s/%s -- output exists at %s", model_key, pooling, out_path)
        return

    from embpy import BioEmbedder

    log.info("=== %s  pooling=%s ===", model_key, pooling)
    t0 = time.perf_counter()

    embedder = BioEmbedder(device=device)

    all_embeddings: list[np.ndarray | None] = []
    all_smiles: list[str] = []
    n_chunks = (len(smiles) + batch_size - 1) // batch_size

    for i in range(0, len(smiles), batch_size):
        chunk = smiles[i : i + batch_size]
        chunk_idx = i // batch_size + 1
        if chunk_idx % 50 == 1 or chunk_idx == n_chunks:
            log.info("  chunk %d/%d  (%d SMILES)", chunk_idx, n_chunks, len(chunk))
        try:
            actual_pooling = "mean" if pooling == "graph" else pooling
            embs = embedder.embed_molecules_batch(
                chunk, model=model_key, pooling_strategy=actual_pooling,
            )
            for smi, emb in zip(chunk, embs, strict=True):
                if emb is not None:
                    all_embeddings.append(emb)
                    all_smiles.append(smi)
        except Exception:
            log.exception("  Error in chunk %d", chunk_idx)

    elapsed = time.perf_counter() - t0
    log.info(
        "  %s/%s done -- %d/%d valid in %.1fs",
        model_key, pooling, len(all_embeddings), len(smiles), elapsed,
    )

    if not all_embeddings:
        log.warning("  No valid embeddings -- skipping save")
        return

    emb_matrix = np.stack(all_embeddings)
    dim = emb_matrix.shape[1]
    col_names = [f"e_{j}" for j in range(dim)]
    df = pd.DataFrame(emb_matrix, columns=col_names)
    df.insert(0, "smiles", all_smiles)
    df.to_csv(out_path, index=False)
    log.info("  Saved %s  (%d x %d)", out_path, len(df), dim)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate molecule embeddings (single model + pooling).",
    )
    parser.add_argument("--smiles-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=ALL_MOLECULE_MODELS)
    parser.add_argument(
        "--pooling", required=True,
        help="Pooling strategy: mean | max | cls (transformers), flat (fingerprints), graph (GNNs)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    valid = _valid_poolings(args.model)
    if args.pooling not in valid:
        parser.error(
            f"Pooling '{args.pooling}' not valid for {args.model}. "
            f"Choose from: {valid}"
        )

    smiles = _load_smiles(args.smiles_file)
    if not smiles:
        log.warning("No SMILES found -- exiting.")
        return

    _embed_and_save(
        smiles, args.model, args.pooling,
        args.output_dir, args.batch_size, args.device,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
