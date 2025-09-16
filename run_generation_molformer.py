#!/usr/bin/env python3
import argparse
import logging
import os
import re
from collections.abc import Iterable
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from embpy.embedder import BioEmbedder  # uses your MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
VALID_POOLINGS = ["cls", "mean", "max"]


def slugify(name: str) -> str:
    name = name.strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_one_pool(
    smiles: list[str], ids: list[str], model_key: str, pooling: str, batch_size: int, out_csv: Path
) -> tuple[int, int]:
    """

    Streams embeddings in micro-batches to avoid OOM and appends to out_csv.
    Returns (num_valid, num_invalid).
    """
    assert pooling in VALID_POOLINGS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    n_valid = 0
    n_invalid = 0

    embedder = BioEmbedder(device="auto")

    for sm_chunk, id_chunk in zip(chunked(smiles, batch_size), chunked(ids, batch_size), strict=False):
        # embed this micro-batch
        emb_list = embedder.embed_molecules_batch(
            identifiers=sm_chunk,
            model=model_key,
            pooling_strategy=pooling,
        )
        # determine embedding dim from first valid
        dim = None
        for e in emb_list:
            if e is not None:
                dim = e.shape[0]
                break
        if dim is None:
            # all invalid in this micro-batch
            n_invalid += len(sm_chunk)
            continue

        cols = [f"e_{j}" for j in range(dim)]
        rows = []
        idxs = []
        for rid, e in zip(id_chunk, emb_list, strict=False):
            if e is None:
                n_invalid += 1
                continue
            if e.shape[0] != dim:
                raise RuntimeError(f"Inconsistent embedding dim for {rid}: {e.shape[0]} vs {dim}")
            rows.append(e)
            idxs.append(rid)
            n_valid += 1

        if rows:
            df = pd.DataFrame(np.vstack(rows), columns=cols, index=idxs)
            df.to_csv(out_csv, mode=("a" if wrote_header else "w"), header=not wrote_header, index=True)
            wrote_header = True

    return n_valid, n_invalid


def main():
    ap = argparse.ArgumentParser(description="Sharded MoLFormer embeddings (CLS/MEAN/MAX) without OOM.")
    ap.add_argument("--csv", required=True, help="Input CSV path.")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES column name.")
    ap.add_argument("--id-col", default=None, help="Row ID column (defaults to smiles col).")
    ap.add_argument("--model", default="molformer_base", help="MODEL_REGISTRY key (e.g., molformer_base).")
    ap.add_argument("--poolings", nargs="+", default=["cls", "mean", "max"], choices=VALID_POOLINGS)
    ap.add_argument("--outdir", required=True, help="Output directory for shard CSVs.")
    ap.add_argument("--batch-size", type=int, default=64, help="Micro-batch size to avoid GPU OOM.")
    ap.add_argument("--num-shards", type=int, default=None, help="Total number of shards.")
    ap.add_argument("--shard-id", type=int, default=None, help="This shard index (0-based).")
    args = ap.parse_args()

    # Slurm auto-detect (if not provided)
    if args.num_shards is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        args.num_shards = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    if args.shard_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        args.shard_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if (args.num_shards is None) ^ (args.shard_id is None):
        raise SystemExit("Provide both --num-shards and --shard-id (or use Slurm array env vars).")
    if args.num_shards is None:
        # no sharding: treat as single shard
        args.num_shards, args.shard_id = 1, 0

    # Load CSV
    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"SMILES column '{args.smiles_col}' not found. Columns: {list(df.columns)}")
    if args.id_col is None:
        args.id_col = args.smiles_col
    if args.id_col not in df.columns:
        raise SystemExit(f"ID column '{args.id_col}' not found. Columns: {list(df.columns)}")

    smiles_all = df[args.smiles_col].fillna("").astype(str).tolist()
    ids_all = df[args.id_col].fillna("").astype(str).tolist()
    N = len(smiles_all)
    print(f"Total molecules: {N}")

    # Compute shard slice (≈ equal-sized)
    per = ceil(N / args.num_shards)
    start = args.shard_id * per
    end = min(N, (args.shard_id + 1) * per)
    if start >= end:
        print(f"Shard {args.shard_id}/{args.num_shards}: nothing to do.")
        return
    smiles = smiles_all[start:end]
    ids = ids_all[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: rows [{start}:{end}) → {len(smiles)} items")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = slugify(args.model)

    # Run each pooling → write shard CSVs
    for p in args.poolings:
        out_csv = outdir / f"{tag}_embeddings_{p}.sh{args.shard_id:03d}.csv"
        ok, bad = run_one_pool(smiles, ids, args.model, p, args.batch_size, out_csv)
        print(f"[{p}] valid={ok}, invalid={bad}, wrote={out_csv}")


if __name__ == "__main__":
    main()
