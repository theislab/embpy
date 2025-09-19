#!/usr/bin/env python3
import argparse
import os
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd

from embpy.embedder import BioEmbedder

VALID_POOLINGS = ["mean", "cls", "max"]


def chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def write_chunk(path: str, df: pd.DataFrame, wrote_header: bool):
    df.to_csv(path, mode="a", header=not wrote_header, index=True)


def main(
    in_csv: str,
    out_prefix: str,
    smiles_col: str | None,
    id_col: str | None,
    model_key: str,
    batch_size: int,
):
    # 1) Read input (first row is header)
    df = pd.read_csv(in_csv)
    if smiles_col is None:
        smiles_col = df.columns[0]
    if smiles_col not in df.columns:
        raise SystemExit(f"SMILES column '{smiles_col}' not found. Columns: {list(df.columns)}")

    if id_col is None:
        id_col = smiles_col
    if id_col not in df.columns:
        raise SystemExit(f"ID column '{id_col}' not found. Columns: {list(df.columns)}")

    smiles_all = df[smiles_col].astype(str).tolist()
    ids_all = df[id_col].astype(str).tolist()
    print(f"Found {len(smiles_all)} molecules in input file.")

    # 2) Initialize embedder (auto: CUDA/MPS/CPU)
    embedder = BioEmbedder(device="auto")

    # 3) Prepare outputs (make dir if prefix includes folders)
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_paths: dict[str, str] = {p: f"{out_prefix}_{p}.csv.part" for p in VALID_POOLINGS}
    final_paths: dict[str, str] = {p: f"{out_prefix}_{p}.csv" for p in VALID_POOLINGS}
    wrote_header = dict.fromkeys(VALID_POOLINGS, False)
    invalid_ids: list[str] = []

    # 4) Embed (do all three poolings; keep rows valid for ALL poolings)
    t0 = time.time()
    offset = 0
    for sm_chunk in chunked(smiles_all, batch_size):
        id_chunk = ids_all[offset : offset + len(sm_chunk)]
        print(f"→ Embedding rows {offset}..{offset + len(sm_chunk) - 1}")

        # Run each pooling; this returns list[np.ndarray|None] preserving order
        by_pool = {}
        for p in VALID_POOLINGS:
            by_pool[p] = embedder.embed_molecules_batch(
                identifiers=sm_chunk,
                model=model_key,  # "chemberta_zinc_v1"
                pooling_strategy=p,  # "mean" | "cls" | "max"
            )

        # Determine rows valid across ALL poolings (mostly guards invalid SMILES)
        valid_mask = []
        for i in range(len(sm_chunk)):
            ok = all(by_pool[p][i] is not None for p in VALID_POOLINGS)
            valid_mask.append(ok)
            if not ok:
                invalid_ids.append(id_chunk[i])

        # Append one chunk per pooling
        for p in VALID_POOLINGS:
            # find dim from first valid
            dim = None
            for i, ok in enumerate(valid_mask):
                if ok:
                    dim = by_pool[p][i].shape[0]
                    break
            if dim is None:
                continue  # nothing valid in this chunk for this pooling

            cols = [f"e{i}" for i in range(dim)]
            rows = []
            idxs = []
            for rid, emb, ok in zip(id_chunk, by_pool[p], valid_mask, strict=False):
                if not ok:
                    continue
                if emb.shape[0] != dim:
                    raise RuntimeError(f"Inconsistent dim for ID {rid} in pooling '{p}': {emb.shape[0]} vs {dim}")
                rows.append(emb)
                idxs.append(rid)

            if rows:
                chunk_df = pd.DataFrame(np.vstack(rows), columns=cols, index=idxs)
                write_chunk(tmp_paths[p], chunk_df, wrote_header[p])
                wrote_header[p] = True

        offset += len(sm_chunk)

    # 5) Finalize files
    for p in VALID_POOLINGS:
        if wrote_header[p]:
            os.replace(tmp_paths[p], final_paths[p])
            print(f"Wrote: {final_paths[p]}")
        else:
            if os.path.exists(tmp_paths[p]):
                os.remove(tmp_paths[p])
            print(f"No valid rows for pooling '{p}'")

    # Log invalids (if any)
    if invalid_ids:
        bad_path = f"{out_prefix}_invalid_smiles.txt"
        with open(bad_path, "w") as fh:
            for rid in invalid_ids:
                fh.write(str(rid) + "\n")
        print(f"Logged {len(invalid_ids)} invalid SMILES to: {bad_path}")

    t1 = time.time()
    print(f"✓ Finished ChemBERTa embeddings in {t1 - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate ChemBERTa embeddings (mean/cls/max) via BioEmbedder.")
    ap.add_argument("--in_csv", required=True, help="Input CSV with header.")
    ap.add_argument("--out_prefix", required=True, help="Output prefix. Writes <prefix>_{mean,cls,max}.csv")
    ap.add_argument("--smiles_col", default=None, help="Column with SMILES (default: first column).")
    ap.add_argument("--id_col", default=None, help="Column to use as row index (default: SMILES column).")
    ap.add_argument("--model_key", default="chemberta_zinc_v1", help="Registry key in BioEmbedder.")
    ap.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing.")
    args = ap.parse_args()

    main(
        in_csv=args.in_csv,
        out_prefix=args.out_prefix,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        model_key=args.model_key,
        batch_size=args.batch_size,
    )
