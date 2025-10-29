#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from embpy.models.dna_models import EnformerWrapper


def infer_pooling_from_name(name: str) -> str:
    n = name.lower()
    if "max" in n:
        return "max"
    if "median" in n:
        return "median"
    return "mean"


def read_fasta(path: Path):
    for rec in SeqIO.parse(str(path), "fasta"):
        yield rec.id, str(rec.seq)


def extract_ensembl_from_header(h: str) -> str:
    """
    Expect headers like '>ENSG00000123456|GENE|source=...'
    Returns the left-most token split by '|' (or whole header if no '|').
    Strips any Ensembl version suffix '.<v>'.
    """
    gid = h.split("|", 1)[0].strip()
    # drop FASTA leading '>' if present (Bio.SeqIO gives ids without '>')
    gid = gid.split()[0]
    if "." in gid and gid.upper().startswith("ENSG"):
        gid = gid.split(".", 1)[0]
    return gid


def to_space_separated(vec: np.ndarray) -> str:
    # Create a compact space-separated string for CSV
    # (avoids brackets/commas so CSV stays simple)
    return " ".join(f"{x:.8g}" for x in vec.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description="Embed FASTA sequences with Enformer (single-sequence API).")
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--enformer_id", type=str, default="EleutherAI/enformer-official-rough")
    ap.add_argument("--glob", type=str, default="*.fasta")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        print(f"No FASTA files in {args.input_dir} matching {args.glob}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    model = EnformerWrapper(args.enformer_id)
    model.load(device=device)
    model.model.eval()
    print(f"[enformer] {args.enformer_id} on {args.device}")

    # Use bf16/amp on CUDA-capable GPUs if available to reduce memory;
    # safe no-op on CPU/MPS.
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

    for fasta in files:
        pooling = infer_pooling_from_name(fasta.name)
        print(f"\n[file] {fasta.name}  pooling={pooling}")

        out_csv = args.out_dir / f"{fasta.stem}_embeddings.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Write header (overwrite per FASTA)
        pd.DataFrame(
            columns=["ensembl_gene_id", "embedding_dim", "model", "pooling", "source_file", "seq_len", "embedding"]
        ).to_csv(out_csv, index=False)

        n_written = 0
        for rec_id, seq in read_fasta(fasta):
            gene_id = extract_ensembl_from_header(rec_id)
            # normalize DNA (replace U->T, upper-case)
            dna = seq.upper().replace("U", "T")

            # single-sequence embedding
            with torch.inference_mode():
                # (Note: wrapper.embed handles preprocessing to the model length)
                emb = model.embed(dna, pooling_strategy=pooling)  # returns 1D numpy array or torch tensor
                emb = np.asarray(emb, dtype=np.float32).ravel()

            row = {
                "ensembl_gene_id": gene_id,
                "embedding_dim": int(emb.shape[0]),
                "model": "enformer",
                "pooling": pooling,
                "source_file": fasta.name,
                "seq_len": len(dna),
                "embedding": to_space_separated(emb),
            }
            # append one row at a time to keep memory low
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=False, index=False)

            n_written += 1
            if n_written % 10 == 0:
                print(f"  wrote {n_written} rows -> {out_csv.name}")

            # free cached GPU memory between sequences
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        print(f"[done] {fasta.name}: wrote {n_written} rows → {out_csv}")

    print("\nAll Enformer jobs complete.")


if __name__ == "__main__":
    main()
