#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from embpy.models.dna_models import BorzoiWrapper


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
    Returns the left-most token split by '|' (or the whole header if no '|').
    Strips any Ensembl version suffix '.<v>'.
    """
    gid = h.split("|", 1)[0].strip()
    gid = gid.split()[0]
    if "." in gid and gid.upper().startswith("ENSG"):
        gid = gid.split(".", 1)[0]
    return gid


def to_space_separated(vec: np.ndarray) -> str:
    # compact space-separated string for CSV
    return " ".join(f"{x:.8g}" for x in vec.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description="Embed FASTA sequences with Borzoi (single-sequence API).")
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--borzoi_id", type=str, default="johahi/borzoi-replicate-0")
    ap.add_argument("--glob", type=str, default="*.fasta")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        print(f"No FASTA files in {args.input_dir} matching {args.glob}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    model = BorzoiWrapper(args.borzoi_id)
    model.load(device=device)
    # some wrappers expose .model; if not, this is harmless
    if hasattr(model, "model"):
        model.model.eval()
    print(f"[borzoi] {args.borzoi_id} on {args.device}")

    # AMP only relevant on CUDA; harmless otherwise
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()

    for fasta in files:
        pooling = infer_pooling_from_name(fasta.name)
        print(f"\n[file] {fasta.name}  pooling={pooling}")

        out_csv = args.out_dir / f"{fasta.stem}_embeddings.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # fresh header per FASTA (overwrite)
        pd.DataFrame(
            columns=["ensembl_gene_id", "embedding_dim", "model", "pooling", "source_file", "seq_len", "embedding"]
        ).to_csv(out_csv, index=False)

        n_written = 0
        for rec_id, seq in read_fasta(fasta):
            gene_id = extract_ensembl_from_header(rec_id)
            dna = seq.upper().replace("U", "T")

            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        emb = model.embed(input=dna, pooling_strategy=pooling)
                else:
                    emb = model.embed(input=dna, pooling_strategy=pooling)

            emb = np.asarray(emb, dtype=np.float32).ravel()

            row = {
                "ensembl_gene_id": gene_id,
                "embedding_dim": int(emb.shape[0]),
                "model": "borzoi",
                "pooling": pooling,
                "source_file": fasta.name,
                "seq_len": len(dna),
                "embedding": to_space_separated(emb),
            }
            # append row
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=False, index=False)

            n_written += 1
            if n_written % 10 == 0:
                print(f"  wrote {n_written} rows -> {out_csv.name}")

            # free cached GPU memory between sequences
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        print(f"[done] {fasta.name}: wrote {n_written} rows → {out_csv}")

    print("\nAll Borzoi jobs complete.")


if __name__ == "__main__":
    main()
