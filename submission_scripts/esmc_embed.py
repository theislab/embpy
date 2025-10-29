#!/usr/bin/env python3
"""
Generate ESMC protein embeddings for a list of Ensembl gene IDs.

Inputs
------
- Plain text file with one Ensembl gene ID per line, OR
- CSV/TSV with a column named 'ensembl_gene_id'

Outputs
-------
CSV with columns:
  ensembl_gene_id, ensembl_transcript_id, seq_len, pooling, model,
  embedding_dim, embedding
Where `embedding` is a space-separated list of floats.

Example
-------
python esmc_embed.py \
  --input /path/to/gene_ids.txt \
  --output /path/to/embeddings_esmc_mean.csv \
  --model esmc_300m --pooling mean --batch-size 16 --device auto
"""

import argparse
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

# If your ESMCWrapper lives in a module, import it here:
# from your_package.wrappers import ESMCWrapper
# For this script, we assume ESMCWrapper is importable in the environment:
from ESMCWrapper import ESMCWrapper  # adjust if needed
from pyensembl import EnsemblRelease

# ------------------ Ensembl helpers ------------------


def load_ensembl(release: int = 109) -> EnsemblRelease:
    ens = EnsemblRelease(release)
    # If running the first time on a machine without cache, uncomment:
    # ens.download()
    # ens.index()
    return ens


def gene_to_best_protein(
    ens: EnsemblRelease,
    ensembl_gene_id: str,
    min_len: int = 20,
) -> tuple[str, str] | None:
    """Return (transcript_id, protein_sequence) for the longest protein-coding transcript."""
    try:
        gene = ens.gene_by_id(ensembl_gene_id)
    except Exception:
        return None

    best_tid, best_seq, best_len = None, None, -1
    for tx in gene.transcripts:
        try:
            if getattr(tx, "biotype", None) != "protein_coding":
                continue
            prot = tx.protein_sequence
            if prot is None:
                continue
            seq = str(prot).strip()
            if len(seq) < min_len:
                continue
        except Exception:
            continue
        if len(seq) > best_len:
            best_len = len(seq)
            best_tid = tx.transcript_id
            best_seq = seq

    if best_tid is None or best_seq is None:
        return None
    return best_tid, best_seq


def genes_to_proteins(
    ensembl_gene_ids: Sequence[str],
    release: int = 109,
    min_len: int = 20,
) -> dict[str, tuple[str, str]]:
    ens = load_ensembl(release)
    out: dict[str, tuple[str, str]] = {}
    for gid in ensembl_gene_ids:
        res = gene_to_best_protein(ens, gid, min_len=min_len)
        if res is not None:
            out[gid] = res
    return out


# ------------------ IO helpers ------------------


def load_gene_ids(path: str) -> list[str]:
    """
    Load Ensembl gene IDs from:
      - .txt: one ID per line
      - .csv/.tsv: must contain column 'ensembl_gene_id'
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        if "ensembl_gene_id" not in df.columns:
            raise KeyError(f"Input {path} must contain column 'ensembl_gene_id'")
        ids = df["ensembl_gene_id"].astype(str).str.strip().tolist()
    else:
        # assume text file with one id per line
        with open(path) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    # de-duplicate, preserve order
    seen = set()
    unique_ids = []
    for gid in ids:
        if gid not in seen:
            seen.add(gid)
            unique_ids.append(gid)
    return unique_ids


def embedding_to_str(vec: np.ndarray) -> str:
    """Space-separated floats (compact, stable)."""
    return " ".join(f"{x:.8g}" for x in np.asarray(vec, dtype=np.float32))


# ------------------ Main embedding pipeline ------------------


def run(
    input_path: str,
    output_path: str,
    model_name: str = "esmc_300m",
    pooling: str = "mean",
    batch_size: int = 16,
    device_str: str = "auto",
    min_aa_len: int = 20,
    ensembl_release: int = 109,
) -> None:
    # 1) Load gene IDs
    gene_ids = load_gene_ids(input_path)
    if not gene_ids:
        raise RuntimeError("No gene IDs found in input.")
    print(f"[info] Loaded {len(gene_ids)} gene IDs from {input_path}")

    # 2) Map genes -> protein sequences (canonical = longest protein-coding)
    mapping = genes_to_proteins(gene_ids, release=ensembl_release, min_len=min_aa_len)
    if not mapping:
        raise RuntimeError("No protein sequences resolved for any gene IDs.")
    print(f"[info] Resolved proteins for {len(mapping)} / {len(gene_ids)} genes")

    # 3) Prepare embedding client
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[info] Using device: {device}")

    wrapper = ESMCWrapper(model_path_or_name=model_name)
    wrapper.load(device=device)

    # 4) Batch embed
    gids = list(mapping.keys())
    txids = [mapping[g][0] for g in gids]
    seqs = [mapping[g][1] for g in gids]
    print(f"[info] Embedding {len(seqs)} proteins (batch_size={batch_size}, pooling={pooling})")

    rows = []
    N = len(seqs)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_seqs = seqs[start:end]
        batch_gids = gids[start:end]
        batch_txids = txids[start:end]

        outs = wrapper.embed_batch(batch_seqs, pooling_strategy=pooling, return_hidden_states=False)
        for gid, tid, seq, out in zip(batch_gids, batch_txids, batch_seqs, outs, strict=False):
            emb = out["embedding"]
            rows.append(
                {
                    "ensembl_gene_id": gid,
                    "ensembl_transcript_id": tid,
                    "seq_len": len(seq),
                    "pooling": pooling,
                    "model": model_name,
                    "embedding_dim": int(emb.shape[0]),
                    "embedding": embedding_to_str(emb),
                }
            )

        if (end % max(batch_size, 1)) == 0 or end == N:
            print(f"[prog] {end}/{N} done", flush=True)

    df = pd.DataFrame(rows)
    # 5) Save
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[done] Wrote {len(df)} rows to {output_path}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Generate ESMC embeddings for Ensembl genes.")
    ap.add_argument("--input", required=True, help="Input file: .txt (one ID/line) or CSV/TSV with 'ensembl_gene_id'.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--model", default="esmc_300m", help="ESMC checkpoint (e.g., esmc_300m, esmc_600m).")
    ap.add_argument("--pooling", choices=["mean", "max", "cls"], default="mean", help="Pooling strategy.")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or 'cuda:0', etc.")
    ap.add_argument("--min-aa-len", type=int, default=20, help="Minimum AA length to keep a protein.")
    ap.add_argument("--ensembl-release", type=int, default=109, help="Ensembl release (109=GRCh38).")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device_str=args.device,
        min_aa_len=args.min_aa_len,
        ensembl_release=args.ensembl_release,
    )
