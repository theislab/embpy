#!/usr/bin/env python
"""Generate protein embeddings for all protein-coding genes.

Fetches canonical or isoform protein sequences and embeds them with
a specified protein language model and pooling strategy.

Example
-------
.. code-block:: bash

    python embed_full_proteome.py \\
        --model esm2_650M \\
        --pooling mean \\
        --isoform canonical \\
        --output data/embeddings/protein_embeddings/esm2_650M/canonical_mean.npz
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


PROTEIN_MODELS = [
    # ESM-1b
    "esm1b",
    # ESM-2
    "esm2_8M",
    "esm2_35M",
    "esm2_150M",
    "esm2_650M",
    "esm2_3B",
    # ESM-C
    "esmc_300m",
    "esmc_600m",
    # ESM3
    "esm3_small",
    # ProtT5
    "prot_t5_xl",
    "prot_t5_xl_half",
]

POOLING_STRATEGIES = ["mean", "max", "cls"]
ISOFORM_MODES = ["canonical", "all"]


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings for all protein-coding genes.",
    )
    parser.add_argument("--model", type=str, required=True, choices=PROTEIN_MODELS)
    parser.add_argument("--pooling", type=str, default="mean", choices=POOLING_STRATEGIES)
    parser.add_argument("--isoform", type=str, default="canonical", choices=ISOFORM_MODES)
    parser.add_argument("--organism", type=str, default="human")
    parser.add_argument("--biotype", type=str, default="protein_coding")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Base data directory with pre-downloaded sequences.")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Full Proteome Protein Embedding")
    logger.info("=" * 70)
    logger.info(f"Model:    {args.model}")
    logger.info(f"Pooling:  {args.pooling}")
    logger.info(f"Isoform:  {args.isoform}")
    logger.info(f"Organism: {args.organism}")
    logger.info(f"Biotype:  {args.biotype}")
    logger.info(f"Output:   {args.output}")
    logger.info("=" * 70)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    from embpy.embedder import BioEmbedder

    embedder = BioEmbedder(device=device)

    if args.isoform == "canonical":
        _embed_canonical(embedder, args)
    else:
        _embed_all_isoforms(embedder, args)


def _try_load_protein_sequences(data_dir: str | None) -> dict[str, str] | None:
    """Try to load pre-downloaded canonical protein sequences from disk."""
    if data_dir is None:
        return None
    seq_dir = Path(data_dir) / "proteome" / "canonical"
    if not seq_dir.exists():
        return None
    for npz_file in seq_dir.glob("*canonical_proteins*.npz"):
        logger.info("Loading pre-downloaded protein sequences from %s", npz_file)
        data = np.load(npz_file, allow_pickle=True)
        gene_ids = list(data["gene_ids"])
        seqs = list(data["sequences"])
        logger.info("Loaded %d canonical protein sequences from disk (instant!)", len(gene_ids))
        return dict(zip(gene_ids, seqs))
    return None


def _try_load_isoform_sequences(data_dir: str | None) -> dict[str, dict[str, str]] | None:
    """Try to load pre-downloaded isoform sequences from disk.

    Returns {gene_id: {isoform_accession: sequence}}.
    """
    if data_dir is None:
        return None
    seq_dir = Path(data_dir) / "proteome" / "isoforms"
    if not seq_dir.exists():
        return None
    for npz_file in seq_dir.glob("*all_isoforms*.npz"):
        logger.info("Loading pre-downloaded isoform sequences from %s", npz_file)
        data = np.load(npz_file, allow_pickle=True)
        gene_ids = list(data["gene_ids"])
        isoform_ids = list(data["isoform_ids"])
        seqs = list(data["sequences"])
        result: dict[str, dict[str, str]] = {}
        for gid, iso_id, seq in zip(gene_ids, isoform_ids, seqs):
            result.setdefault(gid, {})[iso_id] = seq
        logger.info(
            "Loaded %d isoforms from %d genes from disk (instant!)",
            len(seqs), len(result),
        )
        return result
    return None


def _embed_canonical(embedder, args) -> None:
    """Embed canonical protein for every gene."""
    from embpy.resources.protein_resolver import ProteinResolver

    preloaded = _try_load_protein_sequences(getattr(args, "data_dir", None))

    if preloaded:
        gene_ids = list(preloaded.keys())
        logger.info("Using %d pre-downloaded protein sequences", len(gene_ids))
    else:
        pr = ProteinResolver(organism=args.organism)
        logger.info("Fetching all %s gene IDs...", args.biotype)
        gene_ids = pr._get_all_gene_ids(biotype=args.biotype)
        if not gene_ids:
            logger.error("No genes found")
            sys.exit(1)
        logger.info("Found %d genes", len(gene_ids))

    checkpoint_file = None
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = args.checkpoint_dir / f"ckpt_protein_{args.model}_{args.isoform}_{args.pooling}.npz"

    gene_ids_done: list[str] = []
    embeddings_done: list[np.ndarray] = []

    if checkpoint_file and checkpoint_file.exists():
        ckpt = np.load(checkpoint_file, allow_pickle=True)
        gene_ids_done = list(ckpt["gene_ids"])
        embeddings_done = [np.array(e) for e in ckpt["embeddings"]]
        logger.info("Resumed from checkpoint with %d genes", len(gene_ids_done))

    processed = set(gene_ids_done)
    total = len(gene_ids)
    start_time = time.time()

    for i, gene_id in enumerate(gene_ids):
        if gene_id in processed:
            continue

        try:
            if preloaded and gene_id in preloaded:
                seq = preloaded[gene_id]
                emb = embedder.embed_protein(
                    identifier=seq,
                    model=args.model,
                    id_type="sequence",
                    pooling_strategy=args.pooling,
                )
            else:
                emb = embedder.embed_protein(
                    identifier=gene_id,
                    model=args.model,
                    id_type="ensembl_id",
                    organism=args.organism,
                    pooling_strategy=args.pooling,
                    isoform="canonical",
                )
            if emb is not None:
                gene_ids_done.append(gene_id)
                embeddings_done.append(np.asarray(emb, dtype=np.float32).ravel())
        except Exception as e:
            logger.warning("Failed %s: %s", gene_id, e)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | Embedded: %d | Rate: %.2f genes/s | ETA: %.1f min",
                i + 1,
                total,
                100 * (i + 1) / total,
                len(embeddings_done),
                rate,
                eta,
            )
            if checkpoint_file and embeddings_done:
                np.savez(
                    checkpoint_file,
                    gene_ids=np.array(gene_ids_done, dtype=str),
                    embeddings=np.array(embeddings_done, dtype=object),
                )

        if torch.cuda.is_available() and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Embedded %d/%d genes", len(embeddings_done), total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if embeddings_done:
        emb_matrix = np.stack(embeddings_done)
        np.savez(
            args.output,
            gene_ids=np.array(gene_ids_done, dtype=str),
            embeddings=emb_matrix,
            model=args.model,
            pooling=args.pooling,
            isoform="canonical",
            organism=args.organism,
        )
        logger.info("Saved %d embeddings to %s (shape %s)", len(gene_ids_done), args.output, emb_matrix.shape)
    else:
        logger.error("No embeddings generated")
        sys.exit(1)


def _embed_all_isoforms(embedder, args) -> None:
    """Embed all isoforms for every gene."""
    from embpy.resources.protein_resolver import ProteinResolver

    preloaded_iso = _try_load_isoform_sequences(getattr(args, "data_dir", None))

    if preloaded_iso:
        gene_ids = list(preloaded_iso.keys())
        logger.info("Using %d genes with pre-downloaded isoforms", len(gene_ids))
    else:
        pr = ProteinResolver(organism=args.organism)
        logger.info("Fetching all %s gene IDs...", args.biotype)
        gene_ids = pr._get_all_gene_ids(biotype=args.biotype)

    if not gene_ids:
        logger.error("No genes found")
        sys.exit(1)
    logger.info("Found %d genes", len(gene_ids))

    all_ids: list[str] = []
    all_gene_ids: list[str] = []
    all_isoform_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    total = len(gene_ids)
    start_time = time.time()

    for i, gene_id in enumerate(gene_ids):
        try:
            if preloaded_iso and gene_id in preloaded_iso:
                iso_seqs = preloaded_iso[gene_id]
                for iso_id, seq in iso_seqs.items():
                    emb = embedder.embed_protein(
                        identifier=seq,
                        model=args.model,
                        id_type="sequence",
                        pooling_strategy=args.pooling,
                    )
                    if emb is not None:
                        all_ids.append(f"{gene_id}|{iso_id}")
                        all_gene_ids.append(gene_id)
                        all_isoform_ids.append(iso_id)
                        all_embeddings.append(np.asarray(emb, dtype=np.float32).ravel())
            else:
                result = embedder.embed_protein(
                    identifier=gene_id,
                    model=args.model,
                    id_type="ensembl_id",
                    organism=args.organism,
                    pooling_strategy=args.pooling,
                    isoform="all",
                )
                if isinstance(result, dict):
                    for iso_id, emb in result.items():
                        all_ids.append(f"{gene_id}|{iso_id}")
                        all_gene_ids.append(gene_id)
                        all_isoform_ids.append(iso_id)
                        all_embeddings.append(np.asarray(emb, dtype=np.float32).ravel())
        except Exception as e:
            logger.warning("Failed %s: %s", gene_id, e)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | Isoforms embedded: %d | Rate: %.2f genes/s | ETA: %.1f min",
                i + 1,
                total,
                100 * (i + 1) / total,
                len(all_embeddings),
                rate,
                eta,
            )

        if torch.cuda.is_available() and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Embedded %d isoforms from %d genes", len(all_embeddings), total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if all_embeddings:
        emb_matrix = np.stack(all_embeddings)
        np.savez(
            args.output,
            ids=np.array(all_ids, dtype=str),
            gene_ids=np.array(all_gene_ids, dtype=str),
            isoform_ids=np.array(all_isoform_ids, dtype=str),
            embeddings=emb_matrix,
            model=args.model,
            pooling=args.pooling,
            isoform="all",
            organism=args.organism,
        )
        logger.info("Saved %d isoform embeddings to %s (shape %s)", len(all_ids), args.output, emb_matrix.shape)
    else:
        logger.error("No embeddings generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
