#!/usr/bin/env python
"""Generate DNA embeddings for all protein-coding genes in the human genome.

Supports multiple models, pooling strategies, and gene regions (full/exons/introns).

Example
-------
.. code-block:: bash

    python embed_full_genome.py \\
        --model enformer_human_rough \\
        --pooling mean \\
        --region full \\
        --output data/dna_embeddings/enformer_human_rough/full_mean.npz
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


DNA_MODELS = [
    # Genomic track models
    "enformer_human_rough",
    "borzoi_v0",
    "borzoi_v1",
    # Evo2 models
    "evo2_7b",
    "evo2_40b",
    "evo2_7b_base",
    "evo2_1b_base",
    # Evo1 models
    "evo1_8k",
    "evo1_131k",
    "evo1.5_8k",
    "evo1_crispr",
    "evo1_transposon",
    # Nucleotide Transformer v1/v2
    "nt_500m_human_ref",
    "nt_500m_1000g",
    "nt_2b5_1000g",
    "nt_2b5_multi",
    "nt_v2_50m",
    "nt_v2_100m",
    "nt_v2_250m",
    "nt_v2_500m",
    # Nucleotide Transformer v3
    "ntv3_8m_pre",
    "ntv3_100m_pre",
    "ntv3_100m_pos",
    "ntv3_650m_pre",
    "ntv3_650m_pos",
    # GENA-LM
    "gena_lm_bert_base",
    "gena_lm_bert_large",
    "gena_lm_bert_base_multi",
    "gena_lm_bigbird_base",
    # HyenaDNA
    "hyenadna_tiny_1k",
    "hyenadna_small_32k",
    "hyenadna_medium_160k",
    "hyenadna_medium_450k",
    "hyenadna_large_1m",
    # Caduceus
    "caduceus_ph_131k",
    "caduceus_ps_131k",
]

POOLING_STRATEGIES = ["mean", "max", "cls"]
REGIONS = ["full", "exons", "introns"]


def get_device(device_str: str) -> torch.device:
    """Get the torch device based on string specification."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def embed_all_genes(
    model_name: str,
    pooling: str,
    region: Literal["full", "exons", "introns"],
    organism: str,
    biotype: str,
    device: torch.device,
    checkpoint_dir: Path | None = None,
) -> tuple[list[str], list[np.ndarray], list[int]]:
    """Embed all genes using the BioEmbedder.

    Parameters
    ----------
    model_name : str
        Name of the DNA model to use.
    pooling : str
        Pooling strategy (mean, max, cls).
    region : {"full", "exons", "introns"}
        Which part of the gene to embed.
    organism : str
        Organism name (e.g., "human").
    biotype : str
        Gene biotype filter (e.g., "protein_coding").
    device : torch.device
        Device to run inference on.
    checkpoint_dir : Path, optional
        Directory to save intermediate checkpoints.

    Returns
    -------
    gene_ids : list[str]
        List of gene identifiers.
    embeddings : list[np.ndarray]
        List of embedding vectors.
    seq_lengths : list[int]
        List of sequence lengths.
    """
    from embpy.embedder import BioEmbedder

    logger.info(f"Initializing BioEmbedder with device={device}")
    embedder = BioEmbedder(device=device)

    logger.info(f"Fetching all {biotype} genes for {organism}...")
    gene_resolver = embedder.gene_resolver

    sequences = gene_resolver.get_gene_sequences(biotype=biotype)
    if not sequences:
        logger.error("Failed to retrieve gene sequences")
        return [], [], []
    logger.info(f"Retrieved {len(sequences)} gene sequences")

    gene_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    seq_lengths: list[int] = []

    checkpoint_file = None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"checkpoint_{model_name}_{region}_{pooling}.npz"
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            ckpt = np.load(checkpoint_file, allow_pickle=True)
            gene_ids = list(ckpt["gene_ids"])
            embeddings = [np.array(e) for e in ckpt["embeddings"]]
            seq_lengths = list(ckpt["seq_lengths"])
            logger.info(f"Resumed from checkpoint with {len(gene_ids)} genes")

    processed_genes = set(gene_ids)
    items = list(sequences.items())
    total = len(items)
    start_time = time.time()

    for i, (gene_id, dna_seq) in enumerate(items):
        if gene_id in processed_genes:
            continue

        try:
            emb = embedder.embed_gene(
                identifier=gene_id,
                model=model_name,
                id_type="ensembl_id",
                organism=organism,
                pooling_strategy=pooling,
                region=region,
            )

            if emb is not None:
                emb_np = np.asarray(emb, dtype=np.float32).ravel()
                gene_ids.append(gene_id)
                embeddings.append(emb_np)
                seq_len = len(dna_seq) if dna_seq else 0
                seq_lengths.append(seq_len)

        except Exception as e:
            logger.warning(f"Failed to embed {gene_id}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i + 1}/{total} ({100 * (i + 1) / total:.1f}%) | "
                f"Embedded: {len(embeddings)} | "
                f"Rate: {rate:.2f} genes/s | ETA: {eta / 60:.1f} min"
            )

            if checkpoint_file and len(embeddings) > 0:
                np.savez(
                    checkpoint_file,
                    gene_ids=np.array(gene_ids, dtype=str),
                    embeddings=np.array(embeddings, dtype=object),
                    seq_lengths=np.array(seq_lengths, dtype=int),
                )

        if device.type == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Successfully embedded {len(embeddings)}/{total} genes")
    return gene_ids, embeddings, seq_lengths


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate DNA embeddings for all protein-coding genes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=DNA_MODELS,
        help="DNA model to use for embedding.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=POOLING_STRATEGIES,
        help="Pooling strategy (default: mean).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        choices=REGIONS,
        help="Gene region to embed: full, exons, or introns (default: full).",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="human",
        help="Organism (default: human).",
    )
    parser.add_argument(
        "--biotype",
        type=str,
        default="protein_coding",
        help="Gene biotype filter (default: protein_coding).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path (.npz).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for intermediate checkpoints (optional).",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Full Genome DNA Embedding")
    logger.info("=" * 70)
    logger.info(f"Model:    {args.model}")
    logger.info(f"Pooling:  {args.pooling}")
    logger.info(f"Region:   {args.region}")
    logger.info(f"Organism: {args.organism}")
    logger.info(f"Biotype:  {args.biotype}")
    logger.info(f"Output:   {args.output}")
    logger.info("=" * 70)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    gene_ids, embeddings, seq_lengths = embed_all_genes(
        model_name=args.model,
        pooling=args.pooling,
        region=args.region,
        organism=args.organism,
        biotype=args.biotype,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    if not embeddings:
        logger.error("No embeddings generated. Exiting.")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    emb_matrix = np.stack(embeddings)

    np.savez(
        args.output,
        gene_ids=np.array(gene_ids, dtype=str),
        embeddings=emb_matrix,
        seq_lengths=np.array(seq_lengths, dtype=int),
        model=args.model,
        pooling=args.pooling,
        region=args.region,
        organism=args.organism,
        biotype=args.biotype,
    )

    logger.info(f"Saved {len(gene_ids)} embeddings to {args.output}")
    logger.info(f"Embedding matrix shape: {emb_matrix.shape}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
