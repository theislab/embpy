#!/usr/bin/env python
"""Unified gene (DNA) embedding script.

Supports four input modes
-------------------------
1. **biomart**  – BioMart CSV + genome FASTA directory  → bulk local sequences.
2. **single**   – A single gene name, Ensembl ID, or raw DNA sequence.
3. **all**      – Fetch all protein-coding genes via the Ensembl REST API.
4. **adata**    – An ``.h5ad`` AnnData file with an ``ensembl_id`` / gene-name column.

Examples
--------
.. code-block:: bash

    # Mode 1 – local BioMart
    python embed_genes.py --input-mode biomart \\
        --mart-file genes.csv --genome-dir /data/genome \\
        --model enformer_human_rough --output embeddings.npz

    # Mode 2 – single gene
    python embed_genes.py --input-mode single \\
        --identifier TP53 --model borzoi_v0 --output tp53.npz

    # Mode 3 – all protein-coding genes
    python embed_genes.py --input-mode all \\
        --model enformer_human_rough --output all_genes.npz

    # Mode 4 – from AnnData
    python embed_genes.py --input-mode adata \\
        --adata-path my_data.h5ad --gene-column ensembl_id \\
        --model evo2_7b --output adata_embs.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embpy.models.base import BaseModelWrapper
from embpy.resources.gene_resolver import GeneResolver, detect_identifier_type

# Model name → (WrapperClass import path, HF-or-local identifier)
DNA_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "enformer_human_rough": ("embpy.models.dna_models.EnformerWrapper", "EleutherAI/enformer-official-rough"),
    "borzoi_v0": ("embpy.models.dna_models.BorzoiWrapper", "johahi/borzoi-replicate-0"),
    "borzoi_v1": ("embpy.models.dna_models.BorzoiWrapper", "johani/borzoi-replicate-1"),
    "evo2_7b": ("embpy.models.dna_models.Evo2Wrapper", "evo2_7b"),
    "evo2_40b": ("embpy.models.dna_models.Evo2Wrapper", "evo2_40b"),
    "evo2_7b_base": ("embpy.models.dna_models.Evo2Wrapper", "evo2_7b_base"),
    "evo2_1b_base": ("embpy.models.dna_models.Evo2Wrapper", "evo2_1b_base"),
}


def _import_wrapper(dotted_path: str) -> type[BaseModelWrapper]:
    """Dynamically import a wrapper class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def _to_space_separated(vec: np.ndarray) -> str:
    return " ".join(f"{x:.8g}" for x in vec.astype(np.float32))


def _resolve_sequences(
    args: argparse.Namespace,
    resolver: GeneResolver,
) -> dict[str, str]:
    """Return ``{identifier: dna_sequence}`` for the chosen input mode."""

    mode = args.input_mode

    if mode == "biomart":
        logging.info("Input mode: biomart (local bulk)")
        return resolver.load_sequences_from_biomart(
            mart_file=args.mart_file,
            chrom_folder=args.genome_dir,
            biotype=args.biotype,
        )

    if mode == "single":
        identifier = args.identifier
        logging.info(f"Input mode: single – identifier='{identifier}'")
        id_kind = detect_identifier_type(identifier)

        if id_kind == "dna_sequence":
            return {identifier[:40]: identifier}

        if id_kind == "ensembl_id":
            if args.mart_file and args.genome_dir:
                seq = resolver.get_local_dna_sequence(identifier, id_type="ensembl_id")
            else:
                seq = resolver.get_dna_sequence(identifier, id_type="ensembl_id", organism=args.organism)
        else:
            if args.mart_file and args.genome_dir:
                seq = resolver.get_local_dna_sequence(identifier, id_type="symbol")
            else:
                seq = resolver.get_dna_sequence(identifier, id_type="symbol", organism=args.organism)

        if seq is None:
            logging.error(f"Could not resolve sequence for '{identifier}'.")
            sys.exit(1)
        return {identifier: seq}

    if mode == "all":
        logging.info("Input mode: all (Ensembl REST API fetch of all genes)")
        seqs = resolver.get_gene_sequences(biotype=args.biotype)
        if not seqs:
            logging.error("No sequences retrieved.")
            sys.exit(1)
        return seqs

    if mode == "adata":
        logging.info(f"Input mode: adata – file={args.adata_path}")
        genes = resolver.load_genes_from_adata(args.adata_path, column=args.gene_column)
        if not genes:
            logging.error("No genes found in AnnData file.")
            sys.exit(1)

        sequences: dict[str, str] = {}
        for gene in genes:
            id_kind = detect_identifier_type(gene)
            if id_kind == "dna_sequence":
                sequences[gene[:40]] = gene
            elif id_kind == "ensembl_id":
                if args.mart_file and args.genome_dir:
                    seq = resolver.get_local_dna_sequence(gene, id_type="ensembl_id")
                else:
                    seq = resolver.get_dna_sequence(gene, id_type="ensembl_id", organism=args.organism)
                if seq:
                    sequences[gene] = seq
            else:
                if args.mart_file and args.genome_dir:
                    seq = resolver.get_local_dna_sequence(gene, id_type="symbol")
                else:
                    seq = resolver.get_dna_sequence(gene, id_type="symbol", organism=args.organism)
                if seq:
                    sequences[gene] = seq
        logging.info(f"Resolved {len(sequences)}/{len(genes)} gene sequences from AnnData.")
        return sequences

    raise ValueError(f"Unknown input mode: {mode}")


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description="Embed gene DNA sequences with a chosen model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input mode
    ap.add_argument(
        "--input-mode",
        required=True,
        choices=["biomart", "single", "all", "adata"],
        help="How gene sequences are provided.",
    )

    # Biomart / local args
    ap.add_argument("--mart-file", type=str, default=None, help="Path to BioMart CSV (for biomart/single/adata modes).")
    ap.add_argument("--genome-dir", type=str, default=None, help="Path to chromosome FASTA directory.")

    # Single mode
    ap.add_argument("--identifier", type=str, default=None, help="Gene name, Ensembl ID, or raw DNA sequence.")

    # AnnData mode
    ap.add_argument("--adata-path", type=str, default=None, help="Path to .h5ad file.")
    ap.add_argument("--gene-column", type=str, default=None, help="Column in adata.var to use for gene IDs.")

    # All mode
    ap.add_argument("--biotype", type=str, default="protein_coding", help="Gene biotype filter (default: protein_coding).")
    ap.add_argument("--organism", type=str, default="human", help="Organism (default: human).")

    # Model / embedding
    ap.add_argument("--model", type=str, required=True, choices=list(DNA_MODEL_REGISTRY.keys()), help="DNA model to use.")
    ap.add_argument("--pooling", type=str, default="mean", help="Pooling strategy (mean, max, cls).")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")

    # Output
    ap.add_argument("--output", type=Path, required=True, help="Output file path (.npz or .csv).")
    ap.add_argument("--output-format", type=str, default="npz", choices=["npz", "csv"], help="Output format.")

    # Performance
    ap.add_argument("--batch-size", type=int, default=1, help="Sequences to embed before writing (memory trade-off).")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    # Validate mode-specific args
    if args.input_mode == "biomart" and (not args.mart_file or not args.genome_dir):
        ap.error("--mart-file and --genome-dir are required for biomart mode.")
    if args.input_mode == "single" and not args.identifier:
        ap.error("--identifier is required for single mode.")
    if args.input_mode == "adata" and not args.adata_path:
        ap.error("--adata-path is required for adata mode.")

    # Set up resolver
    resolver_kwargs: dict = {}
    if args.mart_file and args.genome_dir:
        resolver_kwargs["mart_file"] = args.mart_file
        resolver_kwargs["chromosome_folder"] = args.genome_dir
    resolver = GeneResolver(**resolver_kwargs)

    # Resolve sequences
    logging.info("Resolving gene sequences...")
    sequences = _resolve_sequences(args, resolver)
    logging.info(f"Total sequences to embed: {len(sequences)}")

    if not sequences:
        logging.error("No sequences to embed. Exiting.")
        sys.exit(1)

    # Load model
    wrapper_path, model_id = DNA_MODEL_REGISTRY[args.model]
    WrapperCls = _import_wrapper(wrapper_path)

    device = _get_device(args.device)
    logging.info(f"Loading model '{args.model}' ({model_id}) on {device}...")

    model = WrapperCls(model_path_or_name=model_id)
    model.load(device=device)
    logging.info("Model loaded.")

    use_amp = device.type == "cuda" and torch.cuda.is_available()

    # Embed
    gene_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    seq_lengths: list[int] = []
    items = list(sequences.items())

    for i, (gene_id, dna_seq) in enumerate(items):
        dna = dna_seq.upper().replace("U", "T")
        try:
            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        emb = model.embed(input=dna, pooling_strategy=args.pooling)
                else:
                    emb = model.embed(input=dna, pooling_strategy=args.pooling)
            emb = np.asarray(emb, dtype=np.float32).ravel()
            gene_ids.append(gene_id)
            embeddings.append(emb)
            seq_lengths.append(len(dna))
        except Exception as e:
            logging.warning(f"Failed to embed {gene_id}: {e}")

        if (i + 1) % 50 == 0:
            logging.info(f"Embedded {i + 1}/{len(items)} sequences.")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logging.info(f"Successfully embedded {len(embeddings)}/{len(items)} sequences.")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "npz":
        emb_matrix = np.stack(embeddings) if embeddings else np.empty((0, 0))
        np.savez(
            args.output,
            gene_ids=np.array(gene_ids, dtype=str),
            embeddings=emb_matrix,
            seq_lengths=np.array(seq_lengths, dtype=int),
            model=args.model,
            pooling=args.pooling,
        )
        logging.info(f"Saved {len(gene_ids)} embeddings to {args.output} (npz).")
    else:
        rows = []
        for gid, emb, slen in zip(gene_ids, embeddings, seq_lengths):
            rows.append({
                "gene_id": gid,
                "embedding_dim": int(emb.shape[0]),
                "model": args.model,
                "pooling": args.pooling,
                "seq_len": slen,
                "embedding": _to_space_separated(emb),
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        logging.info(f"Saved {len(rows)} embeddings to {args.output} (csv).")

    logging.info("Done.")


if __name__ == "__main__":
    main()
