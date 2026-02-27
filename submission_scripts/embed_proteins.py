#!/usr/bin/env python
"""Unified protein embedding script.

Supports four input modes
-------------------------
1. **biomart**  – BioMart CSV (gene IDs → resolve to protein sequences via API).
2. **single**   – A single gene name, Ensembl ID, UniProt ID, or raw amino-acid
                  sequence.
3. **all**      – Fetch all protein-coding genes via BioMart / Ensembl and
                  resolve to protein sequences.
4. **adata**    – An ``.h5ad`` AnnData file with an ``ensembl_id`` / gene-name
                  column.

Examples
--------
.. code-block:: bash

    # Mode 1 – from BioMart gene list
    python embed_proteins.py --input-mode biomart \\
        --mart-file genes.csv \\
        --model esm2_650M --output prot_embs.npz

    # Mode 2 – single gene
    python embed_proteins.py --input-mode single \\
        --identifier TP53 --model esmc_300m --output tp53_prot.npz

    # Mode 3 – all protein-coding genes (API)
    python embed_proteins.py --input-mode all \\
        --model esm2_650M --output all_prot.npz

    # Mode 4 – from AnnData
    python embed_proteins.py --input-mode adata \\
        --adata-path my_data.h5ad --gene-column ensembl_id \\
        --model esmc_600m --output adata_prot.npz
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embpy.models.base import BaseModelWrapper
from embpy.resources.gene_resolver import GeneResolver, detect_identifier_type

PROTEIN_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "esm2_8M": ("embpy.models.protein_models.ESM2Wrapper", "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M": ("embpy.models.protein_models.ESM2Wrapper", "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M": ("embpy.models.protein_models.ESM2Wrapper", "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": ("embpy.models.protein_models.ESM2Wrapper", "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": ("embpy.models.protein_models.ESM2Wrapper", "facebook/esm2_t36_3B_UR50D"),
    "esmc_300m": ("embpy.models.protein_models.ESMCWrapper", "esmc_300m"),
    "esmc_600m": ("embpy.models.protein_models.ESMCWrapper", "esmc_600m"),
}


def _import_wrapper(dotted_path: str) -> type[BaseModelWrapper]:
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


def _is_protein_sequence(s: str) -> bool:
    """Check if a string looks like an amino-acid sequence."""
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwyx]+", s)) and len(s) >= 10


def _to_space_separated(vec: np.ndarray) -> str:
    return " ".join(f"{x:.8g}" for x in vec.astype(np.float32))


def _resolve_protein_sequences(
    args: argparse.Namespace,
    resolver: GeneResolver,
) -> dict[str, str]:
    """Return ``{identifier: protein_sequence}`` for the chosen input mode."""

    mode = args.input_mode

    if mode == "biomart":
        logging.info("Input mode: biomart (gene IDs → protein via API)")
        return resolver.get_all_local_protein_sequences(
            mart_file=args.mart_file,
            biotype=args.biotype,
            organism=args.organism,
        )

    if mode == "single":
        identifier = args.identifier
        logging.info(f"Input mode: single – identifier='{identifier}'")

        if _is_protein_sequence(identifier):
            return {identifier[:40]: identifier}

        id_kind = detect_identifier_type(identifier)
        if id_kind == "dna_sequence":
            logging.error("Provided sequence looks like DNA, not protein. Use embed_genes.py instead.")
            sys.exit(1)

        if id_kind == "ensembl_id":
            id_type = "ensembl_id"
        elif identifier.startswith(("P", "Q", "O", "A")) and re.match(r"^[A-Z][0-9A-Z]{5}$", identifier):
            id_type = "uniprot_id"
        else:
            id_type = "symbol"

        seq = resolver.get_protein_sequence(identifier, id_type=id_type, organism=args.organism)
        if seq is None:
            logging.error(f"Could not resolve protein sequence for '{identifier}'.")
            sys.exit(1)
        return {identifier: seq}

    if mode == "all":
        logging.info("Input mode: all (fetch all protein-coding genes → protein sequences)")
        if args.mart_file:
            return resolver.get_all_local_protein_sequences(
                mart_file=args.mart_file,
                biotype=args.biotype,
                organism=args.organism,
            )
        # Fall back to pyensembl gene list
        if resolver.ensembl is None:
            logging.error("pyensembl not available; provide --mart-file for 'all' mode.")
            sys.exit(1)
        genes = resolver.ensembl.genes()
        if args.biotype.lower() != "all":
            genes = [g for g in genes if g.biotype == args.biotype]
        gene_ids = [g.gene_id for g in genes]
        logging.info(f"Fetching protein sequences for {len(gene_ids)} genes...")
        return resolver.get_protein_sequences_batch(gene_ids, id_type="ensembl_id", organism=args.organism)

    if mode == "adata":
        logging.info(f"Input mode: adata – file={args.adata_path}")
        genes = resolver.load_genes_from_adata(args.adata_path, column=args.gene_column)
        if not genes:
            logging.error("No genes found in AnnData file.")
            sys.exit(1)

        sequences: dict[str, str] = {}
        for gene in genes:
            if _is_protein_sequence(gene):
                sequences[gene[:40]] = gene
                continue

            id_kind = detect_identifier_type(gene)
            if id_kind == "ensembl_id":
                id_type = "ensembl_id"
            else:
                id_type = "symbol"

            seq = resolver.get_protein_sequence(gene, id_type=id_type, organism=args.organism)
            if seq:
                sequences[gene] = seq

        logging.info(f"Resolved {len(sequences)}/{len(genes)} protein sequences from AnnData.")
        return sequences

    raise ValueError(f"Unknown input mode: {mode}")


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description="Embed protein sequences with a chosen model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--input-mode",
        required=True,
        choices=["biomart", "single", "all", "adata"],
        help="How gene/protein identifiers are provided.",
    )

    # Biomart
    ap.add_argument("--mart-file", type=str, default=None, help="Path to BioMart CSV.")

    # Single
    ap.add_argument("--identifier", type=str, default=None, help="Gene name, Ensembl ID, UniProt ID, or AA sequence.")

    # AnnData
    ap.add_argument("--adata-path", type=str, default=None, help="Path to .h5ad file.")
    ap.add_argument("--gene-column", type=str, default=None, help="Column in adata.var for gene IDs.")

    # Shared
    ap.add_argument("--biotype", type=str, default="protein_coding", help="Gene biotype filter.")
    ap.add_argument("--organism", type=str, default="human", help="Organism (default: human).")

    # Model / embedding
    ap.add_argument("--model", type=str, required=True, choices=list(PROTEIN_MODEL_REGISTRY.keys()), help="Protein model.")
    ap.add_argument("--pooling", type=str, default="mean", help="Pooling strategy (mean, max, cls).")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda.")

    # Output
    ap.add_argument("--output", type=Path, required=True, help="Output file path (.npz or .csv).")
    ap.add_argument("--output-format", type=str, default="npz", choices=["npz", "csv"], help="Output format.")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    # Validate mode-specific args
    if args.input_mode == "biomart" and not args.mart_file:
        ap.error("--mart-file is required for biomart mode.")
    if args.input_mode == "single" and not args.identifier:
        ap.error("--identifier is required for single mode.")
    if args.input_mode == "adata" and not args.adata_path:
        ap.error("--adata-path is required for adata mode.")

    # Set up resolver
    resolver_kwargs: dict = {}
    if args.mart_file:
        resolver_kwargs["mart_file"] = args.mart_file
    resolver = GeneResolver(**resolver_kwargs)

    # Resolve sequences
    logging.info("Resolving protein sequences...")
    sequences = _resolve_protein_sequences(args, resolver)
    logging.info(f"Total protein sequences to embed: {len(sequences)}")

    if not sequences:
        logging.error("No sequences to embed. Exiting.")
        sys.exit(1)

    # Load model
    wrapper_path, model_id = PROTEIN_MODEL_REGISTRY[args.model]
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

    for i, (gene_id, prot_seq) in enumerate(items):
        try:
            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        emb = model.embed(input=prot_seq, pooling_strategy=args.pooling)
                else:
                    emb = model.embed(input=prot_seq, pooling_strategy=args.pooling)
            emb = np.asarray(emb, dtype=np.float32).ravel()
            gene_ids.append(gene_id)
            embeddings.append(emb)
            seq_lengths.append(len(prot_seq))
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
