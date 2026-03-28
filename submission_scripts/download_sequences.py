#!/usr/bin/env python
"""Pre-download all protein-coding gene sequences and protein sequences.

Saves sequences to disk so embedding scripts can load them
instantly instead of querying APIs for hours.

Output files (in DATA_DIR):
    sequences/
        GRCh38_ensembl109_protein_coding_dna.npz
        GRCh38_ensembl109_protein_coding_proteins.npz
        GRCh38_ensembl109_protein_coding_exons.npz

Each .npz contains:
    gene_ids: array of Ensembl gene IDs
    sequences: array of sequence strings
    organism, biotype, region: metadata

Usage:
    python download_sequences.py --data-dir /path/to/data
    python download_sequences.py --data-dir /path/to/data --region exons
    python download_sequences.py --data-dir /path/to/data --region introns
    python download_sequences.py --data-dir /path/to/data --type protein
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def download_dna_sequences(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
    region: str = "full",
) -> None:
    """Download all DNA sequences and save to disk."""
    from embpy.resources.gene_resolver import GeneResolver

    resolver = GeneResolver()

    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    release = resolver.release_version
    filename = f"GRCh38_ensembl{release}_{biotype}_{region}_dna.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    logger.info("Downloading %s %s DNA sequences (region=%s)...", biotype, organism, region)

    if region == "full":
        sequences = resolver.get_gene_sequences(biotype=biotype)
        if not sequences:
            logger.error("No sequences retrieved")
            sys.exit(1)

        gene_ids = list(sequences.keys())
        seq_list = list(sequences.values())
    else:
        if resolver.ensembl is None:
            logger.error("pyensembl not available")
            sys.exit(1)

        all_genes = resolver.ensembl.genes()
        if biotype != "all":
            all_genes = [g for g in all_genes if g.biotype == biotype]

        gene_ids = []
        seq_list = []
        total = len(all_genes)

        for i, gene in enumerate(all_genes):
            if (i + 1) % 100 == 0:
                logger.info("Fetching %s %d/%d...", region, i + 1, total)

            seq = resolver.get_gene_region_sequence(
                gene.gene_id, id_type="ensembl_id",
                organism=organism, region=region,
            )
            if seq:
                gene_ids.append(gene.gene_id)
                seq_list.append(seq)

            time.sleep(0.35)

    np.savez(
        output_path,
        gene_ids=np.array(gene_ids, dtype=str),
        sequences=np.array(seq_list, dtype=str),
        organism=organism,
        biotype=biotype,
        region=region,
        ensembl_release=release,
    )
    logger.info(
        "Saved %d %s sequences to %s",
        len(gene_ids), region, output_path,
    )


def download_protein_sequences(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
) -> None:
    """Download all canonical protein sequences and save to disk."""
    from embpy.resources.protein_resolver import ProteinResolver

    pr = ProteinResolver(organism=organism)

    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    filename = f"uniprot_{biotype}_canonical_proteins.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    logger.info("Downloading canonical protein sequences for %s %s genes...", biotype, organism)

    gene_ids = pr._get_all_gene_ids(biotype=biotype)
    if not gene_ids:
        logger.error("No genes found")
        sys.exit(1)

    logger.info("Found %d genes, fetching protein sequences...", len(gene_ids))

    result_gene_ids = []
    result_accessions = []
    result_sequences = []
    total = len(gene_ids)

    for i, gene_id in enumerate(gene_ids):
        if (i + 1) % 100 == 0:
            logger.info("Fetching protein %d/%d...", i + 1, total)

        accession = pr.resolve_uniprot_id(gene_id, id_type="ensembl_id")
        if accession:
            seq = pr.get_canonical_sequence(gene_id, id_type="ensembl_id")
            if seq:
                result_gene_ids.append(gene_id)
                result_accessions.append(accession)
                result_sequences.append(seq)

        time.sleep(0.15)

    np.savez(
        output_path,
        gene_ids=np.array(result_gene_ids, dtype=str),
        accessions=np.array(result_accessions, dtype=str),
        sequences=np.array(result_sequences, dtype=str),
        organism=organism,
        biotype=biotype,
    )
    logger.info(
        "Saved %d protein sequences to %s",
        len(result_gene_ids), output_path,
    )


def download_protein_isoforms(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
) -> None:
    """Download all protein isoforms and save to disk."""
    from embpy.resources.protein_resolver import ProteinResolver

    pr = ProteinResolver(organism=organism)

    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    filename = f"uniprot_{biotype}_all_isoforms.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    logger.info("Downloading all protein isoforms for %s %s genes...", biotype, organism)

    gene_ids = pr._get_all_gene_ids(biotype=biotype)
    if not gene_ids:
        logger.error("No genes found")
        sys.exit(1)

    logger.info("Found %d genes, fetching isoform sequences...", len(gene_ids))

    all_ids = []
    all_gene_ids = []
    all_isoform_ids = []
    all_sequences = []
    total = len(gene_ids)

    for i, gene_id in enumerate(gene_ids):
        if (i + 1) % 100 == 0:
            logger.info("Fetching isoforms %d/%d (%d isoforms so far)...",
                        i + 1, total, len(all_sequences))

        isoforms = pr.get_isoforms(gene_id, id_type="ensembl_id", include_canonical=True)
        for iso_id, seq in isoforms.items():
            all_ids.append(f"{gene_id}|{iso_id}")
            all_gene_ids.append(gene_id)
            all_isoform_ids.append(iso_id)
            all_sequences.append(seq)

        time.sleep(0.2)

    np.savez(
        output_path,
        ids=np.array(all_ids, dtype=str),
        gene_ids=np.array(all_gene_ids, dtype=str),
        isoform_ids=np.array(all_isoform_ids, dtype=str),
        sequences=np.array(all_sequences, dtype=str),
        organism=organism,
        biotype=biotype,
    )
    logger.info(
        "Saved %d isoform sequences (%d genes) to %s",
        len(all_sequences), len(set(all_gene_ids)), output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download gene/protein sequences for embedding.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Base data directory (sequences saved to data_dir/sequences/).",
    )
    parser.add_argument(
        "--type", type=str, default="dna",
        choices=["dna", "protein", "protein_isoforms", "all"],
        help="What to download: dna, protein (canonical), protein_isoforms, or all.",
    )
    parser.add_argument(
        "--region", type=str, default="full",
        choices=["full", "exons", "introns"],
        help="DNA region (ignored for protein).",
    )
    parser.add_argument("--organism", type=str, default="human")
    parser.add_argument("--biotype", type=str, default="protein_coding")

    args = parser.parse_args()

    if args.type in ("dna", "all"):
        download_dna_sequences(
            args.data_dir, args.organism, args.biotype, args.region,
        )

    if args.type in ("protein", "all"):
        download_protein_sequences(
            args.data_dir, args.organism, args.biotype,
        )

    if args.type in ("protein_isoforms", "all"):
        download_protein_isoforms(
            args.data_dir, args.organism, args.biotype,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
