#!/usr/bin/env python
"""Production pipeline: bulk download genome/proteome from FTP and extract per-gene sequences.

Uses Ensembl FTP for genome FASTA + GTF and UniProt FTP for proteome FASTA,
then extracts per-gene sequences locally via pysam (indexed FASTA access)
and pyensembl (gene coordinates). No REST API calls needed.

Output structure (under DATA_DIR):
    genome/
        GRCh38_ensembl{release}_protein_coding_full_dna.npz
        GRCh38_ensembl{release}_protein_coding_exons_dna.npz
        GRCh38_ensembl{release}_protein_coding_introns_dna.npz
    proteome/
        canonical/
            uniprot_protein_coding_canonical_proteins.npz
        isoforms/
            uniprot_protein_coding_all_isoforms.npz

Usage:
    python download_sequences.py --data-dir /path/to/data/datasets --type dna --region full
    python download_sequences.py --data-dir /path/to/data/datasets --type dna --region exons
    python download_sequences.py --data-dir /path/to/data/datasets --type protein
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ENSEMBL_FTP = "https://ftp.ensembl.org/pub"

SPECIES_CONFIG = {
    "human": {
        "assembly": "GRCh38",
        "species_name": "homo_sapiens",
        "release": 109,
        "uniprot_proteome": "UP000005640_9606",
    },
    "mouse": {
        "assembly": "GRCm39",
        "species_name": "mus_musculus",
        "release": 109,
        "uniprot_proteome": "UP000000589_10090",
    },
}


def _download_file(url: str, dest: Path, min_size: int = 100) -> Path:
    """Download a file from a URL if it doesn't already exist and is valid."""
    if dest.exists() and dest.stat().st_size > min_size:
        logger.info("Already downloaded: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    if dest.exists():
        logger.warning("Removing invalid/truncated file: %s (%d bytes)", dest, dest.stat().st_size)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("Downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, str(tmp))
    except Exception:
        logger.info("urllib failed, trying wget...")
        subprocess.run(
            ["wget", "-q", "-O", str(tmp), url],
            check=True, timeout=7200,
        )

    if tmp.stat().st_size < min_size:
        logger.error("Downloaded file too small (%d bytes), likely failed", tmp.stat().st_size)
        tmp.unlink()
        sys.exit(1)

    os.rename(str(tmp), str(dest))
    size_mb = dest.stat().st_size / 1e6
    logger.info("Downloaded %.1f MB", size_mb)
    return dest


def _decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file atomically (write to .tmp, then rename)."""
    out_path = gz_path.with_suffix("")
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("Already decompressed: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path

    tmp_path = out_path.with_suffix(".decompressing")
    logger.info("Decompressing %s...", gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.rename(str(tmp_path), str(out_path))
    logger.info("Decompressed to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


# ==================================================================
# DNA: Bulk genome FASTA + pysam extraction
# ==================================================================

def _prepare_genome(cache_dir: Path, config: dict) -> Path:
    """Download, decompress, and index the genome FASTA -- concurrency-safe via flock."""
    import pysam

    assembly = config["assembly"]
    species = config["species_name"]
    release = config["release"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f"{assembly}.lock"
    genome_gz = cache_dir / f"{assembly}.dna.primary_assembly.fa.gz"
    genome_fa = cache_dir / f"{assembly}.dna.primary_assembly.fa"
    fai_path = Path(str(genome_fa) + ".fai")

    with open(lock_path, "w") as lock_fd:
        logger.info("Acquiring genome prep lock (%s)...", lock_path)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        logger.info("Lock acquired -- preparing genome...")

        genome_url = (
            f"{ENSEMBL_FTP}/release-{release}/fasta/{species}/dna/"
            f"{species[0].upper() + species[1:]}.{assembly}.dna.primary_assembly.fa.gz"
        )
        _download_file(genome_url, genome_gz)
        _decompress_gz(genome_gz)

        if not fai_path.exists():
            logger.info("Indexing genome FASTA with pysam...")
            pysam.faidx(str(genome_fa))
            logger.info("Genome index created: %s", fai_path)
        else:
            logger.info("Genome index already exists: %s", fai_path)

    logger.info("Genome preparation complete")
    return genome_fa


def download_dna_sequences(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
    region: str = "full",
) -> None:
    """Download genome FASTA from Ensembl FTP, then extract per-gene sequences."""
    config = SPECIES_CONFIG.get(organism)
    if not config:
        logger.error("Unsupported organism '%s'. Supported: %s", organism, list(SPECIES_CONFIG.keys()))
        sys.exit(1)

    import pysam
    import pyensembl

    release = config["release"]
    assembly = config["assembly"]

    seq_dir = data_dir / "genome"
    seq_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{assembly}_ensembl{release}_{biotype}_{region}_dna.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    cache_dir = data_dir / ".cache" / "ensembl"
    genome_fa = _prepare_genome(cache_dir, config)
    fasta = pysam.FastaFile(str(genome_fa))

    logger.info("Loading gene annotations via pyensembl (release %d, %s)...", release, organism)
    ens = pyensembl.EnsemblRelease(release=release, species=organism)
    ens.download()
    ens.index()

    all_genes = ens.genes()
    if biotype != "all":
        all_genes = [g for g in all_genes if g.biotype == biotype]
    total = len(all_genes)
    logger.info("Found %d %s genes. Extracting %s sequences...", total, biotype, region)

    available_chroms = set(fasta.references)

    gene_ids = []
    sequences = []
    skipped = 0

    for i, gene in enumerate(all_genes):
        if (i + 1) % 1000 == 0:
            logger.info("  Processed %d/%d genes (%d sequences extracted)...", i + 1, total, len(gene_ids))

        chrom = gene.contig
        if chrom not in available_chroms:
            alt = f"chr{chrom}" if not chrom.startswith("chr") else chrom[3:]
            if alt in available_chroms:
                chrom = alt
            else:
                skipped += 1
                continue

        try:
            if region == "full":
                seq = fasta.fetch(chrom, gene.start - 1, gene.end)
                if gene.strand == "-":
                    seq = _reverse_complement(seq)

            elif region == "exons":
                exon_seqs = []
                for exon in sorted(gene.exons, key=lambda e: e.start):
                    eseq = fasta.fetch(chrom, exon.start - 1, exon.end)
                    exon_seqs.append(eseq)
                if gene.strand == "-":
                    exon_seqs = [_reverse_complement(s) for s in reversed(exon_seqs)]
                seq = "".join(exon_seqs)

            elif region == "introns":
                exons = sorted(gene.exons, key=lambda e: e.start)
                intron_seqs = []
                for j in range(len(exons) - 1):
                    i_start = exons[j].end
                    i_end = exons[j + 1].start - 1
                    if i_end <= i_start:
                        continue
                    iseq = fasta.fetch(chrom, i_start, i_end)
                    intron_seqs.append(iseq)
                if gene.strand == "-":
                    intron_seqs = [_reverse_complement(s) for s in reversed(intron_seqs)]
                seq = "".join(intron_seqs)
            else:
                logger.error("Unknown region '%s'", region)
                sys.exit(1)

            if seq:
                gene_ids.append(gene.gene_id)
                sequences.append(seq)

        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to extract %s for %s: %s", region, gene.gene_id, e)
            skipped += 1

    fasta.close()

    logger.info(
        "Extracted %d sequences (%d skipped) for region=%s",
        len(gene_ids), skipped, region,
    )

    _save_chunked_npz(
        output_path, gene_ids, sequences,
        organism=organism, biotype=biotype, region=region,
        ensembl_release=str(release),
    )


def _reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def _save_chunked_npz(
    output_path: Path,
    gene_ids: list[str],
    sequences: list[str],
    **metadata: str,
) -> None:
    """Save sequences to .npz without holding everything as a numpy array at once."""
    logger.info("Saving %d sequences to %s...", len(gene_ids), output_path)
    np.savez_compressed(
        output_path,
        gene_ids=np.array(gene_ids, dtype=object),
        sequences=np.array(sequences, dtype=object),
        **{k: np.array(v) for k, v in metadata.items()},
    )
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved %s (%.1f MB)", output_path, size_mb)


# ==================================================================
# Protein: Bulk UniProt FASTA
# ==================================================================

def download_protein_sequences(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
) -> None:
    """Download canonical protein sequences from UniProt FTP."""
    config = SPECIES_CONFIG.get(organism)
    if not config:
        logger.error("Unsupported organism '%s'", organism)
        sys.exit(1)

    seq_dir = data_dir / "proteome" / "canonical"
    seq_dir.mkdir(parents=True, exist_ok=True)

    filename = f"uniprot_{biotype}_canonical_proteins.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    proteome_id = config["uniprot_proteome"]
    taxon = proteome_id.split("_")[1]

    cache_dir = data_dir / ".cache" / "uniprot"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fasta_url = (
        f"https://rest.uniprot.org/uniprotkb/stream?"
        f"format=fasta&query=organism_id:{taxon}+AND+reviewed:true"
    )
    fasta_file = cache_dir / f"{proteome_id}_sprot.fasta"

    if not fasta_file.exists():
        logger.info("Downloading Swiss-Prot proteome for taxon %s...", taxon)
        _download_file(fasta_url, fasta_file)

    from Bio import SeqIO

    logger.info("Parsing protein FASTA...")
    gene_ids = []
    accessions = []
    sequences_list = []

    for record in SeqIO.parse(str(fasta_file), "fasta"):
        acc = record.id.split("|")[1] if "|" in record.id else record.id
        seq = str(record.seq)

        gene_name = ""
        desc = record.description
        if "GN=" in desc:
            gn_part = desc.split("GN=")[1]
            gene_name = gn_part.split()[0]

        gene_ids.append(gene_name or acc)
        accessions.append(acc)
        sequences_list.append(seq)

    logger.info("Parsed %d canonical protein sequences", len(gene_ids))

    np.savez_compressed(
        output_path,
        gene_ids=np.array(gene_ids, dtype=object),
        accessions=np.array(accessions, dtype=object),
        sequences=np.array(sequences_list, dtype=object),
        organism=organism,
        biotype=biotype,
    )
    logger.info("Saved %s", output_path)


# ==================================================================
# Protein isoforms (keep existing REST API approach -- already works)
# ==================================================================

def download_protein_isoforms(
    data_dir: Path,
    organism: str = "human",
    biotype: str = "protein_coding",
) -> None:
    """Download all protein isoforms from UniProt REST API."""
    from embpy.resources.protein_resolver import ProteinResolver

    pr = ProteinResolver(organism=organism)

    seq_dir = data_dir / "proteome" / "isoforms"
    seq_dir.mkdir(parents=True, exist_ok=True)

    filename = f"uniprot_{biotype}_all_isoforms.npz"
    output_path = seq_dir / filename

    if output_path.exists():
        logger.info("Already exists: %s", output_path)
        return

    logger.info("Downloading all protein isoforms for %s %s genes...", biotype, organism)

    gene_ids_list = pr._get_all_gene_ids(biotype=biotype)
    if not gene_ids_list:
        logger.error("No genes found")
        sys.exit(1)

    logger.info("Found %d genes, fetching isoform sequences...", len(gene_ids_list))

    all_ids = []
    all_gene_ids = []
    all_isoform_ids = []
    all_sequences = []
    total = len(gene_ids_list)

    for i, gene_id in enumerate(gene_ids_list):
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

    np.savez_compressed(
        output_path,
        ids=np.array(all_ids, dtype=object),
        gene_ids=np.array(all_gene_ids, dtype=object),
        isoform_ids=np.array(all_isoform_ids, dtype=object),
        sequences=np.array(all_sequences, dtype=object),
        organism=organism,
        biotype=biotype,
    )
    logger.info(
        "Saved %d isoform sequences (%d genes) to %s",
        len(all_sequences), len(set(all_gene_ids)), output_path,
    )


# ==================================================================
# CLI
# ==================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production pipeline: bulk download and extract gene/protein sequences.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Base datasets directory (e.g. data/datasets).",
    )
    parser.add_argument(
        "--type", type=str, default="dna",
        choices=["dna", "protein", "protein_isoforms", "all"],
    )
    parser.add_argument(
        "--region", type=str, default="full",
        choices=["full", "exons", "introns"],
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


if __name__ == "__main__":
    main()
