#!/usr/bin/env python
"""Extract exon or intron sequences and optionally embed them.

Supports multiple input modes:

1. **single** — Extract regions for a single gene (symbol or Ensembl ID).
2. **batch**  — Extract regions for a list of genes from a text file (one per line).
3. **adata**  — Extract regions for genes listed in an AnnData ``.h5ad`` file.

Examples
--------
.. code-block:: bash

    # Extract exon sequences for TP53 and save as FASTA
    python extract_gene_regions.py --gene TP53 --region exons --output tp53_exons.fa

    # Extract introns and embed them with Enformer
    python extract_gene_regions.py --gene TP53 --region introns \\
        --model enformer_human_rough --output tp53_introns.npz

    # Batch mode from a gene list file
    python extract_gene_regions.py --gene-list genes.txt --region exons \\
        --model enformer_human_rough --output exon_embeddings.npz

    # From AnnData
    python extract_gene_regions.py --adata my_data.h5ad --gene-column gene_symbol \\
        --region exons --output exon_seqs.fa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from embpy.resources.gene_resolver import GeneResolver


def _write_fasta(
    regions: dict[str, list[dict]],
    output_path: Path,
    region_type: str,
) -> None:
    """Write extracted regions to a FASTA file."""
    with open(output_path, "w") as f:
        for gene, entries in regions.items():
            for entry in entries:
                header = (
                    f">{gene}|{entry['id']}|{region_type}|"
                    f"{entry['start']}-{entry['end']}|strand={entry['strand']}"
                )
                f.write(f"{header}\n{entry['sequence']}\n")
    logging.info(f"Wrote {sum(len(v) for v in regions.values())} sequences to {output_path}")


def _embed_regions(
    regions: dict[str, list[dict]],
    model_name: str,
    pooling_strategy: str,
    output_path: Path,
) -> None:
    """Embed concatenated region sequences and save as .npz."""
    from embpy.embedder import BioEmbedder

    embedder = BioEmbedder(device="auto")

    gene_names: list[str] = []
    embeddings: list[np.ndarray] = []

    for gene, entries in regions.items():
        concat_seq = "".join(str(e["sequence"]) for e in entries)
        if not concat_seq:
            logging.warning(f"No sequence for {gene}; skipping.")
            continue
        try:
            emb = embedder.embed_gene(
                identifier=concat_seq,
                model=model_name,
                id_type="sequence",
                pooling_strategy=pooling_strategy,
            )
            gene_names.append(gene)
            embeddings.append(emb)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Embedding failed for {gene}: {exc}")

    if embeddings:
        np.savez(
            output_path,
            embeddings=np.stack(embeddings),
            gene_names=np.array(gene_names),
        )
        logging.info(f"Saved {len(embeddings)} embeddings to {output_path}")
    else:
        logging.error("No embeddings produced.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract exon/intron sequences from genes and optionally embed them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--gene", type=str, help="Single gene symbol or Ensembl ID.")
    input_grp.add_argument("--gene-list", type=str, help="Text file with one gene per line.")
    input_grp.add_argument("--adata", type=str, help="AnnData .h5ad file.")

    parser.add_argument(
        "--id-type",
        choices=["symbol", "ensembl_id"],
        default="symbol",
        help="Type of gene identifier (default: symbol).",
    )
    parser.add_argument(
        "--region",
        choices=["exons", "introns"],
        required=True,
        help="Which gene region to extract.",
    )
    parser.add_argument("--organism", default="human", help="Organism (default: human).")
    parser.add_argument("--gene-column", type=str, help="Column in adata.var for gene IDs.")
    parser.add_argument("--transcript-id", type=str, help="Use a specific transcript ID.")

    parser.add_argument("--model", type=str, help="Embed with this model (e.g. enformer_human_rough).")
    parser.add_argument("--pooling", default="mean", help="Pooling strategy (default: mean).")
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path (.fa for FASTA, .npz for embeddings).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    resolver = GeneResolver()

    genes: list[str] = []
    if args.gene:
        genes = [args.gene]
    elif args.gene_list:
        genes = Path(args.gene_list).read_text().strip().splitlines()
        genes = [g.strip() for g in genes if g.strip()]
        logging.info(f"Loaded {len(genes)} genes from {args.gene_list}")
    elif args.adata:
        genes = resolver.load_genes_from_adata(args.adata, column=args.gene_column)
        logging.info(f"Loaded {len(genes)} genes from {args.adata}")

    if not genes:
        logging.error("No genes to process.")
        sys.exit(1)

    all_regions: dict[str, list[dict]] = {}
    for gene in genes:
        regions = resolver.get_gene_regions(
            gene,
            id_type=args.id_type,
            organism=args.organism,
            region=args.region,
            transcript_id=args.transcript_id,
        )
        if regions:
            all_regions[gene] = regions
            logging.info(f"  {gene}: {len(regions)} {args.region}")
        else:
            logging.warning(f"  {gene}: no {args.region} found")

    logging.info(f"Extracted {args.region} for {len(all_regions)}/{len(genes)} genes.")

    output = Path(args.output)
    if args.model:
        _embed_regions(all_regions, args.model, args.pooling, output)
    else:
        _write_fasta(all_regions, output, args.region)


if __name__ == "__main__":
    main()
