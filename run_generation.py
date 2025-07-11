#!/usr/bin/env python3
import argparse
import time

import pandas as pd

from embpy.gene_embeddings import GeneEmbeddingProcessor


def main(ccle_csv, mart_file, chrom_dir, out_prefix):
    # 1) Read CCLE table
    ccle = pd.read_csv(ccle_csv, index_col=0)
    genes = list(ccle.index)
    print(f"Found {len(genes)} genes in CCLE file.")

    # 2) Instantiate the processor
    proc = GeneEmbeddingProcessor(
        mart_file=mart_file,
        chromosome_folder=chrom_dir,
    )

    # 3) Compute embeddings
    t0 = time.time()
    out = proc.process_batch(genes, id_type="gene")
    t1 = time.time()
    print(f"→ Finished embedding {len(out['genes'])} genes in {t1 - t0:.1f}s")

    # 4) Save to disk
    #    we'll save the three matrices as CSV (genes × dimensions)
    max_df = pd.DataFrame(out["max"], index=out["genes"])
    mean_df = pd.DataFrame(out["mean"], index=out["genes"])
    median_df = pd.DataFrame(out["median"], index=out["genes"])

    max_df.to_csv(f"{out_prefix}_max.csv")
    mean_df.to_csv(f"{out_prefix}_mean.csv")
    median_df.to_csv(f"{out_prefix}_median.csv")

    print("Done writing:")
    print(f"  {out_prefix}_max.csv")
    print(f"  {out_prefix}_mean.csv")
    print(f"  {out_prefix}_median.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ccle", required=True, help="Path to CCLE CSV")
    p.add_argument("--mart", required=True, help="Path to Mart file")
    p.add_argument("--chrom_dir", required=True, help="Path to chromosome folder")
    p.add_argument("--out_prefix", required=True, help="Prefix for output files")
    args = p.parse_args()
    main(args.ccle, args.mart, args.chrom_dir, args.out_prefix)
