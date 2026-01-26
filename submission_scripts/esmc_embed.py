import logging
import os
import sys

import pandas as pd

from embpy import BioEmbedder

# --- Configuration ---
MODEL_NAME = "esmc_300m"
OUTPUT_DIR = "/lustre/groups/ml01/workspace/goncalo.pinto/embpy/output/genes"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "esmc_human_proteome_max.csv")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def save_results(identifiers, embeddings, filename):
    """Saves results to CSV."""
    data = {}
    for gene, emb in zip(identifiers, embeddings, strict=False):
        if emb is not None:
            data[gene] = emb

    if data:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.to_csv(filename, index_label="gene_id")
        logging.info(f"Saved {len(df)} embeddings to {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize with API backend (required for auto-discovery)
    embedder = BioEmbedder(device="auto", resolver_backend="api")

    logging.info(f"Starting genome-wide embedding using {MODEL_NAME}...")

    # --- The Magic Call ---
    # We pass identifiers=None to tell it "Find them for me"
    # fetch_all_dna=True triggers the Ensembl gene list fetch
    # It returns embeddings for ALL protein coding genes found.

    # Note: For 20k genes, this might be memory heavy.
    # Ideally, we'd batch the list inside, but for simplicity here we get the full list.
    embeddings = embedder.embed_genes_batch(
        identifiers=None,  # <--- Auto-discovery mode
        model=MODEL_NAME,
        fetch_all_dna=True,  # <--- Triggers discovery
        biotype="protein_coding",  # <--- Filter
        pooling_strategy="max",
    )

    # The embedder internally resolved the list of genes.
    # We need to recover that list to save it.
    # Since the method currently returns a list, we re-fetch the keys to match them up.
    # (In a production version, embed_genes_batch should probably return a dict or DataFrame).

    # For now, we re-fetch the keys to map them (this is fast/cached)
    gene_map = embedder.gene_resolver.get_gene_sequences(biotype="protein_coding")
    gene_keys = list(gene_map.keys())

    # Save
    save_results(gene_keys, embeddings, OUTPUT_CSV)


if __name__ == "__main__":
    main()
