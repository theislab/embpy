#!/usr/bin/env python3

import logging
import sys

# --- [1. Logging Setup] ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("embedding_job.log", mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()

# --- [2. Standard Library Imports] ---
import argparse
import csv  # <-- Import csv
import time
from pathlib import Path

# --- [3. Third-Party Imports] ---
try:
    import numpy as np
    import pyensembl
    import torch
except ImportError as e:
    log.critical(f"Failed to import a core dependency: {e}")
    sys.exit(1)

# --- [4. 'embpy' Package Imports] ---
try:
    from embpy.models.protein_models import ESMCWrapper
    from embpy.resources.gene_resolver import GeneResolver
except ImportError as e:
    log.critical(f"Failed to import from 'embpy' package: {e}")
    sys.exit(1)

# --- [Configuration] ---
GENOME_SPECIES = "human"
GENOME_RELEASE = 109


# --- [Helper Function] ---
def load_model(model_name: str, device: torch.device) -> ESMCWrapper:
    log.info(f"Initializing ESMCWrapper ({model_name})...")
    try:
        embedder = ESMCWrapper(model_path_or_name=model_name)
        embedder.load(device)
        log.info(f"'{model_name}' loaded successfully onto {device}.")
        return embedder
    except Exception as e:
        log.error(f"Failed to load ESMC model '{model_name}': {e}", exc_info=True)
        raise e


def load_processed_set(output_file: Path) -> set[tuple[str, str, str]]:
    """Reads an existing CSV to find completed (gene_id, model, pooling) tuples."""
    processed_set = set()
    if not output_file.is_file():
        return processed_set

    log.info(f"Loading previously processed genes from {output_file}...")
    try:
        with open(output_file, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            # Find column indices (robust to reordering)
            col_map = {name: idx for idx, name in enumerate(header)}
            if not all(k in col_map for k in ["gene_id", "model", "pooling"]):
                log.warning("Output file exists but header is malformed. Starting fresh.")
                return set()

            gene_id_idx = col_map["gene_id"]
            model_idx = col_map["model"]
            pooling_idx = col_map["pooling"]

            for row in reader:
                if len(row) > max(gene_id_idx, model_idx, pooling_idx):
                    processed_set.add((row[gene_id_idx], row[model_idx], row[pooling_idx]))

    except StopIteration:
        pass  # File is empty
    except Exception as e:
        log.error(f"Error reading existing CSV: {e}. Re-running all items.")
        return set()  # On error, just re-run (safe but slower)

    log.info(f"Found {len(processed_set)} existing entries.")
    return processed_set


# --- [Main Execution] ---
def main():
    # --- [5. Setup Argument Parser] ---
    parser = argparse.ArgumentParser(description="Embed all human genes using specified ESMC models and pooling.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["esmc_300m", "esmc_600m"],
        default=["esmc_300m", "esmc_600m"],
        help="Which ESMC model(s) to use for embedding.",
    )
    parser.add_argument(
        "--pooling", type=str, choices=["mean", "max", "cls"], default="mean", help="The pooling strategy to apply."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="esmc_gene_embeddings.csv",  # <-- Changed default name
        help="Path to the output CSV file (will be created or appended).",
    )
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("Starting embedding job with parameters:")
    log.info(f"  Models:  {args.models}")
    log.info(f"  Pooling: {args.pooling}")
    log.info(f"  Output:  {args.output}")
    log.info("=" * 50)

    # 1. Pick Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU.")
    log.info(f"Using device: {device}")

    # 2. Load Models
    models = {name: load_model(name, device) for name in args.models}
    log.info(f"Successfully loaded {len(models)} models.")

    # 3. Initialize Resolver
    log.info("Initializing GeneResolver...")
    resolver = GeneResolver()
    log.info("GeneResolver initialized.")

    # 4. Get Gene List from PyEnsembl
    log.info(f"Initializing PyEnsembl (Species: {GENOME_SPECIES}, Release: {GENOME_RELEASE})...")
    try:
        genome = pyensembl.Genome(species=GENOME_SPECIES, release=GENOME_RELEASE)
        genome.download_self_if_needed()
        genome.index_if_needed()
        genes = genome.genes()
        log.info(f"Found {len(genes)} total genes in {GENOME_SPECIES} release {GENOME_RELEASE}.")
    except Exception as e:
        log.error(f"Failed to initialize PyEnsembl: {e}", exc_info=True)
        sys.exit(1)

    # 5. Process Genes and Save Embeddings
    log.info(f"Starting embedding process. Output will be saved to '{args.output}'.")
    start_job_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # --- CSV-specific setup ---
    output_file = Path(args.output)
    # Load the set of genes we've already processed from the CSV
    processed_set = load_processed_set(output_file)
    file_exists = output_file.is_file() and output_file.stat().st_size > 0

    # Open CSV file in 'append' mode
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        header = ["gene_id", "gene_name", "model", "pooling", "embedding"]

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(header)

        for i, gene in enumerate(genes):
            gene_id = gene.gene_id
            gene_name = gene.gene_name
            log_prefix = f"[{i + 1}/{len(genes)} | {gene_id}]"

            try:
                # Check if we need to run any models for this gene
                models_to_run = []
                for model_name in args.models:
                    lookup_key = (gene_id, model_name, args.pooling)
                    if lookup_key in processed_set:
                        log.info(f"{log_prefix} Already has '{model_name}_{args.pooling}'. Skipping model.")
                    else:
                        models_to_run.append(model_name)

                if not models_to_run:
                    log.info(f"{log_prefix} All requested embeddings already exist. Skipping gene.")
                    skipped_count += 1
                    continue

                # Get Sequence (only if needed)
                log.info(f"{log_prefix} Processing '{gene_name}'...")
                sequence = resolver.get_protein_sequence(gene_name, id_type="symbol")

                if not sequence:
                    log.warning(f"{log_prefix} ✗ No sequence found for '{gene_name}'.")
                    skipped_count += 1
                    continue

                log.info(f"{log_prefix} ✓ Found sequence (Length: {len(sequence)} aa)")

                # Embed with requested models
                for model_name in models_to_run:
                    embedder = models[model_name]

                    embed_start_time = time.time()
                    embedding_result = embedder.embed(sequence, pooling_strategy=args.pooling)
                    final_embedding = embedding_result["embedding"]
                    embed_time = time.time() - embed_start_time

                    # Convert embedding (numpy array) to a space-separated string
                    embedding_str = " ".join(map(str, final_embedding))

                    # Write the new row to the CSV
                    row = [gene_id, gene_name, model_name, args.pooling, embedding_str]
                    writer.writerow(row)
                    f.flush()  # Ensure it's written to disk immediately

                    # Add to processed set so we don't re-run if script restarts
                    processed_set.add((gene_id, model_name, args.pooling))

                    log.info(f"{log_prefix} ✓ Embedded with {model_name}_{args.pooling} in {embed_time:.2f}s")

                processed_count += 1  # A gene is "processed" if it gets at least one new embedding

            except Exception as e:
                log.error(f"{log_prefix} ✗ FAILED for '{gene_name}': {e}", exc_info=True)
                error_count += 1

    # 6. Final Report
    end_job_time = time.time()
    total_time = (end_job_time - start_job_time) / 3600.0
    log.info("=" * 40)
    log.info("Embedding Job Complete")
    log.info(f"Total Time: {total_time:.2f} hours")
    log.info(f"Successfully Processed (new entries): {processed_count}")
    log.info(f"Skipped (No seq or already done): {skipped_count}")
    log.info(f"Errored:                            {error_count}")
    log.info(f"Results saved to '{args.output}'")
    log.info("=" * 40)


if __name__ == "__main__":
    main()
