#!/usr/bin/env python3

"""
Processes gene embeddings using the Enformer model.

It loads gene coordinate data and a chromosome dictionary, then for each gene in a specified batch,
it computes the maximum, mean, and median of the output embeddings.
"""

import argparse
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from enformer_pytorch import from_pretrained
from tqdm import tqdm


def get_device() -> torch.device:
    """Selects the best available device: CUDA, MPS (for Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class GeneEmbeddingProcessor:
    """Processes gene embeddings using a pre-trained Enformer model.

    Attributes
    ----------
    gene_coordinate_file : str
        Path to the gene coordinate CSV file.
    chromosome_dict_file : str
        Path to the chromosome dictionary pickle file.
    region : str
        Region mode, either "full_gene" or "TSS_only".
    gene_coordinate_df : pd.DataFrame
        DataFrame with gene coordinates.
    chromosome_dict : dict
        Dictionary mapping chromosomes to sequence objects.
    model : torch.nn.Module
        Pre-trained Enformer model.
    device : torch.device
        Device for running computations.
    """

    def __init__(self, gene_coordinate_file: str, chromosome_dict_file: str, region: str):
        """
        Initialize the GeneEmbeddingProcessor.

        This constructor sets up the processor with the file paths for gene coordinates and the chromosome
        dictionary, as well as the region mode ("full_gene" or "TSS_only").

        Args:
            gene_coordinate_file (str): File path for gene coordinates.
            chromosome_dict_file (str): File path for the chromosome dictionary.
            region (str): Region mode, either "full_gene" or "TSS_only".
        """
        self.gene_coordinate_file = gene_coordinate_file
        self.chromosome_dict_file = chromosome_dict_file
        self.region = region

        self.gene_coordinate_df: pd.DataFrame
        self.chromosome_dict: dict[str, Any]
        self.model: torch.nn.Module
        self.device = get_device()

        # Define column names for consistency.
        self.chromosome_col_name = "Chromosome/scaffold name"
        self.HGNC_col_name = "HGNC symbol"
        self.gene_label_name = "label"
        if self.region == "full_gene":
            self.start_col_name = "Gene start (bp)"
            self.end_col_name = "Gene end (bp)"
        else:
            self.end_col_name = "Transcription start site (TSS)"

    def load_data(self) -> None:
        """Loads gene coordinate data and chromosome dictionary from files."""
        logging.info("Loading chromosome dictionary.")
        with open(self.chromosome_dict_file, "rb") as handle:
            self.chromosome_dict = pickle.load(handle)
        logging.info("Loading gene coordinate data.")
        self.gene_coordinate_df = pd.read_csv(self.gene_coordinate_file)

    def initialize_model(self) -> None:
        """Initializes the pre-trained Enformer model and moves it to the proper device."""
        logging.info(f"Initializing the Enformer model on device: {self.device}.")
        self.model = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False)
        # self.model = cast(torch.nn.Module, self.model.to(self.device))
        self.model.eval()

    def process_gene_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        """Processes a single gene row to compute its embeddings.

        This function:
          - Extracts the gene sequence based on the chosen region.
          - Converts nucleotide bases to numerical representations.
          - Pads the sequence if required.
          - Computes the embeddings with the Enformer model.
          - Computes max, mean, and median across embedding dimensions.

        Args:
            row (pd.Series): A row from the gene coordinate DataFrame.

        Returns
        -------
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
                A tuple of (max_embedding, mean_embedding, median_embedding, gene_symbol).
                Returns None if the gene is skipped (e.g. gene length is too short).
        """
        # Extract gene sequence based on region.
        if self.region == "full_gene":
            if row["Gene Length"] < 128:
                logging.warning(
                    f"Gene length of {row[self.HGNC_col_name]} is {row['Gene Length']}, it should be at least 128."
                )
                return None
            gene_seq = np.array(
                self.chromosome_dict[row[self.chromosome_col_name]][
                    row[self.start_col_name] : row[self.end_col_name]
                ].seq.upper()
            )
        else:
            gene_seq = np.array(
                self.chromosome_dict[row[self.chromosome_col_name]][
                    row[self.end_col_name] - 128 * 20 : row[self.end_col_name]
                ].seq.upper()
            )

        # Map nucleotide characters to numerical values.
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        gene_seq_numeric = np.array([mapping.get(base, 4) for base in gene_seq])

        # Create tensor and add a batch dimension.
        seq_tensor = torch.from_numpy(gene_seq_numeric.astype(int)).unsqueeze(0)

        # Pad the sequence if the gene label is 2 or 3, or if using TSS_only.
        num_chunks_gene = None
        num_of_chunks_left = None
        if row[self.gene_label_name] in [2, 3] or self.region == "TSS_only":
            # Compute the number of chunks needed (using ceil division).
            num_chunks_gene = -(-seq_tensor.shape[1] // 128)
            num_of_chunks_left = (1536 * 128 - seq_tensor.shape[1]) // (128 * 2)
            left_pad = 128 * num_of_chunks_left
            right_pad = 196608 - seq_tensor.shape[1] - left_pad
            seq_tensor = F.pad(seq_tensor, pad=(left_pad, right_pad, 0, 0), value=-1)

        if seq_tensor.shape[1] < 196608:
            raise ValueError(f"Gene length is {seq_tensor.shape[1]}, it should not be shorter than 196608")

        seq_tensor = seq_tensor.to(self.device)

        # Compute embeddings with the Enformer model.
        with torch.no_grad():
            _, embeddings = self.model(seq_tensor, return_embeddings=True)
        embeddings = embeddings.cpu().detach().numpy().squeeze(0)

        # If required, slice the embeddings for gene_label==3 or TSS_only.
        if row[self.gene_label_name] == 3 or self.region == "TSS_only":
            if num_chunks_gene is None or num_of_chunks_left is None:
                raise ValueError("Padding parameters were not computed correctly.")
            start_slice = num_of_chunks_left - 320 + 1
            end_slice = start_slice + num_chunks_gene
            embeddings = embeddings[start_slice:end_slice, :]

        # Compute summary statistics.
        max_values = np.max(embeddings, axis=0)
        mean_values = np.mean(embeddings, axis=0)
        median_values = np.median(embeddings, axis=0)
        gene_symbol = row[self.HGNC_col_name]

        # Cleanup
        del seq_tensor, embeddings
        # Clear CUDA cache only if CUDA is used.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return max_values, mean_values, median_values, gene_symbol

    def process_batch(self, batch_index: int, total_batches: int) -> dict[str, np.ndarray]:
        """Processes a batch of genes to compute embeddings.

        The batch is determined by slicing the gene coordinate DataFrame.

        Args:
            batch_index (int): Index of the current batch (0-indexed).
            total_batches (int): Total number of batches.

        Returns
        -------
            Dict[str, np.ndarray]: A dictionary containing keys 'max', 'mean', 'median', and 'genes'
            with corresponding numpy arrays of embeddings and gene symbols.
        """
        num_genes = len(self.gene_coordinate_df)
        batch_size = num_genes // total_batches
        start_idx = batch_index * batch_size
        end_idx = (start_idx + batch_size) if batch_index < total_batches - 1 else num_genes

        batch_df = self.gene_coordinate_df.iloc[start_idx:end_idx]
        logging.info(
            f"Processing batch {batch_index + 1}/{total_batches} with {len(batch_df)} genes out of {num_genes} total genes."
        )

        embeddings_max_list = []
        embeddings_mean_list = []
        embeddings_median_list = []
        genes_list = []

        for _, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0]):
            result = self.process_gene_row(row)
            if result is None:
                continue
            max_emb, mean_emb, median_emb, gene_symbol = result
            embeddings_max_list.append(max_emb)
            embeddings_mean_list.append(mean_emb)
            embeddings_median_list.append(median_emb)
            genes_list.append(gene_symbol)

        return {
            "max": np.array(embeddings_max_list),
            "mean": np.array(embeddings_mean_list),
            "median": np.array(embeddings_median_list),
            "genes": np.array(genes_list),
        }


def save_embeddings(output_file: str, embeddings: dict[str, np.ndarray]) -> None:
    """Saves the computed embeddings and gene symbols to an NPZ file.

    Args:
        output_file (str): Path to the output NPZ file.
        embeddings (dict[str, np.ndarray]): Dictionary containing embeddings and gene symbols.
    """
    np.savez(output_file, allow_pickle=True, **embeddings)
    logging.info(f"Saved embeddings to {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns
    -------
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process gene embeddings with the Enformer model.")
    parser.add_argument("--batch_index", type=int, required=True, help="Batch index (0-indexed).")
    parser.add_argument("--total_batches", type=int, required=True, help="Total number of batches.")
    parser.add_argument("--region", type=str, required=True, choices=["full_gene", "TSS_only"], help="Region type.")
    return parser.parse_args()


def main() -> None:
    """Main function that orchestrates gene embedding computation."""
    args = parse_arguments()

    if not (0 <= args.batch_index < args.total_batches):
        raise ValueError(f"batch_index must be between 0 and {args.total_batches - 1}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    gene_coordinate_file = "/ictstr01/home/icb/yuge.ji/projects/super_rad_project/gene_coordinate.csv"
    chromosome_dict_file = "/ictstr01/home/icb/yuge.ji/projects/super_rad_project/chromosome_dict.pickle"

    processor = GeneEmbeddingProcessor(gene_coordinate_file, chromosome_dict_file, args.region)
    processor.load_data()
    processor.initialize_model()

    embeddings = processor.process_batch(args.batch_index, args.total_batches)

    output_file = f"enformer_stuff/embeddings_genes_{args.region}_{args.batch_index}.npz"
    save_embeddings(output_file, embeddings)


if __name__ == "__main__":
    main()
