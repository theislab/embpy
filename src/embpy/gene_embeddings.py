#!/usr/bin/env python3

"""
Processes gene embeddings using the Enformer or Borzoi model.

It loads gene coordinate data and a chromosome dictionary, then for each gene in a specified batch,
computes embeddings using the selected model. Supports full gene or TSS-only regions,
with adjustable bin sizes: 128 bp for Enformer, 32 bp for Borzoi.
"""

import logging
import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from borzoi_pytorch import Borzoi
from enformer_pytorch import from_pretrained as load_enformer
from tqdm import tqdm


def get_device() -> torch.device:
    """Selects the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class GeneEmbeddingProcessor:
    """
    Processes gene embeddings using Enformer or Borzoi.

    1) __init__ takes gene/transcript IDs and model_type ('enformer' or 'borzoi').
    2) load_data() picks the matching row from the Mart file and loads the chromosome sequence.
    3) load_model() loads the chosen model onto the device.
    4) embed_row() slices the appropriate region (full_gene or TSS-only), applies model-specific
       preprocessing (_preprocess_enformer or _preprocess_borzoi), runs the model, and pools outputs.
    """

    # number of bins upstream of TSS to include
    N_TSS_BINS = 20

    def __init__(
        self,
        mart_file: str,
        chromosome_folder: str,
        model_type: Literal["enformer", "borzoi"] = "enformer",
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ):
        self.mart_file = mart_file
        self.chrom_folder = chromosome_folder
        self.model_type = model_type
        self.gene_id = gene_id
        self.tx_id = transcript_id
        self.region = "full_gene" if gene_id else "TSS_only"
        self.device = get_device()

        # Column names in the Mart file
        self.chrom_col = "Chromosome/scaffold name"
        self.hgnc_col = "HGNC symbol"
        self.start_col = "Gene start (bp)"
        self.end_col = "Gene end (bp)"
        self.tss_col = "Transcription start site (TSS)"

        # To be set in load_data()
        self.row: pd.Series | None = None
        self.chrom_seq: str | None = None
        # To be set in load_model()
        self.model = None

        # Borzoi-specific parameters
        if self.model_type == "borzoi":
            self.ALPHABET_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
            self.NUM_CHANNELS = 4
            self.SEQUENCE_LENGTH = 524288

    def load_data(self) -> None:
        """Read the Mart table, select the row, and load the chromosome sequence."""
        df = pd.read_csv(self.mart_file)
        if self.region == "full_gene":
            mask = df["Gene stable ID"].eq(self.gene_id) | df[self.hgnc_col].eq(self.gene_id)
        else:
            mask = df["Transcript stable ID"].eq(self.tx_id) | df["Transcript stable ID version"].eq(self.tx_id)
        hits = df[mask]
        if hits.empty:
            whom = "gene" if self.region == "full_gene" else "transcript"
            raise KeyError(f"No {whom} entry for {self.gene_id or self.tx_id}")
        if len(hits) > 1:
            logging.warning("Multiple hits in Mart; using the first row.")
        self.row = hits.iloc[0]

        chrom = str(self.row[self.chrom_col])
        fasta_path = os.path.join(self.chrom_folder, f"chr{chrom}.fa")
        rec = SeqIO.read(fasta_path, "fasta")
        self.chrom_seq = str(rec.seq).upper()

    def load_model(self) -> None:
        """Instantiate and load the model (Enformer or Borzoi) onto the device."""
        logging.info(f"Loading {self.model_type} onto {self.device}")
        if self.model_type == "enformer":
            m = load_enformer("EleutherAI/enformer-official-rough", use_tf_gamma=False)
            self.model = m.to(self.device).eval()  # type: ignore
        else:
            self.model = Borzoi.from_pretrained("johahi/borzoi-replicate-0").to(self.device).eval()

    def _preprocess_enformer(self, sequence: str) -> torch.Tensor:
        """Convert a DNA string to integer tensor, pad/trim to 196608 bp for Enformer."""
        mp = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.array([mp.get(b, 4) for b in sequence], dtype=int)
        xt = torch.from_numpy(arr).unsqueeze(0)
        target_len = 196_608
        L = xt.shape[1]
        if L < target_len:
            tot = target_len - L
            left, right = tot // 2, tot - tot // 2
            xt = F.pad(xt, (left, right), value=4)
            logging.debug(f"Padded Enformer sequence {L}→{target_len} bp.")
        elif L > target_len:
            start = (L - target_len) // 2
            xt = xt[:, start : start + target_len]
            logging.warning(f"Truncated Enformer sequence {L}→{target_len} bp.")
        return xt.to(self.device)

    def _preprocess_borzoi(self, sequence: str) -> torch.Tensor:
        """Convert a DNA string to one-hot tensor, pad/trim to 524288 bp for Borzoi."""
        seq = sequence.upper()
        idx = torch.tensor([self.ALPHABET_MAP.get(b, 0) for b in seq], dtype=torch.long).unsqueeze(0)
        oh = F.one_hot(idx, num_classes=self.NUM_CHANNELS).permute(0, 2, 1).float()
        L_in = oh.shape[2]
        L_tar = self.SEQUENCE_LENGTH
        if L_in < L_tar:
            pad_total = L_tar - L_in
            left = pad_total // 2
            right = pad_total - left
            oh = F.pad(oh, (left, right), value=0.0)
            logging.debug(f"Padded Borzoi sequence {L_in}→{L_tar} bp.")
        elif L_in > L_tar:
            start = (L_in - L_tar) // 2
            oh = oh[:, :, start : start + L_tar]
            logging.warning(f"Truncated Borzoi sequence {L_in}→{L_tar} bp.")
        return oh.to(self.device)

    def _get_tss_window(self, row: pd.Series) -> str:
        """
        Extract the promoter window upstream of TSS using 128 bp bins for Enformer
        or 32 bp bins for Borzoi, over N_TSS_BINS bins.
        """
        tss = int(row[self.tss_col])
        bin_size = 128 if self.model_type == "enformer" else 32
        length = bin_size * self.N_TSS_BINS  # bytes upstream of TSS
        start = max(0, tss - length)
        return self.chrom_seq[start:tss]  # type: ignore

    def embed_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        """Embed one row: slice region, preprocess, forward, and pool."""
        # skip if chromosome mismatch
        if str(row[self.chrom_col]) != str(self.row[self.chrom_col]):
            return None

        # choose sequence slice
        if self.region == "full_gene":
            s, e = int(row[self.start_col]), int(row[self.end_col])
            seq = self.chrom_seq[s:e]  # type: ignore
        else:
            seq = self._get_tss_window(row)

        sym = row[self.hgnc_col]

        # forward
        if self.model_type == "enformer":
            xt = self._preprocess_enformer(seq)
            with torch.no_grad():
                _, emb = self.model(xt, return_embeddings=True)  # type: ignore
            emb = emb.cpu().squeeze(0).numpy()
        else:
            oh = self._preprocess_borzoi(seq)
            with torch.no_grad():
                embs = self.model.get_embs_after_crop(oh)
                print(embs)
            emb = embs.squeeze(0).cpu().numpy()

        # pool along bins axis
        if emb.ndim == 2:
            max_pool = emb.max(axis=1)
            mean_pool = emb.mean(axis=1)
            median_pool = np.median(emb, axis=1)
        else:
            max_pool = emb.max(axis=0)
            mean_pool = emb.mean(axis=0)
            median_pool = np.median(emb, axis=0)

        return max_pool, mean_pool, median_pool, sym

    def process_batch(
        self,
        identifiers: Sequence[str],
        id_type: Literal["gene", "transcript"],
    ) -> dict[str, np.ndarray]:
        """Embed a batch of IDs, returning stacked arrays and gene symbols."""
        max_list, mean_list, med_list, symbols = [], [], [], []
        for ident in tqdm(identifiers, desc="Batch embedding"):
            if id_type == "gene":
                self.gene_id, self.tx_id, self.region = ident, None, "full_gene"
            else:
                self.tx_id, self.gene_id, self.region = ident, None, "TSS_only"

            try:
                self.load_data()
                self.load_model()
                m1, m2, m3, sym = self.embed_row(self.row)  # type: ignore
            except Exception as e:
                logging.warning(f"Failed on {ident}: {e}")
                continue

            max_list.append(m1)
            mean_list.append(m2)
            med_list.append(m3)
            symbols.append(sym)

        return {
            "max": np.stack(max_list) if max_list else np.empty((0,)),
            "mean": np.stack(mean_list) if mean_list else np.empty((0,)),
            "median": np.stack(med_list) if med_list else np.empty((0,)),
            "genes": np.array(symbols, dtype=str),
        }


def save_embeddings(output_file: str, embeddings: dict[str, np.ndarray]) -> None:
    """Save embeddings dict to NPZ."""
    np.savez(output_file, allow_pickle=True, **embeddings)
    logging.info(f"Saved embeddings to {output_file}")
