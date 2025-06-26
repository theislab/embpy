#!/usr/bin/env python3

"""
Processes gene embeddings using the Enformer model.

It loads gene coordinate data and a chromosome dictionary, then for each gene in a specified batch,
it computes the maximum, mean, and median of the output embeddings.
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
    """

    # TODO in the future add other models for DNA like borzoi etc...

    Processes gene embeddings using the Enformer model.

    1) __init__ takes either gene_id (→ full_gene) or transcript_id (→ TSS_only).
    2) load_data() picks the matching row from the Mart file and loads the chromosome.
    3) load_model() pulls Enformer onto GPU/CPU.
    4) embed_row(row) does your slice→pad→embed→pool logic exactly as before.
    """

    def __init__(
        self,
        mart_file: str,
        chromosome_folder: str,
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ):
        # if gene_id is not None and transcript_id is not None:
        #    raise ValueError("Provide *at most* one of gene_id or transcript_id.")
        self.mart_file = mart_file
        self.chrom_folder = chromosome_folder
        self.gene_id = gene_id
        self.tx_id = transcript_id

        # full‐gene vs TSS_only
        self.region = "full_gene" if gene_id else "TSS_only"
        self.device = get_device()

        # column headers in Mart file:
        self.chrom_col = "Chromosome/scaffold name"
        self.hgnc_col = "HGNC symbol"
        self.start_col = "Gene start (bp)"
        self.end_col = "Gene end (bp)"
        self.tss_col = "Transcription start site (TSS)"

        # to be filled by load_data()
        self.row = None  # the single pd.Series
        self.chrom_seq = None  # plain uppercase str

        # to be filled by load_model()
        self.model = None

    def load_data(self) -> None:
        """Read the Mart table, pick one row, and load that chromosome's FASTA."""
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

        # load just that chromosome
        chrom = str(self.row[self.chrom_col])
        fasta_path = os.path.join(self.chrom_folder, f"chr{chrom}.fa")
        rec = SeqIO.read(fasta_path, "fasta")
        self.chrom_seq = str(rec.seq).upper()

    def load_model(self) -> None:
        """Instantiates Enformer and moves it to self.device."""
        logging.info(f"Loading Enformer onto {self.device}")
        m = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False)
        self.model = m.to(self.device).eval()  # type: ignore

    def embed_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        """
        Compute embeddings of a row in gene_coordinates_file

        Takes a row from the Mart file, extracts the chromosome sequence,
        and computes the embeddings using Enformer.
        slices self.chrom_seq, pads/truncates to 196608, runs Enformer, then pools.
        """
        # only our chromosome:
        if str(row[self.chrom_col]) != str(self.row[self.chrom_col]):  # type: ignore
            return None

        # 1) slice
        if self.region == "full_gene":
            s, e = int(row[self.start_col]), int(row[self.end_col])
            length = e - s
            if length < 128:
                logging.warning(f"{row[self.hgnc_col]} only {length} bp, skipping")
                return None
            seq = self.chrom_seq[s:e]  # type: ignore
        else:
            tss = int(row[self.tss_col])
            seq = self.chrom_seq[tss - 128 * 20 : tss]  # type: ignore

        mp = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.array([mp.get(b, 4) for b in seq], dtype=int)

        # 3) tensor + batch
        xt = torch.from_numpy(arr).unsqueeze(0)

        # 4) pad or truncate to 196608 equally with 4 (“N”)
        target_len = 196_608
        L = xt.shape[1]
        if L < target_len:
            tot = target_len - L
            left, right = tot // 2, tot - tot // 2
            xt = F.pad(xt, (left, right), value=4)
            logging.debug(f"Padded sequence from {L} to {target_len}.")
        elif L > target_len:
            # Truncate from center
            trim_total = L - target_len
            trim_left = trim_total // 2
            trim_end = trim_left + target_len
            xt = xt[:, trim_left:trim_end]
            logging.warning(f"Truncated sequence from {L} to {target_len} from center.")
        # else L == target_len → do nothing

        assert xt.shape[1] == target_len

        # 5) forward
        xt = xt.to(self.device)
        with torch.no_grad():
            _, emb = self.model(xt, return_embeddings=True)  # type: ignore
        emb = emb.cpu().squeeze(0).numpy()

        # 6) slice back for TSS_only
        if self.region == "TSS_only":
            nc = -(-L // 128)
            nl = (1536 * 128 - L) // (128 * 2)
            start = nl - 320 + 1
            emb = emb[start : start + nc]

        # 7) pool & return
        return (
            emb.max(axis=0),
            emb.mean(axis=0),
            np.median(emb, axis=0),
            row[self.hgnc_col],
        )

    def process_batch(
        self,
        identifiers: Sequence[str],
        id_type: Literal["gene", "transcript"],
    ) -> dict[str, np.ndarray]:
        """
        Loop over a list of gene‐ or transcript‐IDs, embed each one, collect stats.

        Parameters
        ----------
        identifiers : Sequence[str]
            A list of either Ensembl gene IDs / HGNC symbols (if id_type=='gene')
            or Ensembl transcript IDs / versions (if id_type=='transcript').
        id_type : {'gene','transcript'}
            Which kind of IDs are in `identifiers`.

        Returns
        -------
        Dict[str, np.ndarray]
            'max','mean','median' shape=(N, D) arrays and 'genes' list of symbols.
        """
        max_list, mean_list, med_list, symbol_list = [], [], [], []

        for ident in tqdm(identifiers, desc="Batch embedding"):
            # re-configure for each identifier
            if id_type == "gene":
                self.gene_id = ident
                self.tx_id = None
                self.region = "full_gene"
            else:
                self.tx_id = ident
                self.gene_id = None
                self.region = "TSS_only"

            try:
                self.load_data()  # picks the right row & loads that chromosome
                self.load_model()  # loads Enformer onto self.device
                m1, m2, m3, sym = self.embed_row(self.row)  # type: ignore
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Failed on {ident}: {e}")
                continue

            max_list.append(m1)
            mean_list.append(m2)
            med_list.append(m3)
            symbol_list.append(sym)

        return {
            "max": np.stack(max_list, axis=0) if max_list else np.empty((0,)),
            "mean": np.stack(mean_list, axis=0) if mean_list else np.empty((0,)),
            "median": np.stack(med_list, axis=0) if med_list else np.empty((0,)),
            "genes": np.array(symbol_list, dtype=str),
        }


def save_embeddings(output_file: str, embeddings: dict[str, np.ndarray]) -> None:
    """Saves the computed embeddings and gene symbols to an NPZ file.

    Args:
        output_file (str): Path to the output NPZ file.
        embeddings (dict[str, np.ndarray]): Dictionary containing embeddings and gene symbols.
    """
    np.savez(output_file, allow_pickle=True, **embeddings)
    logging.info(f"Saved embeddings to {output_file}")
