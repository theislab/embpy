from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

try:
    from alphagenome.data import genome
    from alphagenome.models import dna_client, dna_output, variant_scorers
except ImportError:
    genome = None  # type: ignore
    dna_client = None  # type: ignore
    dna_output = None  # type: ignore
    variant_scorers = None  # type: ignore

ORGANISMS = {"human": "HOMO_SAPIENS", "mouse": "MUS_MUSCULUS"}


class AlphaGenomeWrapper(BaseModelWrapper):
    """Wrapper for Google DeepMind's AlphaGenome sequence model (cloud API).

    This class handles:
      1. Resolving the API key from ``ALPHAGENOME_API_KEY`` and creating a
         ``dna_client.DnaClient`` (no local weights to load).
      2. Predicting per-track profiles for a raw DNA sequence and pooling
         them into a fixed-length embedding (analogue of the local DNA
         model wrappers' ``embed``).
      3. Scoring a genomic variant's predicted effect on gene expression
         (log2 fold-change between REF and ALT alleles), following the same
         VCF-style chrom/pos/ref/alt convention and log2FC statistic as
         ``embpy.tl.genomics.snp_utils.profile_variant_effect_score`` /
         ``BorzoiWrapper``-based variant scoring elsewhere in this codebase.

    Parameters
    ----------
    model_path_or_name : str, optional
        Label only; AlphaGenome exposes a single hosted model per API
        version (kept for interface consistency with ``BaseModelWrapper``).
    model_version : str, optional
        Name of an ``alphagenome.models.dna_client.ModelVersion`` member
        (e.g. ``"FOLD_0"``). ``None`` uses the server default (all folds).
    organism : str, optional
        ``"human"`` or ``"mouse"``. Default ``"human"``.
    context_window : int, optional
        Sequence length (bp) used to build a default interval centered on
        a variant position when explicit interval bounds are not supplied
        to :meth:`predict_variant_effect`. Defaults to 1,048,576 (1 Mb,
        AlphaGenome's maximum context).
    **kwargs : Any
        Forwarded to ``BaseModelWrapper.__init__``.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "none"]

    DEFAULT_CONTEXT_WINDOW = 1_048_576

    def __init__(
        self,
        model_path_or_name: str = "alphagenome",
        model_version: str | None = None,
        organism: str = "human",
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        if organism not in ORGANISMS:
            raise ValueError(f"Unknown organism '{organism}'. Choose from {list(ORGANISMS)}.")
        self.organism = organism
        self.model_version = model_version
        self.context_window = context_window
        self.client: Any = None

    @staticmethod
    def _resolve_api_key() -> str:
        key = os.environ.get("ALPHAGENOME_API_KEY")
        if not key:
            raise ValueError(
                "ALPHAGENOME_API_KEY environment variable is not set. Source the "
                "AlphaGenome API key file into the environment before calling "
                "AlphaGenomeWrapper.load()."
            )
        return key

    def load(self, device: torch.device | None = None) -> None:
        """Create the AlphaGenome API client. There are no local weights to load.

        Parameters
        ----------
        device : torch.device, optional
            Accepted for interface consistency with ``BaseModelWrapper``;
            AlphaGenome inference runs on Google's servers, not locally.

        Raises
        ------
        ImportError
            If the ``alphagenome`` package is not installed.
        ValueError
            If ``ALPHAGENOME_API_KEY`` is not set in the environment.
        """
        if dna_client is None:
            raise ImportError("alphagenome not installed; cannot load AlphaGenomeWrapper. Run `pip install alphagenome`.")
        if self.client is not None:
            logging.warning("AlphaGenome client already initialized.")
            return

        api_key = self._resolve_api_key()
        model_version = getattr(dna_client.ModelVersion, self.model_version) if self.model_version else None
        self.client = dna_client.create(api_key, model_version=model_version)
        self.model = self.client
        self.device = device
        logging.info(f"AlphaGenome API client ready (organism={self.organism}).")

    def _organism_enum(self) -> Any:
        return getattr(dna_client.Organism, ORGANISMS[self.organism])

    def _default_interval(self, chrom: str, pos: int) -> Any:
        half = self.context_window // 2
        return genome.Interval(chromosome=chrom, start=max(pos - half, 0), end=pos + half)

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        output_type: str = "RNA_SEQ",
        ontology_terms: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Predict a track for a DNA sequence and pool it into an embedding.

        Runs ``predict_sequence`` for a single output modality (default
        RNA-seq) and pools the per-position track values across the
        sequence, yielding one value per output track -- the AlphaGenome
        analogue of ``BorzoiWrapper.embed``'s bin-pooled trunk embedding.

        Parameters
        ----------
        input : str
            Raw DNA sequence string (up to 1,048,576 bp).
        pooling_strategy : str, default "mean"
            "mean", "max", or "none" (returns the raw ``(positions, tracks)``
            array without pooling).
        output_type : str, default "RNA_SEQ"
            Name of an ``alphagenome.models.dna_output.OutputType`` member
            (e.g. "RNA_SEQ", "ATAC", "DNASE", "CAGE", "CHIP_HISTONE").
        ontology_terms : sequence of str, optional
            Restrict predictions to specific tissue/cell-type ontology
            CURIEs (e.g. ``"UBERON:0002107"`` for liver). ``None`` returns
            all tracks for the output type.
        **kwargs : Any
            Ignored; accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Shape ``(num_tracks,)``, or ``(num_positions, num_tracks)`` if
            ``pooling_strategy="none"``.

        Raises
        ------
        RuntimeError
            If the client hasn't been initialized (``load()`` not called).
        ValueError
            If ``pooling_strategy`` is invalid.
        """
        if self.client is None:
            raise RuntimeError("AlphaGenome client not initialized. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling: '{pooling_strategy}'")

        out_type = getattr(dna_output.OutputType, output_type)
        prediction = self.client.predict_sequence(
            sequence=input,
            organism=self._organism_enum(),
            requested_outputs=[out_type],
            ontology_terms=ontology_terms,
        )
        track_data = getattr(prediction, output_type.lower())
        values = np.asarray(track_data.values, dtype=np.float32)  # (positions, tracks)

        if pooling_strategy == "none":
            return values
        elif pooling_strategy == "mean":
            return values.mean(axis=0)
        else:
            return values.max(axis=0)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Sequentially embed a batch of DNA sequences (one API call each).

        AlphaGenome's client has no native multi-sequence batching for
        ``predict_sequence``, so this loops calling :meth:`embed` per input,
        mirroring ``APIEmbeddingWrapper``'s per-item API call pattern.

        Parameters
        ----------
        inputs : Sequence[str]
            DNA sequence strings.
        pooling_strategy : str, default "mean"
            Forwarded to :meth:`embed`.
        **kwargs : Any
            Forwarded to :meth:`embed`.

        Returns
        -------
        list[np.ndarray]
            One embedding per input, in order.
        """
        if self.client is None:
            raise RuntimeError("AlphaGenome client not initialized. Call load() first.")
        return [self.embed(seq, pooling_strategy=pooling_strategy, **kwargs) for seq in inputs]

    def get_track_metadata(self, output_type: str | None = None) -> pd.DataFrame:
        """Return AlphaGenome's output track metadata table.

        Analogue of ``BorzoiWrapper.get_track_metadata``, but fetched live
        from the API since AlphaGenome has no bundled local track table.

        Parameters
        ----------
        output_type : str, optional
            Restrict to one ``dna_output.OutputType`` name (e.g.
            "RNA_SEQ"). ``None`` concatenates metadata for every output
            type.

        Returns
        -------
        pandas.DataFrame
        """
        if self.client is None:
            raise RuntimeError("AlphaGenome client not initialized. Call load() first.")
        metadata = self.client.output_metadata(organism=self._organism_enum())
        if output_type is not None:
            return getattr(metadata, output_type.lower())
        frames = [df for df in vars(metadata).values() if isinstance(df, pd.DataFrame)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def predict_variant_effect(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        interval_start: int | None = None,
        interval_end: int | None = None,
        output_type: str = "RNA_SEQ",
        ontology_terms: Sequence[str] | None = None,
        variant_id: str = "",
    ) -> tuple[np.ndarray, list[str]]:
        """Score a variant's predicted effect on gene expression.

        Uses AlphaGenome's ``GeneMaskLFCScorer``, which sums predicted
        coverage over each gene's exons for the REF and ALT alleles and
        returns the log2 fold-change -- the same statistic and the same
        VCF-style chrom/pos/ref/alt input convention as
        ``embpy.tl.genomics.snp_utils.profile_variant_effect_score`` /
        ``SNPEmbedder.predict_variant_effect``, which underlie
        ``BorzoiWrapper``-based variant scoring elsewhere in this codebase.

        Parameters
        ----------
        chrom, pos, ref, alt : str, int, str, str
            VCF-style variant fields. ``pos`` is 1-based.
        interval_start, interval_end : int, optional
            0-based, half-open genomic window to score within. If either is
            ``None``, a window of ``self.context_window`` bp centered on
            ``pos`` is used.
        output_type : str, default "RNA_SEQ"
            Assay to score; one of ``alphagenome.models.dna_output.OutputType``
            names.
        ontology_terms : sequence of str, optional
            Restrict scoring to specific tissue/cell-type ontology CURIEs
            (e.g. ``"UBERON:0002107"`` for liver). ``None`` scores all
            tracks for the output type.
        variant_id : str, optional
            Optional identifier (e.g. rsID) attached to the variant.

        Returns
        -------
        scores : np.ndarray
            Shape ``(num_tracks,)``: ``log2(ALT/REF)`` gene-exon coverage
            fold-change per track.
        track_names : list[str]
            Track identifiers aligned with ``scores``.

        Raises
        ------
        RuntimeError
            If the client hasn't been initialized (``load()`` not called).
        """
        if self.client is None:
            raise RuntimeError("AlphaGenome client not initialized. Call load() first.")

        if interval_start is None or interval_end is None:
            interval = self._default_interval(chrom, pos)
        else:
            interval = genome.Interval(chromosome=chrom, start=interval_start, end=interval_end)

        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=ref,
            alternate_bases=alt,
            name=variant_id,
        )
        scorer = variant_scorers.GeneMaskLFCScorer(requested_output=getattr(dna_output.OutputType, output_type))

        scored = self.client.score_variant(
            interval=interval,
            variant=variant,
            variant_scorers=[scorer],
            organism=self._organism_enum(),
        )
        df = variant_scorers.tidy_scores(scored)
        if ontology_terms is not None:
            df = df[df["ontology_curie"].isin(set(ontology_terms))]

        scores = df["raw_score"].to_numpy(dtype=np.float32)
        track_names = df["track_name"].tolist()
        return scores, track_names
