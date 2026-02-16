from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData


class PerturbationProcessor:
    """
    Preprocessor for perturbation experimental data.

    Provides utilities to normalize identifiers (gene symbols, SMILES, etc.),
    resolve them to canonical forms, and build embedding matrices from
    BioEmbedder results, ready for downstream analysis.

    Parameters
    ----------
    embedder : BioEmbedder, optional
        An initialized BioEmbedder instance. If provided, used for
        resolving identifiers and computing embeddings.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def normalize_gene_names(
        self,
        identifiers: Sequence[str],
        organism: str = "human",
    ) -> dict[str, str | None]:
        """
        Normalize gene identifiers to canonical Ensembl IDs.

        Handles common issues: case normalization, alias resolution,
        deprecated symbol mapping.

        Parameters
        ----------
        identifiers
            List of gene symbols or Ensembl IDs to normalize.
        organism
            Target organism for resolution. Defaults to "human".

        Returns
        -------
        dict mapping input identifiers to canonical Ensembl gene IDs,
        or None where resolution failed.
        """
        raise NotImplementedError("normalize_gene_names will be implemented in a future release.")

    def resolve_identifiers(
        self,
        identifiers: Sequence[str],
        id_type: str = "auto",
        organism: str = "human",
    ) -> pd.DataFrame:
        """
        Resolve a mixed list of identifiers to a structured DataFrame.

        Auto-detects identifier types (gene symbol, Ensembl ID, SMILES, etc.)
        and resolves each to its canonical form with metadata.

        Parameters
        ----------
        identifiers
            List of identifiers to resolve.
        id_type
            Type hint: "auto", "gene_symbol", "ensembl_id", "smiles", "drug_name".
        organism
            Target organism for gene resolution.

        Returns
        -------
        DataFrame with columns: original_id, canonical_id, id_type, resolved (bool).
        """
        raise NotImplementedError("resolve_identifiers will be implemented in a future release.")

    def build_embedding_matrix(
        self,
        identifiers: Sequence[str],
        model: str,
        id_type: str = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
    ) -> AnnData:
        """
        Build an AnnData object with embeddings for a list of perturbations.

        Resolves identifiers, computes embeddings via BioEmbedder, and
        returns a structured AnnData where obs are perturbations and
        X contains embedding vectors.

        Parameters
        ----------
        identifiers
            List of perturbation identifiers (genes, molecules, etc.).
        model
            Model name to use for embedding (e.g., "esm2_650M", "chemberta2MTR").
        id_type
            Type of identifiers provided.
        organism
            Target organism.
        pooling_strategy
            Pooling strategy for the embedding model.

        Returns
        -------
        AnnData with shape (n_perturbations, embedding_dim).
        obs contains metadata about each perturbation.
        """
        raise NotImplementedError("build_embedding_matrix will be implemented in a future release.")

    def filter_failed_embeddings(
        self,
        adata: AnnData,
    ) -> AnnData:
        """
        Remove perturbations that failed to embed from an AnnData object.

        Parameters
        ----------
        adata
            AnnData returned by build_embedding_matrix.

        Returns
        -------
        Filtered AnnData with failed entries removed.
        """
        raise NotImplementedError("filter_failed_embeddings will be implemented in a future release.")

    def combine_perturbation_spaces(
        self,
        genetic: AnnData | None = None,
        molecular: AnnData | None = None,
    ) -> AnnData:
        """
        Combine genetic and molecular perturbation embedding spaces.

        Concatenates embedding matrices and adds metadata to distinguish
        perturbation types, enabling joint analysis.

        Parameters
        ----------
        genetic
            AnnData of gene perturbation embeddings.
        molecular
            AnnData of small molecule perturbation embeddings.

        Returns
        -------
        Combined AnnData with a 'perturbation_type' column in obs.
        """
        raise NotImplementedError("combine_perturbation_spaces will be implemented in a future release.")


def basic_preproc(adata: AnnData) -> int:
    """Run a basic preprocessing on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    raise NotImplementedError("basic_preproc is a placeholder.")
