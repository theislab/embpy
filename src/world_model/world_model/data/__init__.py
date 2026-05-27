"""Data layer: datasets, preprocessing, action embeddings, dataloaders."""

from __future__ import annotations

from .dataloader import DataArtifacts, build_dataloaders
from .datasets.anndata import AnnDataSequenceDataset
from .datasets.base import (
    GeneIndexer,
    PerturbationSequenceDataset,
    SequenceSample,
    load_gene_embedding_table,
    load_obsm_state_matrix,
)
from .embeddings import (
    ActionEmbeddingProvider,
    BioEmbedderProvider,
    EmbeddingCacheKey,
    PrecomputedProvider,
    build_provider,
)
from .preprocessing import sequence_collate_fn
from .splits import (
    SplitArtifact,
    load_split,
    make_split,
    save_split,
    split_or_load,
)

__all__ = [
    "ActionEmbeddingProvider",
    "AnnDataSequenceDataset",
    "BioEmbedderProvider",
    "DataArtifacts",
    "EmbeddingCacheKey",
    "GeneIndexer",
    "PerturbationSequenceDataset",
    "PrecomputedProvider",
    "SequenceSample",
    "SplitArtifact",
    "build_dataloaders",
    "build_provider",
    "load_gene_embedding_table",
    "load_obsm_state_matrix",
    "load_split",
    "make_split",
    "save_split",
    "sequence_collate_fn",
    "split_or_load",
]
