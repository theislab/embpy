"""Canonical embedding output layer.

One in-memory result (:class:`EmbeddingResult`) is the single source of
truth; pure exporter functions turn it into the two supported backends
(AnnData, table, or payload dict) at the edges. See :mod:`embpy.io.result`
for the layering rule.
"""

from __future__ import annotations

from .exporters import (
    materialize_perturbation_obsm,
    route_output,
    to_anndata,
    to_anndata_many,
    to_payload,
    to_payloads,
    to_table,
    to_tables,
)
from .harmonize import harmonize
from .legacy import load_legacy_embedding
from .normalize import NormalizedInput, normalize_embedding_input
from .result import EmbeddingProvenance, EmbeddingResult

__all__ = [
    "EmbeddingProvenance",
    "EmbeddingResult",
    "NormalizedInput",
    "harmonize",
    "load_legacy_embedding",
    "materialize_perturbation_obsm",
    "normalize_embedding_input",
    "route_output",
    "to_anndata",
    "to_anndata_many",
    "to_payload",
    "to_payloads",
    "to_table",
    "to_tables",
]
