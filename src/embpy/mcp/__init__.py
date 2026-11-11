"""Model Context Protocol (MCP) server and bioinformatics skills for embpy."""

from __future__ import annotations

from typing import Any

from .tools import (
    annotate_metadata,
    benchmark_perturbations,
    benchmark_scib,
    compute_embeddings,
    compute_umap,
    embed_cells,
    evaluate_perturbations,
    evaluate_scib,
    inspect_anndata,
    inspect_data,
    list_available_models,
    list_model_registry,
    map_gene_identifiers,
    preprocess_counts,
    resolve_identifiers,
)

__all__ = [
    "create_server",
    "main",
    # Universal 9 tools
    "inspect_data",
    "resolve_identifiers",
    "preprocess_counts",
    "list_model_registry",
    "compute_embeddings",
    "annotate_metadata",
    "compute_umap",
    "benchmark_scib",
    "benchmark_perturbations",
    # Aliases
    "inspect_anndata",
    "list_available_models",
    "map_gene_identifiers",
    "embed_cells",
    "evaluate_scib",
    "evaluate_perturbations",
]


def __getattr__(name: str) -> Any:
    # Imported on first use rather than with the package: `python -m embpy.mcp.server`
    # imports this package first, and an eager import of .server here makes runpy warn
    # that the module was already imported, which MCP clients log as a server error.
    if name in ("create_server", "main"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
