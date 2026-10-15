"""Model Context Protocol (MCP) server for embpy bioinformatics skills.

Serves the complete universal single-cell and multi-modal biological embedding pipeline
over standard input/output (stdio) with strict stdio isolation: all diagnostic logging,
warnings, and library prints are directed to stderr so that stdout carries exclusively
clean JSON-RPC protocol frames.
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

# Backward and forward compatibility across mcp 1.x and 2.x
try:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except (ImportError, ModuleNotFoundError):
        from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]
except (ImportError, ModuleNotFoundError):
    FastMCP = None  # type: ignore[misc,assignment]

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

logger = logging.getLogger("embpy.mcp")


def _setup_stderr_logging(level: int = logging.INFO) -> None:
    """Configure root and package logging strictly to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Ensure warnings also go to stderr
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers = []
    warnings_logger.addHandler(logging.StreamHandler(sys.stderr))


def create_server(name: str = "embpy-mcp") -> FastMCP:
    """Instantiate and configure the embpy MCP server with registered bioinformatics tools.

    Parameters
    ----------
    name : str, default="embpy-mcp"
        Identifier for the MCP server instance.

    Returns
    -------
    FastMCP
        Configured MCP server ready to serve requests over stdio or SSE.
    """
    _setup_stderr_logging()

    if FastMCP is None:
        raise ImportError(
            "The 'mcp' package is required to create an MCP server. "
            "Install it via 'pip install \"embpy[mcp]\"' or 'pip install mcp'."
        )

    server = FastMCP(
        name=name,
        instructions=(
            "embpy Universal MCP Server exposes an end-to-end biological representation pipeline: "
            "1. inspect_data: Inspect .h5ad metadata, layers, candidate label/batch columns, and count nature. "
            "2. resolve_identifiers: Harmonize biological IDs (Ensembl to HGNC, molecule names to SMILES). "
            "3. preprocess_counts: Filter, library size normalize, log1p, and select HVGs, saving to disk. "
            "4. list_model_registry: Check available foundation models (~150), modalities, and compute hardware. "
            "5. compute_embeddings: Run multi-model embeddings (PCA, scVI, foundation models) via BioEmbedder. "
            "6. annotate_metadata: Enrich AnnData with GO terms, pathways, disease associations, and pharmacology. "
            "7. compute_umap: Compute kNN graph and 2D UMAP coordinates on chosen representation. "
            "8. benchmark_scib: Benchmark embeddings on biological conservation and batch integration. "
            "9. benchmark_perturbations: Score perturbation prediction accuracy via ArcInstitute cell-eval."
        ),
    )

    # Register 9 universal tools
    server.tool(name="inspect_data")(inspect_data)
    server.tool(name="resolve_identifiers")(resolve_identifiers)
    server.tool(name="preprocess_counts")(preprocess_counts)
    server.tool(name="list_model_registry")(list_model_registry)
    server.tool(name="compute_embeddings")(compute_embeddings)
    server.tool(name="annotate_metadata")(annotate_metadata)
    server.tool(name="compute_umap")(compute_umap)
    server.tool(name="benchmark_scib")(benchmark_scib)
    server.tool(name="benchmark_perturbations")(benchmark_perturbations)

    # Register backward-compatible aliases
    server.tool(name="inspect_anndata")(inspect_anndata)
    server.tool(name="map_gene_identifiers")(map_gene_identifiers)
    server.tool(name="list_available_models")(list_available_models)
    server.tool(name="embed_cells")(embed_cells)
    server.tool(name="evaluate_scib")(evaluate_scib)
    server.tool(name="evaluate_perturbations")(evaluate_perturbations)

    logger.info("embpy MCP server initialized with 9 universal tools and aliases.")
    return server


def main() -> None:
    """CLI entrypoint for embpy-mcp command."""
    _setup_stderr_logging()
    logger.info("Starting embpy MCP server on stdio transport...")
    server = create_server()
    try:
        server.run(transport="stdio")
    except (KeyboardInterrupt, SystemExit):
        logger.info("embpy MCP server shutting down cleanly.")
    except Exception as exc:
        logger.critical("embpy MCP server encountered fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
