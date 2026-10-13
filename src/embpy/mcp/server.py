"""Model Context Protocol (MCP) server for embpy single-cell bioinformatics skills.

Serves tools over standard input/output (stdio) with strict stdio isolation:
all diagnostic logging, warnings, and library prints are directed to stderr
so that stdout carries exclusively clean JSON-RPC protocol frames.
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

# Backward and forward compatibility across mcp 1.x and 2.x
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

from .tools import embed_cells, evaluate_scib, inspect_anndata

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

    server = FastMCP(
        name=name,
        instructions=(
            "embpy MCP Server provides bioinformatics and single-cell machine learning skills "
            "for inspecting AnnData (.h5ad) datasets, computing biological foundation-model and "
            "PCA embeddings via BioEmbedder, and evaluating representation quality using scIB metrics."
        ),
    )

    # Register core single-cell bioinformatics tools
    server.tool(name="inspect_anndata")(inspect_anndata)
    server.tool(name="embed_cells")(embed_cells)
    server.tool(name="evaluate_scib")(evaluate_scib)

    logger.info("embpy MCP server initialized with %d tools: inspect_anndata, embed_cells, evaluate_scib", 3)
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
