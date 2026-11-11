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
            "embpy MCP server: tools for a biological representation pipeline. "
            "1. inspect_data: .h5ad dimensions, layers, candidate label/batch columns, raw vs normalized counts. "
            "2. resolve_identifiers: gene symbols <-> Ensembl ids, compound names/SMILES -> canonical SMILES. "
            "The organism argument is always respected; omit it to use human (Ensembl prefixes still decide), "
            "or pass 'auto' to opt into species detection. "
            "3. preprocess_counts: filter, normalize, log1p, select HVGs, save to disk. "
            "4. list_model_registry: models each modality can run, and the compute device. "
            "5. compute_embeddings: run models for one modality; choose the input with obs_column / var_column. "
            "Models are validated against the modality before anything runs. "
            "6. annotate_metadata: GO, pathways, domains, diseases, interactions, expression; reports "
            "per type how many entities were annotated, had no match, or hit a service error. "
            "7. compute_umap: kNN graph and 2D UMAP on a chosen representation. "
            "8. benchmark_scib: scIB bio-conservation / batch-integration metrics. "
            "9. benchmark_perturbations: cell-eval scoring of predicted vs observed perturbations "
            "(prediction and reference must be different groups). "
            "Large data stays on disk; tools return compact JSON."
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


# Fortran-backed libraries the tools load lazily. On Windows with MCP SDK 1.x
# the stdio reader sits in a blocking ReadFile on stdin, and a Fortran runtime
# (scipy's BLAS/LAPACK wrappers) that initialises while that read is pending
# waits behind it on the stdin handle, so the first tool call that imports
# scipy.linalg never returns. Importing them before the transport starts avoids
# it. Keep the list short: it delays the handshake, and clients give a server
# about 30 s to answer (Claude Code's default MCP_TIMEOUT).
_PRELOAD_MODULES = (
    "scipy.linalg",
    "scipy.sparse.linalg",
    "scipy.special",
    "scipy.optimize",
    "scipy.stats",
    "scipy.spatial",
    "scipy.interpolate",
    "sklearn.decomposition",
    "sklearn.neighbors",
)


def _needs_native_preload() -> bool:
    """Only Windows with MCP SDK 1.x reads stdin in a way that deadlocks."""
    if sys.platform != "win32":
        return False
    try:
        from importlib.metadata import version

        return int(version("mcp").split(".")[0]) < 2
    except Exception:  # noqa: BLE001 -- unknown version: preload to be safe
        return True


def _preload_native_modules() -> None:
    import importlib

    for name in _PRELOAD_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- optional dependencies may be absent
            logger.debug("Preloading %s skipped: %s", name, exc)


def main() -> None:
    """CLI entrypoint for embpy-mcp command."""
    _setup_stderr_logging()
    if _needs_native_preload():
        _preload_native_modules()
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
