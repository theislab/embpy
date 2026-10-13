"""Standalone client test harness for live execution of embpy MCP server over stdio.

Spawns 'python -m embpy.mcp.server' as a sub-process, establishes a JSON-RPC session
over stdio, queries the tool list, and executes the complete 7-tool suite live:
inspect_anndata, list_available_models, map_gene_identifiers, embed_cells,
evaluate_scib, evaluate_perturbations, and compute_umap.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import numpy as np
from anndata import AnnData
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


def create_sample_h5ad(target_dir: str) -> str:
    """Create a self-contained sample .h5ad dataset for live MCP testing."""
    os.makedirs(target_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    n_obs, n_vars = 24, 30
    X = rng.poisson(lam=3.0, size=(n_obs, n_vars)).astype(np.float32)

    ad = AnnData(X=X)
    ad.obs_names = [f"cell_{i}" for i in range(n_obs)]
    ad.var_names = [f"gene_{i}" for i in range(n_vars)]
    ad.obs["cell_type"] = (["CD4_T", "CD8_T", "B_cell"] * (n_obs // 3))[:n_obs]
    ad.obs["batch"] = (["tech_1", "tech_2"] * (n_obs // 2))[:n_obs]
    ad.obs["perturbation"] = (["ctrl", "ctrl", "pert_stim", "pert_stim"] * 10)[:n_obs]
    ad.obs["split"] = (["train", "test"] * 20)[:n_obs]

    ad.layers["counts"] = X.copy()

    file_path = os.path.join(target_dir, "mcp_test_dataset.h5ad")
    ad.write_h5ad(file_path)
    return file_path


async def run_live_mcp_harness() -> None:
    """Run live JSON-RPC end-to-end tests against embpy MCP server."""
    test_dir = tempfile.mkdtemp(prefix="embpy_mcp_harness_")
    try:
        sample_path = create_sample_h5ad(test_dir)
        embedded_path = os.path.join(test_dir, "mcp_embedded.h5ad")
        umap_path = os.path.join(test_dir, "mcp_with_umap.h5ad")
        scib_csv_path = os.path.join(test_dir, "mcp_scib_metrics.csv")
        pert_csv_path = os.path.join(test_dir, "mcp_pert_metrics.csv")

        print(f"[TEST HARNESS] Starting embpy MCP server sub-process with Python: {sys.executable}")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "embpy.mcp.server"],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                print("[TEST HARNESS] Initializing MCP protocol session...")
                init_result = await session.initialize()
                print(f"[TEST HARNESS] Connected to server: {init_result.server_info.name} (v{init_result.server_info.version})")

                # 1. tools/list verification
                print("\n[TEST HARNESS] 1. Requesting tools/list...")
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                print(f"[TEST HARNESS] Registered {len(tool_names)} tools: {tool_names}")
                expected_tools = (
                    "inspect_anndata",
                    "list_available_models",
                    "map_gene_identifiers",
                    "embed_cells",
                    "evaluate_scib",
                    "evaluate_perturbations",
                    "compute_umap",
                )
                for tool in expected_tools:
                    assert tool in tool_names, f"Expected tool '{tool}' missing from server!"

                # 2. tools/call: list_available_models
                print("\n[TEST HARNESS] 2. Executing list_available_models...")
                models_call = await session.call_tool("list_available_models", {})
                assert not models_call.is_error
                models_payload = json.loads(models_call.content[0].text)
                assert models_payload["status"] == "success"
                print(f"               Compute device: {models_payload['compute_device']} ({models_payload['device_details']})")
                print(f"               Runnable models: {models_payload['runnable_models']}")

                # 3. tools/call: map_gene_identifiers
                print("\n[TEST HARNESS] 3. Executing map_gene_identifiers...")
                map_call = await session.call_tool(
                    "map_gene_identifiers",
                    {"genes": ["TP53", "GAPDH", "CD4"], "target_type": "ensembl", "organism": "human"},
                )
                assert not map_call.is_error
                map_payload = json.loads(map_call.content[0].text)
                assert map_payload["status"] == "success"
                print(f"               Mapped {map_payload['mapped_count']}/{map_payload['total_genes']} genes ({map_payload['mapping_rate']}%)")
                print(f"               Sample: {map_payload['mapping_sample']}")

                # 4. tools/call: inspect_anndata
                print("\n[TEST HARNESS] 4. Executing inspect_anndata...")
                inspect_call = await session.call_tool("inspect_anndata", {"path": sample_path})
                assert not inspect_call.is_error
                inspect_payload = json.loads(inspect_call.content[0].text)
                assert inspect_payload["status"] == "success"
                print(f"               Dimensions: {inspect_payload['n_obs']} cells, {inspect_payload['n_vars']} genes")
                print(f"               Expression nature: {inspect_payload['expression_nature']['detected_nature']}")

                # 5. tools/call: embed_cells
                print("\n[TEST HARNESS] 5. Executing embed_cells (PCA baseline)...")
                embed_call = await session.call_tool(
                    "embed_cells",
                    {
                        "path": sample_path,
                        "models": ["pca"],
                        "output_path": embedded_path,
                        "preprocessing": "none",
                        "device": "cpu",
                    },
                )
                assert not embed_call.is_error
                embed_payload = json.loads(embed_call.content[0].text)
                assert embed_payload["status"] == "success"
                assert "X_pca" in embed_payload["embedding_keys"]
                print(f"               Embedding computed: {embed_payload['embedding_keys']}")
                print(f"               Shapes: {embed_payload['embedding_shapes']}")

                # 6. tools/call: compute_umap
                print("\n[TEST HARNESS] 6. Executing compute_umap...")
                umap_call = await session.call_tool(
                    "compute_umap",
                    {"path": embedded_path, "embedding_key": "X_pca", "output_path": umap_path, "n_neighbors": 5},
                )
                assert not umap_call.is_error
                umap_payload = json.loads(umap_call.content[0].text)
                assert umap_payload["status"] == "success"
                print(f"               UMAP key: {umap_payload['umap_key']}, shape: {umap_payload['umap_shape']}")

                # 7. tools/call: evaluate_scib
                print("\n[TEST HARNESS] 7. Executing evaluate_scib...")
                scib_call = await session.call_tool(
                    "evaluate_scib",
                    {
                        "path": embedded_path,
                        "embedding_keys": ["X_pca"],
                        "label_key": "cell_type",
                        "batch_key": "batch",
                        "output_csv": scib_csv_path,
                    },
                )
                assert not scib_call.is_error
                scib_payload = json.loads(scib_call.content[0].text)
                assert scib_payload["status"] == "success"
                print(f"               Total scIB score: {scib_payload['metrics']['X_pca'].get('total')}")
                print(f"               Markdown table generated: {'Yes' if scib_payload.get('markdown_table') else 'No'}")

                # 8. tools/call: evaluate_perturbations
                print("\n[TEST HARNESS] 8. Executing evaluate_perturbations...")
                pert_call = await session.call_tool(
                    "evaluate_perturbations",
                    {
                        "path": sample_path,
                        "split_col": "split",
                        "control_pert": "ctrl",
                        "pert_col": "perturbation",
                        "profile": "minimal",
                        "output_csv": pert_csv_path,
                    },
                )
                assert not pert_call.is_error
                pert_payload = json.loads(pert_call.content[0].text)
                assert pert_payload["status"] == "success"
                print(f"               Evaluated {pert_payload['n_perturbations']} perturbations.")
                print(f"               Markdown table generated: {'Yes' if pert_payload.get('markdown_table') else 'No'}")

        print("\n[TEST HARNESS] All 7 live MCP tool executions PASSED successfully with pure JSON-RPC stdout!")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_live_mcp_harness())
