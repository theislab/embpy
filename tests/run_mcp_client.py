"""Standalone client test harness for live execution of embpy MCP server over stdio.

Spawns 'python -m embpy.mcp.server' as a sub-process, establishes a JSON-RPC session
over stdio, queries the tool list, and executes inspect_anndata, embed_cells, and
evaluate_scib live against a test dataset.
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
        scib_csv_path = os.path.join(test_dir, "mcp_scib_metrics.csv")

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
                print("[TEST HARNESS] Requesting tools/list...")
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                print(f"[TEST HARNESS] Tools registered: {tool_names}")
                for required in ("inspect_anndata", "embed_cells", "evaluate_scib"):
                    assert required in tool_names, f"Expected tool '{required}' missing from server!"

                # 2. tools/call: inspect_anndata
                print("\n[TEST HARNESS] Executing tools/call: inspect_anndata...")
                inspect_call = await session.call_tool("inspect_anndata", {"path": sample_path})
                assert not inspect_call.is_error, "inspect_anndata returned is_error=True"
                inspect_payload = json.loads(inspect_call.content[0].text)
                print(f"[TEST HARNESS] inspect_anndata status: {inspect_payload.get('status')}")
                assert inspect_payload["status"] == "success"
                assert inspect_payload["n_obs"] == 24
                assert inspect_payload["n_vars"] == 30
                assert "cell_type" in inspect_payload["obs_columns"]
                print(f"               n_obs: {inspect_payload['n_obs']}, n_vars: {inspect_payload['n_vars']}")

                # 3. tools/call: embed_cells
                print("\n[TEST HARNESS] Executing tools/call: embed_cells (PCA baseline)...")
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
                assert not embed_call.is_error, "embed_cells returned is_error=True"
                embed_payload = json.loads(embed_call.content[0].text)
                print(f"[TEST HARNESS] embed_cells status: {embed_payload.get('status')}")
                assert embed_payload["status"] == "success"
                assert "X_pca" in embed_payload["embedding_keys"]
                assert os.path.exists(embedded_path), "Saved embedded .h5ad was not created!"
                print(f"               Computed embeddings: {embed_payload['embedding_keys']}")
                print(f"               Embedding shapes: {embed_payload['embedding_shapes']}")

                # 4. tools/call: evaluate_scib
                print("\n[TEST HARNESS] Executing tools/call: evaluate_scib...")
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
                assert not scib_call.is_error, "evaluate_scib returned is_error=True"
                scib_payload = json.loads(scib_call.content[0].text)
                print(f"[TEST HARNESS] evaluate_scib status: {scib_payload.get('status')}")
                assert scib_payload["status"] == "success"
                assert "X_pca" in scib_payload["metrics"]
                assert os.path.exists(scib_csv_path), "Exported scIB CSV was not created!"
                print(f"               Scored metrics: {list(scib_payload['metrics']['X_pca'].keys())}")
                print(f"               Total score: {scib_payload['metrics']['X_pca'].get('total')}")

        print("\n[TEST HARNESS] All live MCP tool executions PASSED successfully with pure JSON-RPC stdout!")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_live_mcp_harness())
