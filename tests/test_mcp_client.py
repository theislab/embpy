"""Call the embpy MCP server through a real MCP client.

Each test spawns ``python -m embpy.mcp.server`` and talks to it over stdio with
the SDK's ``ClientSession``, so tool registration, argument validation and
result serialization are exercised exactly as an assistant would see them.
Works with MCP SDK 1.x (camelCase result fields) and 2.x (snake_case).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

if os.environ.get("EMBPY_REQUIRE_MCP") == "1":
    import mcp  # noqa: F401  -- the MCP CI job must fail, not skip, without it
else:
    pytest.importorskip("mcp", reason="mcp not installed")

from mcp import StdioServerParameters  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _field(obj, snake: str, camel: str):
    value = getattr(obj, snake, None)
    return getattr(obj, camel, None) if value is None else value


def _payload(result) -> dict:
    """Return the tool's JSON payload from a CallToolResult of either SDK major version."""
    assert not _field(result, "is_error", "isError"), result
    structured = _field(result, "structured_content", "structuredContent")
    if structured is None:
        structured = json.loads(result.content[0].text)
    # SDKs that wrap non-object returns put them under "result"; ours are objects.
    if set(structured) == {"result"} and isinstance(structured["result"], dict):
        structured = structured["result"]
    return structured


def _run(scenario):
    async def _main():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "embpy.mcp.server"],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await scenario(session)

    asyncio.run(_main())


@pytest.fixture
def cells_path(tmp_path):
    rng = np.random.RandomState(0)
    n_obs, n_vars = 30, 40
    adata = AnnData(X=rng.poisson(lam=2.5, size=(n_obs, n_vars)).astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.obs["cell_type"] = (["T_cell", "B_cell", "Monocyte"] * n_obs)[:n_obs]
    adata.obs["batch"] = (["batch_1", "batch_2"] * n_obs)[:n_obs]
    adata.obs["perturbation"] = (["ctrl", "ctrl", "pert_1", "pert_1"] * n_obs)[:n_obs]
    adata.obs["split"] = (["train", "test"] * n_obs)[:n_obs]
    path = str(tmp_path / "cells.h5ad")
    adata.write_h5ad(path)
    return path


@pytest.fixture
def molecules_path(tmp_path):
    adata = AnnData(obs=pd.DataFrame({"smiles": ["CCO", "CCN"]}, index=["sample_A", "sample_B"]))
    path = str(tmp_path / "molecules.h5ad")
    adata.write_h5ad(path)
    return path


def test_tools_and_input_options_are_advertised():
    async def scenario(session):
        tools = {t.name: t for t in (await session.list_tools()).tools}
        for name in (
            "inspect_data",
            "resolve_identifiers",
            "preprocess_counts",
            "list_model_registry",
            "compute_embeddings",
            "annotate_metadata",
            "compute_umap",
            "benchmark_scib",
            "benchmark_perturbations",
        ):
            assert name in tools, name
        embed_params = _field(tools["compute_embeddings"], "input_schema", "inputSchema")["properties"]
        for option in ("obs_column", "var_column", "input_axis", "id_type", "organism"):
            assert option in embed_params, option
        resolve_params = _field(tools["resolve_identifiers"], "input_schema", "inputSchema")["properties"]
        assert "organism" in resolve_params and "strict" in resolve_params

    _run(scenario)


def test_single_cell_workflow_through_client(cells_path, tmp_path):
    has_scib = __import__("importlib").util.find_spec("scib") is not None
    has_cell_eval = __import__("importlib").util.find_spec("cell_eval") is not None

    async def scenario(session):
        inspect = _payload(await session.call_tool("inspect_data", {"path": cells_path}))
        assert inspect["status"] == "success" and inspect["n_obs"] == 30

        registry = _payload(await session.call_tool("list_model_registry", {"entity_type": "single_cell"}))
        assert "pca" in registry["recommended_models"]["single_cell"]

        prepped = str(tmp_path / "prepped.h5ad")
        prep = _payload(
            await session.call_tool(
                "preprocess_counts", {"path": cells_path, "min_genes": 5, "scale": True, "output_path": prepped}
            )
        )
        assert prep["status"] == "success" and "scaled" in prep["layers"]

        embedded = str(tmp_path / "embedded.h5ad")
        emb = _payload(
            await session.call_tool(
                "compute_embeddings",
                {"path": cells_path, "models": ["pca"], "preprocessing": "none", "device": "cpu", "output_path": embedded},
            )
        )
        assert emb["status"] == "success" and "X_pca" in emb["embedding_keys"]

        umap = _payload(
            await session.call_tool(
                "compute_umap", {"path": embedded, "embedding_key": "X_pca", "n_neighbors": 5, "output_path": str(tmp_path / "u.h5ad")}
            )
        )
        assert umap["status"] == "success" and umap["umap_shape"] == [30, 2]

        if has_scib:
            scib = _payload(
                await session.call_tool(
                    "benchmark_scib",
                    {"path": embedded, "embedding_keys": ["X_pca"], "label_key": "cell_type", "batch_key": "batch"},
                )
            )
            assert scib["status"] == "success" and "X_pca" in scib["metrics"]

        same = _payload(
            await session.call_tool(
                "benchmark_perturbations",
                {"path": cells_path, "split_col": "split", "pred_label": "test", "ref_label": "test", "control_pert": "ctrl"},
            )
        )
        assert same["status"] == "error" and "themselves" in same["error"]
        if has_cell_eval:
            pert = _payload(
                await session.call_tool(
                    "benchmark_perturbations",
                    {"path": cells_path, "split_col": "split", "pred_label": "test", "ref_label": "train", "control_pert": "ctrl"},
                )
            )
            assert pert["status"] == "success"

    _run(scenario)


def test_entity_tools_through_client(molecules_path, tmp_path):
    async def scenario(session):
        res = _payload(await session.call_tool("resolve_identifiers", {"identifiers": ["TP53", "BRCA1"]}))
        assert res["status"] == "success"
        assert res["sample_mapping"]["TP53"] == "ENSG00000141510"

        # Capitalization does not override the requested species.
        egfr = _payload(await session.call_tool("resolve_identifiers", {"identifiers": ["Egfr"], "organism": "human"}))
        assert egfr["sample_mapping"]["Egfr"] == "ENSG00000146648"
        mismatch = _payload(
            await session.call_tool("resolve_identifiers", {"identifiers": ["ENSMUSG00000059552"], "organism": "human"})
        )
        assert mismatch["error_type"] == "SpeciesMismatchError"

        out = str(tmp_path / "mol_embedded.h5ad")
        emb = _payload(
            await session.call_tool(
                "compute_embeddings",
                {
                    "path": molecules_path,
                    "models": "morgan_fp",
                    "entity_type": "molecule",
                    "obs_column": "smiles",
                    "device": "cpu",
                    "output_path": out,
                },
            )
        )
        assert emb["status"] == "success", emb
        (key,) = emb["embedding_keys"]
        assert emb["embedding_shapes"][key] == [2, 2048]

        wrong = _payload(
            await session.call_tool(
                "compute_embeddings",
                {"path": molecules_path, "models": "state", "entity_type": "perturbation", "obs_column": "smiles"},
            )
        )
        assert wrong["status"] == "error" and wrong["error_type"] == "UnsupportedModelError"

    _run(scenario)


def test_invalid_arguments_are_rejected_by_the_schema(cells_path):
    async def scenario(session):
        result = await session.call_tool("resolve_identifiers", {"identifiers": ["TP53"], "target_id_type": "banana"})
        assert _field(result, "is_error", "isError")

    _run(scenario)
