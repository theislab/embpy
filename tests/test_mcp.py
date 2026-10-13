"""Automated unit and integration tests for embpy MCP server and bioinformatics tools."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
from anndata import AnnData

from embpy.mcp import create_server, embed_cells, evaluate_scib, inspect_anndata


@pytest.fixture
def sample_adata_path(tmp_path):
    """Create a self-contained AnnData .h5ad fixture for tool execution."""
    rng = np.random.RandomState(42)
    n_obs, n_vars = 30, 40
    X = rng.poisson(lam=2.5, size=(n_obs, n_vars)).astype(np.float32)

    ad = AnnData(X=X)
    ad.obs_names = [f"cell_{i}" for i in range(n_obs)]
    ad.var_names = [f"gene_{i}" for i in range(n_vars)]

    ad.obs["cell_type"] = (["T_cell", "B_cell", "Monocyte"] * (n_obs // 3))[:n_obs]
    ad.obs["batch"] = (["batch_1", "batch_2"] * (n_obs // 2))[:n_obs]
    ad.layers["counts"] = X.copy()
    ad.obsm["X_prior"] = rng.randn(n_obs, 8).astype(np.float32)

    file_path = str(tmp_path / "sample_test_data.h5ad")
    ad.write_h5ad(file_path)
    return file_path


def test_inspect_anndata_success(sample_adata_path):
    """Test inspect_anndata accurately reads dimensions, slots, and metadata."""
    res = inspect_anndata(sample_adata_path)
    assert res["status"] == "success"
    assert res["n_obs"] == 30
    assert res["n_vars"] == 40
    assert "cell_type" in res["obs_columns"]
    assert "batch" in res["obs_columns"]
    assert "counts" in res["layers"]
    assert "X_prior" in res["obsm_keys"]
    assert "cell_type" in res["obs_summary"]
    assert res["obs_summary"]["cell_type"]["n_unique"] == 3


def test_inspect_anndata_missing_file():
    """Test inspect_anndata returns actionable error when file is missing."""
    res = inspect_anndata("non_existent_dataset.h5ad")
    assert res["status"] == "error"
    assert res["error_type"] == "FileNotFoundError"
    assert "not found" in res["error"].lower()


def test_embed_cells_success(sample_adata_path, tmp_path):
    """Test embed_cells runs model inference (PCA) and produces valid embeddings."""
    out_path = str(tmp_path / "embedded_cells.h5ad")
    res = embed_cells(
        path=sample_adata_path,
        models=["pca"],
        output_path=out_path,
        preprocessing="none",
        device="cpu",
    )
    assert res["status"] == "success"
    assert res["n_cells"] == 30
    assert "pca" in res["models_computed"]
    assert "X_pca" in res["embedding_keys"]
    assert os.path.exists(out_path)

    # Verify written output matches reported shapes
    import anndata

    saved_ad = anndata.read_h5ad(out_path)
    assert "X_pca" in saved_ad.obsm
    assert saved_ad.obsm["X_pca"].shape[0] == 30


def test_embed_cells_invalid_inputs(sample_adata_path):
    """Test embed_cells gracefully handles invalid paths and empty rosters."""
    err_path = embed_cells(path="missing_file.h5ad", models=["pca"])
    assert err_path["status"] == "error"
    assert err_path["error_type"] == "FileNotFoundError"

    err_models = embed_cells(path=sample_adata_path, models=[])
    assert err_models["status"] == "error"
    assert err_models["error_type"] == "ValueError"


def test_evaluate_scib_success(sample_adata_path, tmp_path):
    """Test evaluate_scib evaluates an existing embedding against annotations."""
    # First embed with PCA
    embedded_path = str(tmp_path / "for_scib.h5ad")
    embed_res = embed_cells(
        path=sample_adata_path,
        models=["pca"],
        output_path=embedded_path,
        preprocessing="none",
        device="cpu",
    )
    assert embed_res["status"] == "success"

    csv_path = str(tmp_path / "scib_scores.csv")
    eval_res = evaluate_scib(
        path=embedded_path,
        embedding_keys=["X_pca"],
        label_key="cell_type",
        batch_key="batch",
        output_csv=csv_path,
    )
    assert eval_res["status"] == "success"
    assert "X_pca" in eval_res["metrics"]
    assert os.path.exists(csv_path)

    metrics_pca = eval_res["metrics"]["X_pca"]
    assert "bio_conservation" in metrics_pca
    assert "total" in metrics_pca
    assert len(eval_res["summary"]) == 1


def test_evaluate_scib_missing_keys(sample_adata_path):
    """Test evaluate_scib catches missing obs labels or obsm embedding keys."""
    # Label key absent
    res_bad_label = evaluate_scib(
        path=sample_adata_path,
        embedding_keys=["X_prior"],
        label_key="missing_label_col",
    )
    assert res_bad_label["status"] == "error"
    assert res_bad_label["error_type"] == "KeyError"
    assert "label_key" in res_bad_label["error"]

    # Embedding key absent
    res_bad_emb = evaluate_scib(
        path=sample_adata_path,
        embedding_keys=["X_nonexistent"],
        label_key="cell_type",
    )
    assert res_bad_emb["status"] == "error"
    assert res_bad_emb["error_type"] == "KeyError"
    assert "Embedding keys" in res_bad_emb["error"]


def test_mcp_server_instance_and_tools():
    """Verify MCP server instantiates cleanly and registers all three skills."""
    import asyncio

    async def _check():
        server = create_server("test-server")
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert {"inspect_anndata", "embed_cells", "evaluate_scib"}.issubset(tool_names)

    asyncio.run(_check())

