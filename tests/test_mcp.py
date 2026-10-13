"""Automated unit and integration tests for embpy MCP server and 7 bioinformatics tools."""

from __future__ import annotations

import asyncio
import os
import numpy as np
import pytest
from anndata import AnnData

from embpy.mcp import (
    compute_umap,
    create_server,
    embed_cells,
    evaluate_perturbations,
    evaluate_scib,
    inspect_anndata,
    list_available_models,
    map_gene_identifiers,
)


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
    ad.obs["perturbation"] = (["ctrl", "ctrl", "pert_1", "pert_1"] * 10)[:n_obs]
    ad.obs["split"] = (["train", "test"] * 20)[:n_obs]


    ad.layers["counts"] = X.copy()
    ad.obsm["X_prior"] = rng.randn(n_obs, 8).astype(np.float32)

    file_path = str(tmp_path / "sample_test_data.h5ad")
    ad.write_h5ad(file_path)
    return file_path


def test_inspect_anndata_success(sample_adata_path):
    """Test inspect_anndata accurately reads dimensions, slots, and expression nature."""
    res = inspect_anndata(sample_adata_path)
    assert res["status"] == "success"
    assert res["n_obs"] == 30
    assert res["n_vars"] == 40
    assert "cell_type" in res["obs_columns"]
    assert "batch" in res["obs_columns"]
    assert "counts" in res["layers"]
    assert "X_prior" in res["obsm_keys"]
    assert "expression_nature" in res
    assert res["expression_nature"]["detected_nature"] in ("raw_counts", "continuous_normalized", "unknown")
    assert "cell_type" in res["obs_summary"]
    assert res["obs_summary"]["cell_type"]["n_unique"] == 3


def test_inspect_anndata_missing_file():
    """Test inspect_anndata returns actionable error when file is missing."""
    res = inspect_anndata("non_existent_dataset.h5ad")
    assert res["status"] == "error"
    assert res["error_type"] == "FileNotFoundError"
    assert "not found" in res["error"].lower()


def test_list_available_models_success():
    """Test list_available_models returns hardware details and runnable models."""
    res = list_available_models()
    assert res["status"] == "success"
    assert res["compute_device"] in ("cpu", "cuda", "mps")
    assert "pca" in res["runnable_models"]
    assert "scgpt" in res["all_singlecell_models"]
    assert "optional_extras_status" in res
    assert "mcp" in res["optional_extras_status"]
    assert res["optional_extras_status"]["mcp"]["installed"] is True


def test_map_gene_identifiers_symbols_to_ensembl():
    """Test map_gene_identifiers converts HGNC gene symbols to Ensembl IDs."""
    res = map_gene_identifiers(genes=["TP53", "GAPDH"], target_type="ensembl", organism="human")
    assert res["status"] == "success"
    assert res["total_genes"] == 2
    assert res["mapped_count"] >= 1
    assert "TP53" in res["mapping_sample"]
    assert res["mapping_sample"]["TP53"].startswith("ENSG")


def test_map_gene_identifiers_ensembl_to_symbols():
    """Test map_gene_identifiers reverse converts Ensembl IDs to symbols."""
    res = map_gene_identifiers(
        genes=["ENSG00000141510", "ENSG00000111640"],
        target_type="symbol",
        organism="human",
    )
    assert res["status"] == "success"
    assert res["total_genes"] == 2
    assert res["mapped_count"] >= 1
    assert "ENSG00000141510" in res["mapping_sample"]
    assert res["mapping_sample"]["ENSG00000141510"] == "TP53"


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


def test_compute_umap_success(sample_adata_path, tmp_path):
    """Test compute_umap computes neighbor graph and 2D UMAP coordinates."""
    # First generate PCA embedding
    emb_path = str(tmp_path / "for_umap.h5ad")
    embed_cells(
        path=sample_adata_path,
        models=["pca"],
        output_path=emb_path,
        preprocessing="none",
        device="cpu",
    )

    umap_path = str(tmp_path / "with_umap.h5ad")
    res = compute_umap(
        path=emb_path,
        embedding_key="X_pca",
        output_path=umap_path,
        n_neighbors=5,
    )
    assert res["status"] == "success"
    assert res["n_cells"] == 30
    assert res["umap_key"] == "X_umap_pca"
    assert res["umap_shape"] == [30, 2]
    assert os.path.exists(umap_path)


def test_evaluate_scib_success(sample_adata_path, tmp_path):
    """Test evaluate_scib evaluates an existing embedding against annotations."""
    embedded_path = str(tmp_path / "for_scib.h5ad")
    embed_cells(
        path=sample_adata_path,
        models=["pca"],
        output_path=embedded_path,
        preprocessing="none",
        device="cpu",
    )

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
    assert "markdown_table" in eval_res
    assert os.path.exists(csv_path)

    metrics_pca = eval_res["metrics"]["X_pca"]
    assert "bio_conservation" in metrics_pca
    assert "total" in metrics_pca
    assert len(eval_res["summary"]) == 1


def test_evaluate_scib_missing_keys(sample_adata_path):
    """Test evaluate_scib catches missing obs labels or obsm embedding keys."""
    res_bad_label = evaluate_scib(
        path=sample_adata_path,
        embedding_keys=["X_prior"],
        label_key="missing_label_col",
    )
    assert res_bad_label["status"] == "error"
    assert res_bad_label["error_type"] == "KeyError"

    res_bad_emb = evaluate_scib(
        path=sample_adata_path,
        embedding_keys=["X_nonexistent"],
        label_key="cell_type",
    )
    assert res_bad_emb["status"] == "error"
    assert res_bad_emb["error_type"] == "KeyError"


def test_evaluate_perturbations_success(sample_adata_path, tmp_path):
    """Test evaluate_perturbations evaluates predictions against reference controls."""
    csv_path = str(tmp_path / "pert_scores.csv")
    res = evaluate_perturbations(
        path=sample_adata_path,
        split_col="split",
        control_pert="ctrl",
        pert_col="perturbation",
        profile="minimal",
        output_csv=csv_path,
    )
    assert res["status"] == "success"
    assert res["control_pert"] == "ctrl"
    assert "markdown_table" in res
    assert res["n_perturbations"] >= 1
    assert os.path.exists(csv_path)


def test_mcp_server_instance_and_tools():
    """Verify MCP server instantiates cleanly and registers all 7 bioinformatics skills."""
    async def _check():
        server = create_server("test-server")
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "inspect_anndata",
            "list_available_models",
            "map_gene_identifiers",
            "embed_cells",
            "evaluate_scib",
            "evaluate_perturbations",
            "compute_umap",
        }
        assert expected.issubset(tool_names)

    asyncio.run(_check())
