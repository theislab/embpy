"""Unit and integration tests for the embpy MCP tools (the functions behind the MCP server).

Calls through a real MCP client over stdio live in ``test_mcp_client.py``.
"""

from __future__ import annotations

import asyncio
import os
import numpy as np
import pytest
from anndata import AnnData

from unittest import mock

from embpy.mcp import (
    annotate_metadata,
    benchmark_perturbations,
    benchmark_scib,
    compute_embeddings,
    compute_umap,
    create_server,
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


def _require_mcp():
    """Skip without the mcp package, unless EMBPY_REQUIRE_MCP=1 (set by the MCP CI job)."""
    if os.environ.get("EMBPY_REQUIRE_MCP") == "1":
        import mcp  # noqa: F401
    else:
        pytest.importorskip("mcp", reason="mcp not installed")


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
    assert isinstance(res["optional_extras_status"]["mcp"]["installed"], bool)


def test_map_gene_identifiers_symbols_to_ensembl(monkeypatch):
    """Test map_gene_identifiers converts HGNC gene symbols to Ensembl IDs."""
    from embpy.resources.gene.resolver import GeneResolver

    monkeypatch.setattr(
        GeneResolver,
        "symbols_to_ensembl_batch",
        lambda self, symbols, organism="human", **kwargs: (
            {"TP53": "ENSG00000141510", "GAPDH": "ENSG00000111640"},
            {"status": "success", "mapped_count": 2, "unresolved_count": 0, "mapping_rate": 100.0, "total": 2},
        )
        if kwargs.get("return_diagnostics")
        else {"TP53": "ENSG00000141510", "GAPDH": "ENSG00000111640"},
    )
    res = map_gene_identifiers(genes=["TP53", "GAPDH"], target_type="ensembl", organism="human")
    assert res["status"] in ("success", "partial_success")
    assert res["total_genes"] == 2
    assert res["mapped_count"] >= 1
    assert "TP53" in res["mapping_sample"]
    assert res["mapping_sample"]["TP53"].startswith("ENSG")


def test_map_gene_identifiers_ensembl_to_symbols(monkeypatch):
    """Test map_gene_identifiers reverse converts Ensembl IDs to symbols."""
    from embpy.resources.gene.resolver import GeneResolver

    monkeypatch.setattr(
        GeneResolver,
        "ensembl_to_symbols_batch",
        lambda self, ensembl_ids, organism="human", **kwargs: (
            {"ENSG00000141510": "TP53", "ENSG00000111640": "GAPDH"},
            {"status": "success", "mapped_count": 2, "unresolved_count": 0, "mapping_rate": 100.0, "total": 2},
        )
        if kwargs.get("return_diagnostics")
        else {"ENSG00000141510": "TP53", "ENSG00000111640": "GAPDH"},
    )
    res = map_gene_identifiers(
        genes=["ENSG00000141510", "ENSG00000111640"],
        target_type="symbol",
        organism="human",
    )
    assert res["status"] in ("success", "partial_success")
    assert res["total_genes"] == 2
    assert res["mapped_count"] >= 1
    assert "ENSG00000141510" in res["mapping_sample"]
    assert res["mapping_sample"]["ENSG00000141510"] == "TP53"


def test_map_gene_identifiers_partial_success_fallback(monkeypatch):
    """Test map_gene_identifiers reports partial_success and diagnostics when network fails."""
    from embpy.resources.gene.resolver import GeneResolver

    monkeypatch.setattr(
        GeneResolver,
        "symbols_to_ensembl_batch",
        lambda self, symbols, organism="human", **kwargs: (
            {"TP53": "ENSG00000141510", "UNRESOLVED_GENE": None},
            {
                "status": "partial_success",
                "mapped_count": 1,
                "unresolved_count": 1,
                "mapping_rate": 50.0,
                "warning": "Ensembl REST API timed out on 1 genes.",
                "action_required": "Install embpy[genome] for offline resolution.",
            },
        )
        if kwargs.get("return_diagnostics")
        else {"TP53": "ENSG00000141510", "UNRESOLVED_GENE": None},
    )
    res = map_gene_identifiers(genes=["TP53", "UNRESOLVED_GENE"], target_type="ensembl", organism="human")
    assert res["status"] == "partial_success"
    assert res["mapped_count"] == 1
    assert res["unresolved_count"] == 1
    assert "warning" in res


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
    pytest.importorskip("scib", reason="scib is not installed")
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
    pytest.importorskip("cell_eval", reason="cell-eval is not installed")
    csv_path = str(tmp_path / "pert_scores.csv")
    res = evaluate_perturbations(
        path=sample_adata_path,
        split_col="split",
        pred_label="test",
        ref_label="train",
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
    _require_mcp()

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


def test_gene_conversions_human_mouse_versioned():
    """Test human and mouse gene conversions including versioned IDs."""
    # Human forward
    res_human_fwd = resolve_identifiers(identifiers=["TP53"])
    assert res_human_fwd["status"] in ("success", "partial_success")
    assert res_human_fwd["sample_mapping"].get("TP53") == "ENSG00000141510"

    # Human reverse with versioned ID
    res_human_rev = resolve_identifiers(identifiers=["ENSG00000141510.1"])
    assert res_human_rev["status"] in ("success", "partial_success")
    assert res_human_rev["sample_mapping"].get("ENSG00000141510.1") == "TP53"

    # Mouse explicit forward
    res_mouse_fwd = resolve_identifiers(identifiers=["Trp53"], organism="mouse")
    assert res_mouse_fwd["status"] in ("success", "partial_success")
    assert res_mouse_fwd["sample_mapping"].get("Trp53") == "ENSMUSG00000059552"

    # Mouse automatic reverse with versioned ID (auto-recognizes ENSMUSG)
    res_mouse_rev = resolve_identifiers(identifiers=["ENSMUSG00000059552.2"])
    assert res_mouse_rev["status"] in ("success", "partial_success")
    assert res_mouse_rev["sample_mapping"].get("ENSMUSG00000059552.2") == "Trp53"


def test_duplicate_blank_unknown_gene_resolution():
    """Test pipeline gracefully handles duplicate names, blank inputs, and unknown genes."""
    raw_inputs = ["TP53", "TP53", "", "   ", "UNKNOWN_NONEXISTENT_XYZ", "BRCA1"]
    with mock.patch("embpy.resources.gene.resolver._ensembl_post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp
        res = resolve_identifiers(identifiers=raw_inputs)
    assert res["status"] in ("success", "partial_success")
    mapping = res["sample_mapping"]
    assert mapping.get("TP53") == "ENSG00000141510"
    assert mapping.get("BRCA1") == "ENSG00000012048"
    assert "UNKNOWN_NONEXISTENT_XYZ" in res["unmapped_sample"] or res["unmapped_count"] >= 1


def test_row_aligned_mappings_preserved(tmp_path):
    """Test saving gene mappings preserves exact row alignment, order, duplicates, and missing values."""
    import pandas as pd

    # AnnData with var_names distinct from symbol column
    n_obs, n_vars = 10, 4
    rng = np.random.RandomState(42)
    X = rng.poisson(lam=2.0, size=(n_obs, n_vars)).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"feature_{i}" for i in range(n_vars)]
    # Row 0: TP53, Row 1: TP53 (duplicate), Row 2: unknown/missing, Row 3: BRCA1
    adata.var["symbol"] = ["TP53", "TP53", "UNKNOWN_GENE", "BRCA1"]
    adata.obs["smiles"] = ["CC(=O)Oc1ccccc1C(=O)O"] * n_obs

    in_path = str(tmp_path / "aligned_input.h5ad")
    adata.write_h5ad(in_path)

    out_path = str(tmp_path / "aligned_output.h5ad")
    with mock.patch("embpy.resources.gene.resolver._ensembl_post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp
        res = resolve_identifiers(
            path=in_path,
            var_column="symbol",
            output_path=out_path,
        )
    assert res["status"] in ("success", "partial_success")

    # Reopen and assert perfect row-alignment preservation
    reopened = AnnData(X=X)
    import anndata
    reopened = anndata.read_h5ad(out_path)

    assert list(reopened.var_names) == ["feature_0", "feature_1", "feature_2", "feature_3"]
    mapped_col = reopened.var["symbol_mapped"]
    assert mapped_col.iloc[0] == "ENSG00000141510"
    assert mapped_col.iloc[1] == "ENSG00000141510"
    assert pd.isna(mapped_col.iloc[2]) or mapped_col.iloc[2] is None
    assert mapped_col.iloc[3] == "ENSG00000012048"

    # Test obs_column resolution
    obs_out = str(tmp_path / "aligned_obs_output.h5ad")
    res_obs = resolve_identifiers(
        path=in_path,
        obs_column="smiles",
        entity_type="molecule",
        output_path=obs_out,
    )
    assert res_obs["status"] in ("success", "partial_success")
    reopened_obs = anndata.read_h5ad(obs_out)
    assert "smiles_mapped" in reopened_obs.obs
    assert len(reopened_obs.obs["smiles_mapped"]) == n_obs


def test_reverse_cache_official_symbol_casing():
    """Test that resolving lowercase user input does not poison reverse cache with incorrect casing."""
    # Resolve lowercase symbol
    res_fwd = resolve_identifiers(identifiers=["tp53"])
    assert res_fwd["status"] in ("success", "partial_success")
    assert res_fwd["sample_mapping"].get("tp53") == "ENSG00000141510"

    # Resolve Ensembl ID back to symbol -> must return canonical official symbol TP53
    res_rev = resolve_identifiers(identifiers=["ENSG00000141510"], target_id_type="symbol")
    assert res_rev["status"] in ("success", "partial_success")
    assert res_rev["sample_mapping"].get("ENSG00000141510") == "TP53"


def test_preprocessing_strictness_and_scale(sample_adata_path, tmp_path):
    """Test that explicit preprocessing thresholds are honored and scale option is functional."""
    # 1. Enforce strict min_genes=200 on 40-gene dataset (auto_adjust=False) -> filters out all cells
    res_strict = preprocess_counts(
        path=sample_adata_path,
        min_genes=200,
        auto_adjust=False,
    )
    assert res_strict["status"] == "success"
    assert res_strict["n_cells"] == 0

    # 2. Test scale=True executes sc.pp.scale and produces scaled layer
    scaled_out = str(tmp_path / "scaled_output.h5ad")
    res_scale = preprocess_counts(
        path=sample_adata_path,
        min_genes=5,
        scale=True,
        auto_adjust=False,
        output_path=scaled_out,
    )
    assert res_scale["status"] == "success"
    assert "scaled" in res_scale["layers"]

    import anndata
    adata_scaled = anndata.read_h5ad(scaled_out)
    assert "scaled" in adata_scaled.layers


def test_ambiguous_benchmark_inputs_rejected(sample_adata_path):
    """Test benchmark_perturbations strictly rejects ambiguous inputs without pred_label and ref_label."""
    res = benchmark_perturbations(
        path=sample_adata_path,
        split_col="split",
        control_pert="ctrl",
        pert_col="perturbation",
    )
    assert res["status"] == "error"
    assert res["error_type"] == "ValueError"
    assert "Ambiguous benchmark inputs" in res["error"]


def test_dynamic_model_registry():
    """Test list_model_registry dynamically populates registered models and avoids unregistered names."""
    res = list_model_registry()
    assert res["status"] == "success"
    assert res["total_models_available"] >= 100

    # molformer_base must exist; invalid molformer must not exist
    mol_models = res["recommended_models"]["molecule"]
    assert "molformer_base" in mol_models
    assert "molformer" not in mol_models


# ---------------------------------------------------------------------------
# 1. The requested species is respected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["EGFR", "Egfr", "egfr"])
def test_explicit_organism_is_not_overridden_by_capitalization(symbol):
    res = resolve_identifiers(identifiers=[symbol], organism="human", target_id_type="ensembl")
    assert res["status"] == "success"
    assert res["sample_mapping"][symbol] == "ENSG00000146648"
    assert res["organism_used"] == {"human": 1}
    if symbol != "EGFR":
        assert "case-insensitive" in res["notes"][symbol]


def test_explicit_mouse_organism_resolves_mouse_gene():
    res = resolve_identifiers(identifiers=["EGFR"], organism="mouse", target_id_type="ensembl")
    assert res["sample_mapping"]["EGFR"] == "ENSMUSG00000020122"
    assert res["organism_used"] == {"mouse": 1}


def test_ensembl_id_from_other_species_is_an_error():
    res = resolve_identifiers(identifiers=["ENSMUSG00000059552"], organism="human")
    assert res["status"] == "error"
    assert res["error_type"] == "SpeciesMismatchError"
    assert "ENSMUSG00000059552" in res["conflicting_identifiers"]


def test_species_detection_is_opt_in_and_reported():
    # Default: symbols are looked up in human whatever their case.
    default = resolve_identifiers(identifiers=["Trp53"], target_id_type="ensembl")
    assert default["organism_used"] == {"human": 1}
    assert default["sample_mapping"].get("Trp53") != "ENSMUSG00000059552"

    auto = resolve_identifiers(identifiers=["Trp53", "TP53"], organism="auto")
    assert auto["status"] == "success"
    assert auto["sample_mapping"]["Trp53"] == "ENSMUSG00000059552"
    assert auto["sample_mapping"]["TP53"] == "ENSG00000141510"
    assert auto["organism_used"] == {"mouse": 1, "human": 1}


# ---------------------------------------------------------------------------
# 2. Annotation success means the requested annotations were retrieved
# ---------------------------------------------------------------------------


@pytest.fixture
def gene_pert_path(tmp_path):
    ad = AnnData(X=np.ones((3, 2), dtype=np.float32))
    ad.obs_names = ["c0", "c1", "c2"]
    ad.obs["perturbation"] = ["TP53", "GAPDH", "TP53"]
    p = str(tmp_path / "genes.h5ad")
    ad.write_h5ad(p)
    return p


def _fake_annotators(monkeypatch, *, go=None, pathways=None, fail=False):
    """Patch the gene/protein annotators; record the organisms and calls they receive."""
    import requests

    from embpy.resources._service_errors import report_service_error
    from embpy.resources.gene.annotator import GeneAnnotator
    from embpy.resources.protein.annotator import ProteinAnnotator

    seen = {"organisms": [], "calls": []}
    orig_gene_init, orig_prot_init = GeneAnnotator.__init__, ProteinAnnotator.__init__

    def gene_init(self, organism="human", rate_limit_delay=0.2):
        seen["organisms"].append(("gene", organism))
        orig_gene_init(self, organism=organism, rate_limit_delay=0)

    def prot_init(self, organism="human", rate_limit_delay=0.2):
        seen["organisms"].append(("protein", organism))
        orig_prot_init(self, organism=organism, rate_limit_delay=0)

    def get_pathways(self, gene):
        seen["calls"].append(("pathways", gene))
        if fail:
            report_service_error("https://mygene.info/v3/query", requests.ConnectionError("down"))
        return (pathways or {}).get(gene, {"reactome": [], "kegg": [], "wikipathways": []})

    def annotate(self, identifier, id_type="auto", sources="all"):
        seen["calls"].append(("uniprot:" + ",".join(sources), identifier, id_type))
        if fail:
            report_service_error("https://rest.uniprot.org", requests.Timeout("down"))
            return {"identifier": identifier, "error": "Could not fetch UniProt entry"}
        return {"identifier": identifier, "go_terms": (go or {}).get(identifier, {})}

    monkeypatch.setattr(GeneAnnotator, "__init__", gene_init)
    monkeypatch.setattr(ProteinAnnotator, "__init__", prot_init)
    monkeypatch.setattr(GeneAnnotator, "get_pathways", get_pathways)
    monkeypatch.setattr(ProteinAnnotator, "annotate", annotate)
    return seen


def test_annotation_runs_every_requested_type_for_the_requested_organism(monkeypatch, gene_pert_path, tmp_path):
    go = {
        "TP53": {"biological_process": [{"id": "GO:0006915"}]},
        "GAPDH": {"molecular_function": [{"id": "GO:0004365"}]},
    }
    pathways = {"TP53": {"reactome": [{"id": "R-1"}], "kegg": [], "wikipathways": []}}
    seen = _fake_annotators(monkeypatch, go=go, pathways=pathways)
    out = str(tmp_path / "annotated.h5ad")
    res = annotate_metadata(
        path=gene_pert_path,
        entity_type="gene",
        annotation_types=["go", "pathways"],
        column="perturbation",
        organism="mouse",
        output_path=out,
    )
    assert ("gene", "mouse") in seen["organisms"]
    assert ("protein", "mouse") in seen["organisms"]
    # GO goes to the GO operation, with symbol ids; both requested types ran.
    assert ("uniprot:go", "TP53", "symbol") in seen["calls"]
    assert ("pathways", "GAPDH") in seen["calls"]
    assert res["status"] == "partial_success"  # GAPDH has no pathways
    assert res["per_type"]["go"] == {"annotated": 2, "no_match": 0, "service_error": 0}
    assert res["per_type"]["pathways"] == {"annotated": 1, "no_match": 1, "service_error": 0}
    import anndata

    saved = anndata.read_h5ad(out)
    assert list(saved.obs["gene_go_n"]) == [1, 1, 1]
    assert list(saved.obs["gene_pathways_n"]) == [1, 0, 1]


def test_annotation_sources_are_honored(monkeypatch, gene_pert_path, tmp_path):
    pathways = {g: {"reactome": [], "kegg": [{"id": "hsa1"}], "wikipathways": []} for g in ("TP53", "GAPDH")}
    _fake_annotators(monkeypatch, pathways=pathways)
    res = annotate_metadata(
        path=gene_pert_path,
        entity_type="gene",
        annotation_types="pathways",
        sources=["reactome"],
        column="perturbation",
        output_path=str(tmp_path / "o.h5ad"),
    )
    # Only Reactome was asked for, and it had nothing: KEGG hits must not count.
    assert res["sources_queried"] == {"pathways": ["reactome"]}
    assert res["status"] == "error"
    assert res["error_type"] == "NoAnnotationsFound"

    bad = annotate_metadata(
        path=gene_pert_path, entity_type="gene", annotation_types="go", sources=["kegg"], column="perturbation"
    )
    assert bad["status"] == "error"
    assert bad["error_type"] == "ValueError"


def test_annotation_empty_responses_are_not_success(monkeypatch, gene_pert_path, tmp_path):
    _fake_annotators(monkeypatch)
    out = str(tmp_path / "should_not_exist.h5ad")
    res = annotate_metadata(
        path=gene_pert_path,
        entity_type="gene",
        annotation_types=["go", "pathways"],
        column="perturbation",
        output_path=out,
    )
    assert res["status"] == "error"
    assert res["error_type"] == "NoAnnotationsFound"
    assert res["per_type"]["go"]["no_match"] == 2
    assert res["successful_count"] == 0
    assert not os.path.exists(out)


def test_annotation_service_failures_are_reported_as_failures(monkeypatch, gene_pert_path, tmp_path):
    _fake_annotators(monkeypatch, fail=True)
    out = str(tmp_path / "should_not_exist.h5ad")
    res = annotate_metadata(
        path=gene_pert_path,
        entity_type="gene",
        annotation_types=["go", "pathways"],
        column="perturbation",
        output_path=out,
    )
    assert res["status"] == "error"
    assert res["error_type"] == "AnnotationServiceError"
    assert res["per_type"]["go"]["service_error"] == 2
    assert res["per_type"]["pathways"]["service_error"] == 2
    assert res["service_errors_sample"]
    assert not os.path.exists(out)


def test_annotation_rejects_unsupported_requests(gene_pert_path):
    res = annotate_metadata(
        path=gene_pert_path, entity_type="gene", annotation_types="diseases", organism="mouse", column="perturbation"
    )
    assert res["status"] == "error"
    assert "human" in res["error"]
    res = annotate_metadata(path=gene_pert_path, entity_type="molecule", annotation_types="go", column="perturbation")
    assert res["status"] == "error"
    assert "not available" in res["error"]


# ---------------------------------------------------------------------------
# 3. Embedding inputs can come from a chosen column
# ---------------------------------------------------------------------------


def test_molecule_embedding_from_obs_column_matches_python_api(tmp_path):
    import anndata
    import pandas as pd

    from embpy import BioEmbedder

    adata = AnnData(obs=pd.DataFrame({"smiles": ["CCO", "CCN"]}, index=["sample_A", "sample_B"]))
    in_path = str(tmp_path / "molecules.h5ad")
    adata.write_h5ad(in_path)
    out_path = str(tmp_path / "molecules_embedded.h5ad")

    res = compute_embeddings(
        path=in_path,
        models="morgan_fp",
        entity_type="molecule",
        obs_column="smiles",
        output_path=out_path,
        device="cpu",
    )
    assert res["status"] == "success", res
    (key,) = res["embedding_keys"]
    assert res["embedding_slots"][key] == "obsm"
    assert res["embedding_shapes"][key] == [2, 2048]

    direct = BioEmbedder(device="cpu").embed(
        anndata.read_h5ad(in_path), entity_type="molecule", model="morgan_fp", obs_column="smiles", output="anndata"
    )
    saved = anndata.read_h5ad(out_path)
    np.testing.assert_array_equal(np.asarray(saved.obsm[key]), np.asarray(direct.obsm[key]))


def test_embedding_input_options_are_in_the_tool_signature():
    import inspect

    params = inspect.signature(compute_embeddings).parameters
    for name in ("obs_column", "var_column", "input_axis", "id_type", "organism"):
        assert name in params


def test_perturbation_embedding_requires_label_column(sample_adata_path):
    res = compute_embeddings(path=sample_adata_path, models="gene2vec", entity_type="perturbation")
    assert res["status"] == "error"
    assert "obs_column" in res["error"]


# ---------------------------------------------------------------------------
# 4. Recommended models run on the modality they are recommended for
# ---------------------------------------------------------------------------


def test_every_recommended_model_is_valid_for_its_modality():
    from embpy.mcp.tools import _EMBED_ROUTES, _RECOMMENDED_MODELS, _validate_embedding_model

    assert set(_RECOMMENDED_MODELS) == set(_EMBED_ROUTES)
    for modality, models in _RECOMMENDED_MODELS.items():
        for model in models:
            assert _validate_embedding_model(modality, model) is None, (modality, model)


def test_every_alias_names_a_real_model():
    from embpy.mcp.tools import MODEL_ALIASES, _model_kind

    for alias, target in MODEL_ALIASES.items():
        assert _model_kind(target) is not None, (alias, target)


def test_registry_lists_only_models_each_path_can_run():
    from embpy.mcp.tools import _validate_embedding_model

    res = list_model_registry()
    for modality, models in res["available_models"].items():
        for model in models:
            assert _validate_embedding_model(modality, model) is None, (modality, model)
    assert "boltz2" not in res["available_models"]["morphology"]
    assert "state" not in res["available_models"]["perturbation"]


@pytest.mark.parametrize(
    ("entity_type", "model", "message"),
    [
        ("text", "bge_large", "Unknown model"),
        ("morphology", "structure", "Unknown model"),
        ("morphology", "boltz2", "protein model"),
        ("perturbation", "state", "single_cell model"),
        ("single_cell", "esm2_8M", "protein model"),
    ],
)
def test_model_modality_mismatches_are_rejected_before_running(sample_adata_path, entity_type, model, message):
    res = compute_embeddings(path=sample_adata_path, models=model, entity_type=entity_type, obs_column="perturbation")
    assert res["status"] == "error"
    assert res["error_type"] == "UnsupportedModelError"
    assert message in res["error"]


# ---------------------------------------------------------------------------
# 5. Conversion requests are validated and status follows the final mappings
# ---------------------------------------------------------------------------


def test_unsupported_target_types_are_rejected():
    res = resolve_identifiers(identifiers=["TP53"], target_id_type="banana")
    assert res["status"] == "error"
    assert "banana" in res["error"]
    res = resolve_identifiers(identifiers=["TP53"], entity_type="gene", target_id_type="smiles")
    assert res["status"] == "error"
    res = resolve_identifiers(identifiers=["CCO"], entity_type="molecule", target_id_type="ensembl")
    assert res["status"] == "error"


def test_blank_identifiers_count_and_fail_strict_mode():
    res = resolve_identifiers(identifiers=["TP53", "  "], strict=True)
    assert res["status"] == "error"
    assert res["blank_count"] == 1
    assert res["error_type"] == "UnresolvedIdentifiersError"

    lenient = resolve_identifiers(identifiers=["TP53", ""])
    assert lenient["status"] == "partial_success"
    assert lenient["total_submitted"] == 2
    assert lenient["mapped_count"] == 1


def test_status_follows_final_mappings(monkeypatch):
    from embpy.resources.gene.resolver import GeneResolver

    def fake_batch(self, symbols, organism="human", **kwargs):
        res = {s: None for s in symbols}
        diag = {"status": "partial_success", "unresolved_count": len(symbols), "warning": "Ensembl timed out."}
        return (res, diag) if kwargs.get("return_diagnostics") else res

    monkeypatch.setattr(GeneResolver, "symbols_to_ensembl_batch", fake_batch)
    res = resolve_identifiers(identifiers=["NOT_A_GENE_1", "NOT_A_GENE_2"])
    assert res["status"] == "error"
    assert res["mapped_count"] == 0


def test_molecule_conversion_reports_smiles_target():
    res = resolve_identifiers(identifiers=["OCC"], entity_type="molecule")
    assert res["status"] == "success"
    assert res["target_id_type"] == "smiles"
    assert res["sample_mapping"]["OCC"] == "CCO"


def test_strict_mode_does_not_write_output(tmp_path):
    adata = AnnData(X=np.ones((1, 2), dtype=np.float32))
    adata.var_names = ["f0", "f1"]
    adata.var["symbol"] = ["TP53", ""]
    in_path = str(tmp_path / "in.h5ad")
    adata.write_h5ad(in_path)
    out_path = str(tmp_path / "out.h5ad")
    res = resolve_identifiers(path=in_path, var_column="symbol", strict=True, output_path=out_path)
    assert res["status"] == "error"
    assert not os.path.exists(out_path)


# ---------------------------------------------------------------------------
# 6. Prediction and reference groups must differ
# ---------------------------------------------------------------------------


def test_identical_prediction_and_reference_labels_are_rejected(sample_adata_path):
    res = benchmark_perturbations(
        path=sample_adata_path,
        split_col="split",
        pred_label="test",
        ref_label="test",
        control_pert="ctrl",
        pert_col="perturbation",
    )
    assert res["status"] == "error"
    assert "themselves" in res["error"]


def test_same_prediction_and_reference_data_is_rejected(sample_adata_path, tmp_path):
    import shutil

    res = benchmark_perturbations(
        path=sample_adata_path, path_pred=sample_adata_path, control_pert="ctrl", pert_col="perturbation"
    )
    assert res["status"] == "error"
    assert "same file" in res["error"]

    copy_path = str(tmp_path / "copy.h5ad")
    shutil.copy(sample_adata_path, copy_path)
    res = benchmark_perturbations(
        path=sample_adata_path, path_pred=copy_path, control_pert="ctrl", pert_col="perturbation"
    )
    assert res["status"] == "error"
    assert "same cells" in res["error"]


def test_auto_preprocessing_embeds_small_datasets(sample_adata_path, tmp_path):
    """embed_cells' default QC (min_genes=200) would drop every cell of a 40-gene dataset."""
    res = compute_embeddings(path=sample_adata_path, models="pca", device="cpu",
                             output_path=str(tmp_path / "small_embedded.h5ad"))
    assert res["status"] == "success", res
    assert res["embedding_shapes"]["X_pca"][0] == 30
    assert "min_genes" in res["preprocessing_adjustments"]
