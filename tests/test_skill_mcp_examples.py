"""Run the MCP Skills' tool calls offline, exactly as they are written in SKILL.md.

tests/test_skills.py checks these calls statically (tool names, parameters,
models, annotation types). Here every call of the single-cell, entity-embedding
and identifier/annotation Skills is executed, in the order the Skill gives, on
small generated files in a temporary ``data/`` folder. Web services are replaced
by recorded answers and any other HTTP request fails the test, so the check is
deterministic and runs in the required MCP job.

Calls that need a model download (``NEEDS_DOWNLOAD``) run only in the
skill-examples job (``EMBPY_RUN_SKILL_EXAMPLES=1``).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import embpy.mcp.tools as mcp_tools

ROOT = Path(__file__).resolve().parents[1]
GENE_FIXTURES = ROOT / "tests" / "embpy" / "fixtures" / "gene"
TOOLS = {
    name: getattr(mcp_tools, name)
    for name in (
        "inspect_data", "resolve_identifiers", "preprocess_counts", "list_model_registry",
        "compute_embeddings", "annotate_metadata", "compute_umap", "benchmark_scib",
        "benchmark_perturbations",
    )
}
# (skill, model) pairs whose weights or tables come from the Hugging Face Hub.
NEEDS_DOWNLOAD = {
    ("embpy-entity-embeddings", "gene2vec"),
    ("embpy-entity-embeddings", "esm2_8M"),
}
RUN_DOWNLOADS = os.environ.get("EMBPY_RUN_SKILL_EXAMPLES") == "1"
OK = ("success", "partial_success")


def _require(*modules: str) -> None:
    """Skip without the optional packages, except in the MCP job (EMBPY_REQUIRE_MCP=1), where they must exist."""
    for module in modules:
        if os.environ.get("EMBPY_REQUIRE_MCP") == "1":
            __import__(module)
        else:
            pytest.importorskip(module, reason=f"{module} not installed")


def skill_calls(skill: str) -> list[tuple[str, dict]]:
    """The tool calls in the Skill's ```python blocks, in order, with literal arguments."""
    text = (ROOT / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    calls = []
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in TOOLS:
                calls.append((node.func.id, {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}))
    assert calls, f"{skill} shows no tool calls"
    return calls


def _local(kwargs: dict, root: Path) -> dict:
    """Point the Skill's data/... paths into the test folder."""
    out = dict(kwargs)
    for key in ("path", "path_pred", "output_path", "output_csv", "output_dir"):
        value = out.get(key)
        if isinstance(value, str) and value.startswith("data/"):
            out[key] = str(root / value)
    return out


def _models(kwargs: dict) -> list[str]:
    models = kwargs.get("models", "pca")
    return [models] if isinstance(models, str) else list(models)


# ---------------------------------------------------------------- test data
def _counts(n_cells: int, n_genes: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, size=n_genes)
    blocks, types = [], []
    for t in range(3):
        lam = base.copy()
        lam[t * 20:(t + 1) * 20] *= 6
        blocks.append(rng.poisson(lam, size=(n_cells // 3, n_genes)))
        types += [f"type_{t}"] * (n_cells // 3)
    return np.vstack(blocks).astype(np.float32), types


def _write_cells(path: Path) -> None:
    X, types = _counts(150, 400)
    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
    adata.obs["cell_type"] = pd.Categorical(types)
    adata.obs["batch"] = pd.Categorical(np.tile(["b1", "b2"], adata.n_obs // 2))
    adata.write_h5ad(path)


def _perturbation_cells(n_cells: int, seed: int) -> AnnData:
    X, _ = _counts(n_cells, 60, seed=seed)
    adata = AnnData(X=np.log1p(X))
    adata.obs_names = [f"c{seed}_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
    adata.obs["perturbation"] = pd.Categorical(np.tile(["ctrl", "pert_A", "pert_B"], adata.n_obs // 3))
    return adata


def _write_perturbation_data(folder: Path) -> None:
    both = _perturbation_cells(180, seed=1)
    both.obs["split"] = pd.Categorical(np.tile(["predicted", "observed"], both.n_obs // 2))
    both.write_h5ad(folder / "norman.h5ad")
    _perturbation_cells(90, seed=2).write_h5ad(folder / "observed.h5ad")
    _perturbation_cells(90, seed=3).write_h5ad(folder / "predicted.h5ad")


def _write_entities(folder: Path) -> None:
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]
    AnnData(obs=pd.DataFrame({"smiles": smiles}, index=[f"w{i}" for i in range(4)])).write_h5ad(
        folder / "compounds.h5ad"
    )
    screen = AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame({"target_gene": ["TP53", "EGFR", "TP53", "MTOR"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame({"gene_symbol": ["Trp53", "Brca1", "Gapdh"]}, index=["g0", "g1", "g2"]),
    )
    screen.write_h5ad(folder / "screen.h5ad")
    AnnData(obs=pd.DataFrame({"gene_symbol": ["TP53", "EGFR"]}, index=["p0", "p1"])).write_h5ad(
        folder / "proteins.h5ad"
    )
    genes = AnnData(X=np.ones((2, 3), dtype=np.float32))
    genes.var_names = ["TP53", "EGFR", "MTOR"]
    genes.write_h5ad(folder / "genes.h5ad")


@pytest.fixture
def data_root(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    _write_cells(folder / "pbmc.h5ad")
    _write_perturbation_data(folder)
    _write_entities(folder)
    return tmp_path


# ---------------------------------------------------------------- recorded services
ENSEMBL = {
    "human": {"TP53": "ENSG00000141510", "P53": "ENSG00000141510", "BRCA1": "ENSG00000012048",
              "EGFR": "ENSG00000146648", "MTOR": "ENSG00000198793"},
    "mouse": {"Trp53": "ENSMUSG00000059552", "Brca1": "ENSMUSG00000017146", "Gapdh": "ENSMUSG00000057666"},
}
SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}


@pytest.fixture
def offline_services(monkeypatch):
    """Recorded answers for the services the Skills use; any other request fails."""
    import requests

    from embpy.resources.gene import annotator as gene_annotator
    from embpy.resources.gene import reactome
    from embpy.resources.gene.resolver import GeneResolver
    from embpy.resources.molecule.annotator import MoleculeAnnotator
    from embpy.resources.molecule.resolver import DrugResolver
    from embpy.resources.protein.annotator import ProteinAnnotator

    def no_network(self, method, url, *args, **kwargs):
        raise AssertionError(f"network access in an offline Skill example: {method} {url}")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)

    def symbols_to_ensembl_batch(self, symbols, organism="human", return_diagnostics=False, **kwargs):
        table = ENSEMBL[organism]
        mapping = {s: table.get(s) for s in symbols}
        found = sum(v is not None for v in mapping.values())
        diag = {"status": "success" if found == len(symbols) else "partial_success", "mapped_count": found,
                "unresolved_count": len(symbols) - found, "total": len(symbols),
                "mapping_rate": 100.0 * found / max(len(symbols), 1)}
        return (mapping, diag) if return_diagnostics else mapping

    monkeypatch.setattr(GeneResolver, "symbols_to_ensembl_batch", symbols_to_ensembl_batch)
    monkeypatch.setattr(DrugResolver, "name_to_smiles", lambda self, name, *a, **k: SMILES.get(name.lower()))

    rows = json.loads((GENE_FIXTURES / "mygene_query_post_20.json").read_text(encoding="utf-8"))

    def mygene_batch(url, json_data=None, data=None, timeout=30):
        wanted = data["q"].split(",")
        return [row for q in wanted for row in rows if row["query"] == q]

    monkeypatch.setattr(gene_annotator, "_post_json", mygene_batch)
    index = reactome.ReactomeIndex.from_dir(GENE_FIXTURES / "reactome98", release="98")
    monkeypatch.setattr(reactome, "load_reactome_index", lambda *a, **k: index)

    def uniprot_annotate(self, identifier, id_type="auto", sources="all"):
        return {"identifier": identifier, "organism": "Homo sapiens", "taxonomy_id": 9606,
                "go_terms": {"biological_process": [{"id": "GO:0006915", "name": "apoptotic process"}]}}

    monkeypatch.setattr(ProteinAnnotator, "annotate", uniprot_annotate)

    def bioactivities(self, smiles, *args, **kwargs):
        if smiles != SMILES["aspirin"]:
            return []
        return [{"target_chembl_id": "CHEMBL221", "standard_type": "IC50", "standard_value": 1.0}]

    monkeypatch.setattr(MoleculeAnnotator, "get_bioactivities", bioactivities)
    monkeypatch.setattr(
        MoleculeAnnotator, "get_target_proteins",
        lambda self, smiles: [{"target": "PTGS1"}] if smiles == SMILES["aspirin"] else [],
    )
    monkeypatch.setattr(MoleculeAnnotator, "get_mechanism_of_action", lambda self, smiles: [])


# ---------------------------------------------------------------- the Skills
def test_single_cell_skill_runs_offline(data_root, offline_services):
    """inspect -> preprocess -> registry -> embed -> UMAP -> scIB -> perturbation benchmarks."""
    _require("scanpy", "torch", "scib", "cell_eval")
    has_scvi = importlib.util.find_spec("scvi") is not None
    for name, kwargs in skill_calls("embpy-single-cell-workflow"):
        kwargs = _local(kwargs, data_root)
        if name in ("compute_embeddings", "compute_umap", "benchmark_scib"):
            # The Skill names the output of the previous step: it must exist.
            assert os.path.exists(kwargs["path"]), (name, kwargs["path"])
        if name == "benchmark_scib":
            # The Skill tells the agent to benchmark only the embeddings that were computed.
            import anndata

            present = set(anndata.read_h5ad(kwargs["path"]).obsm)
            kwargs["embedding_keys"] = [k for k in kwargs["embedding_keys"] if k in present]
        res = TOOLS[name](**kwargs)
        assert res["status"] in OK, (name, kwargs, res)
        if name == "compute_embeddings":
            assert "pca" in res["models_computed"], res
            if not has_scvi:
                assert res["status"] == "partial_success" and "scvi" in res["models_failed"], res


def test_entity_embedding_skill_runs_offline(data_root, offline_services):
    _require("torch", "rdkit")
    ran = []
    for name, kwargs in skill_calls("embpy-entity-embeddings"):
        if name == "compute_embeddings" and any(("embpy-entity-embeddings", m) in NEEDS_DOWNLOAD for m in _models(kwargs)):
            continue  # needs a model download: covered by test_entity_embedding_downloads
        res = TOOLS[name](**_local(kwargs, data_root))
        assert res["status"] in OK, (name, kwargs, res)
        ran.append((name, _models(kwargs) if name == "compute_embeddings" else None))
    assert ("compute_embeddings", ["morgan_fp"]) in ran


@pytest.mark.skipif(not RUN_DOWNLOADS, reason="set EMBPY_RUN_SKILL_EXAMPLES=1 (downloads models)")
def test_entity_embedding_downloads(data_root):
    for name, kwargs in skill_calls("embpy-entity-embeddings"):
        if name == "compute_embeddings" and any(("embpy-entity-embeddings", m) in NEEDS_DOWNLOAD for m in _models(kwargs)):
            res = TOOLS[name](**_local(kwargs, data_root))
            assert res["status"] in OK, (name, kwargs, res)


def test_identifier_and_annotation_skill_runs_offline(data_root, offline_services):
    _require("rdkit")
    results = []
    for name, kwargs in skill_calls("embpy-identifiers-and-annotation"):
        res = TOOLS[name](**_local(kwargs, data_root))
        assert res["status"] in OK, (name, kwargs, res)
        results.append((name, kwargs, res))

    first = results[0][2]
    assert first["sample_mapping"]["TP53"] == "ENSG00000141510"
    tp53 = next(r for n, k, r in results if k.get("identifiers") == ["TP53"])
    pathways = tp53["records_preview"]["TP53"]["pathways"]["reactome"]
    assert any(item["id"] == "R-HSA-111448" for item in pathways["items"])
