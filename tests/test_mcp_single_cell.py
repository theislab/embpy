"""Single-cell steps of the embpy MCP tools: what each step keeps, changes and reports.

Covers which inputs ``preprocess_counts`` refuses, which cells and genes it and
``compute_embeddings`` remove, which genes PCA is fitted on, which keys ``compute_umap``
writes, and how rows without a finite embedding are handled.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from embpy.mcp import benchmark_scib, compute_embeddings, compute_umap, preprocess_counts

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@pytest.fixture
def pca_genes(monkeypatch):
    """The genes of every PCA fit in the test, read from the fitted wrapper."""
    from embpy.models.singlecell_models import PCAEmbedding

    fits: list[list[str]] = []
    real_embed_cells = PCAEmbedding.embed_cells

    def embed_cells(self, adata):
        result = real_embed_cells(self, adata)
        genes = list(self._var_names)
        if self._hvg_mask is not None:
            genes = [gene for gene, used in zip(genes, self._hvg_mask, strict=True) if used]
        # One fitted component weight per input gene.
        assert self._pca.components_.shape[1] == len(genes)
        fits.append(genes)
        return result

    monkeypatch.setattr(PCAEmbedding, "embed_cells", embed_cells)
    return fits


@pytest.fixture
def fast_scvi(monkeypatch):
    """scVI returns a fixed latent space instead of training, whether or not scvi-tools is installed."""
    from embpy.models.singlecell_models import ScVIToolsWrapper

    monkeypatch.setattr(
        ScVIToolsWrapper, "embed_cells", lambda self, adata: np.zeros((adata.n_obs, 10), dtype=np.float32)
    )


def _cells(x: np.ndarray, n_types: int = 3) -> AnnData:
    adata = AnnData(X=x.astype(np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene_{j}" for j in range(adata.n_vars)]
    adata.obs["cell_type"] = [f"type_{i % n_types}" for i in range(adata.n_obs)]
    adata.obs["batch"] = [f"b{i % 2}" for i in range(adata.n_obs)]
    return adata


def _raw_with_shallow_cells() -> AnnData:
    """150 cells x 600 genes of raw counts. Cells 75-149 detect exactly 150 genes and the
    last 4 genes are detected in one cell each, so min_genes=100 keeps every cell while the
    default min_genes=200 removes half of them."""
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_rare = 150, 600, 4
    lam = rng.gamma(2.0, 1.5, size=n_genes) + 0.5
    programs = np.ones((3, n_genes))
    for t in range(3):
        programs[t, t * 40 : (t + 1) * 40] = 6.0
    x = rng.poisson(lam * programs[np.arange(n_cells) % 3]).astype(np.float32)
    for i in range(75, n_cells):
        detected = rng.choice(n_genes - n_rare, size=150, replace=False)
        row = np.zeros(n_genes, dtype=np.float32)
        row[detected] = x[i, detected] + 1
        x[i] = row
    x[:, n_genes - n_rare :] = 0
    for j in range(n_rare):
        x[j, n_genes - n_rare + j] = 1
    return _cells(x)


# --- compute_embeddings: removed cells and genes, already preprocessed input --------------


def test_preprocessed_input_is_embedded_as_is(tmp_path, pca_genes):
    """compute_embeddings on preprocess_counts output keeps its cells and fits PCA on its HVGs."""
    import anndata

    raw = str(tmp_path / "raw.h5ad")
    _raw_with_shallow_cells().write_h5ad(raw)
    prepped = str(tmp_path / "prepped.h5ad")
    step2 = preprocess_counts(path=raw, output_path=prepped, min_genes=100, n_top_genes=100)
    assert step2["status"] == "success", step2
    pp = anndata.read_h5ad(prepped)
    assert pp.n_obs == 150 and pp.var["highly_variable"].sum() == 100

    out = str(tmp_path / "embedded.h5ad")
    res = compute_embeddings(path=prepped, models="pca", output_path=out)
    assert res["status"] == "success", res
    embedded = anndata.read_h5ad(out)
    assert list(embedded.obs_names) == list(pp.obs_names)
    assert pca_genes == [list(pp.var_names[pp.var["highly_variable"].to_numpy()])]
    assert (res["n_obs_input"], res["n_obs_output"]) == (pp.n_obs, pp.n_obs)
    assert (res["n_vars_input"], res["n_vars_output"]) == (pp.n_vars, pp.n_vars)
    assert "warning" not in res


def test_removed_cells_and_genes_are_reported(tmp_path):
    """Raw counts with preprocessing='auto': QC removes cells and genes, and the reply says so."""
    import anndata

    raw = str(tmp_path / "raw.h5ad")
    _raw_with_shallow_cells().write_h5ad(raw)
    out = str(tmp_path / "embedded.h5ad")
    res = compute_embeddings(path=raw, models="pca", output_path=out)
    assert res["status"] == "success", res

    embedded = anndata.read_h5ad(out)
    assert (res["n_obs_input"], res["n_vars_input"]) == (150, 600)
    assert (res["n_obs_output"], res["n_vars_output"]) == (embedded.n_obs, embedded.n_vars)
    assert embedded.n_obs == 75 and embedded.n_vars < 600
    assert res["n_cells"] == embedded.n_obs
    assert "75 of 150 cells" in res["warning"]
    assert f"{600 - embedded.n_vars} of 600 genes" in res["warning"]
    assert "cell_75" in res["warning"]


def _skill_dataset() -> AnnData:
    """120 cells x 2100 genes. Cells 0-3 detect exactly 200 genes, two of which no other cell
    detects: min_genes=200 keeps those cells, min_cells=3 then drops the two genes, so the
    cells would not survive a second round of filtering."""
    rng = np.random.default_rng(1)
    n_cells, n_genes = 120, 2100
    lam = rng.gamma(0.8, 1.0, size=n_genes) + 0.1
    programs = np.ones((3, n_genes))
    for t in range(3):
        programs[t, t * 50 : (t + 1) * 50] = 8.0
    x = rng.poisson(lam * programs[np.arange(n_cells) % 3]).astype(np.float32)
    rare = np.arange(n_genes - 8, n_genes)
    x[:, rare] = 0
    common = np.argsort(-lam[: n_genes - 8])[:198]
    for i in range(4):
        x[i] = 0
        x[i, common] = rng.poisson(lam[common]) + 1
        x[i, rare[2 * i : 2 * i + 2]] = 1
    return _cells(x)


def _skill_blocks(skill: str, *tools: str) -> list[str]:
    text = (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    return [block for block in blocks if any(f"{tool}(" in block for tool in tools)]


def test_skill_steps_2_and_3_embed_the_genes_step_2_selected(tmp_path, monkeypatch, pca_genes, fast_scvi):
    """The single-cell Skill's preprocess_counts and compute_embeddings calls, run as written."""
    import anndata

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    _skill_dataset().write_h5ad(tmp_path / "data" / "pbmc.h5ad")

    results = {}
    namespace = {}
    for tool in (preprocess_counts, compute_embeddings):
        namespace[tool.__name__] = lambda _tool=tool, **kwargs: results.setdefault(_tool.__name__, _tool(**kwargs))
    blocks = _skill_blocks("embpy-single-cell-workflow", "preprocess_counts", "compute_embeddings")
    assert len(blocks) == 2
    for block in blocks:
        exec(compile(block, "SKILL.md", "exec"), namespace)

    step2, step3 = results["preprocess_counts"], results["compute_embeddings"]
    assert step2["status"] == "success", step2
    assert step3["status"] == "success", step3
    prepped = anndata.read_h5ad(step2["output_path"])
    selected = list(prepped.var_names[prepped.var["highly_variable"].to_numpy()])
    assert 0 < len(selected) < prepped.n_vars
    assert pca_genes == [selected]
    assert list(anndata.read_h5ad(step3["output_path"]).obs_names) == list(prepped.obs_names)


# --- preprocess_counts: input checks ------------------------------------------------------


def _log_normalized(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)


def _poisson_counts(n_cells: int = 60, n_genes: int = 50, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)


@pytest.mark.parametrize("mode", ["standard", "auto", "none"])
def test_normalized_input_without_counts_is_refused(tmp_path, mode):
    """Log-normalized .X and no counts layer: NotRawCountsError, unless assume_counts=True."""
    path = str(tmp_path / "lognorm.h5ad")
    _cells(_log_normalized(_poisson_counts())).write_h5ad(path)
    out = str(tmp_path / "prepped.h5ad")

    res = preprocess_counts(path=path, output_path=out, preprocessing=mode, min_genes=0, min_cells=0)
    assert res["status"] == "error", res
    assert res["error_type"] == "NotRawCountsError"
    assert res["expression_nature"]["detected_nature"] == "log_normalized"
    assert "log_normalized" in res["error"] and "assume_counts=True" in res["error"]
    assert not os.path.exists(out)

    forced = preprocess_counts(path=path, output_path=out, preprocessing=mode, min_genes=0, min_cells=0,
                               assume_counts=True)
    assert forced["status"] == "success", forced
    assert os.path.exists(out)


def test_counts_layer_is_the_raw_source(tmp_path):
    """Normalized .X next to a counts layer: normalization starts from the counts layer."""
    import anndata

    counts = _poisson_counts()
    adata = _cells(_log_normalized(counts))
    adata.layers["counts"] = counts.copy()
    path = str(tmp_path / "with_counts.h5ad")
    adata.write_h5ad(path)
    out = str(tmp_path / "prepped.h5ad")

    res = preprocess_counts(path=path, output_path=out, min_genes=0, min_cells=0, select_hvg=False)
    assert res["status"] == "success", res
    saved = anndata.read_h5ad(out)
    np.testing.assert_allclose(saved.layers["log_normalized"], _log_normalized(counts), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(saved.layers["counts"], counts)
    np.testing.assert_array_equal(saved.X, counts)


@pytest.mark.parametrize(
    ("thresholds", "emptied"),
    [({"min_genes": 200}, "cell"), ({"min_genes": 0, "min_cells": 1000}, "gene")],
    ids=["no-cells-left", "no-genes-left"],
)
def test_filtering_everything_is_an_error(tmp_path, thresholds, emptied):
    """40 genes with min_genes=200 (or min_cells above the cell count): an error and no file."""
    path = str(tmp_path / "small.h5ad")
    _cells(_poisson_counts(30, 40)).write_h5ad(path)
    out = str(tmp_path / "prepped.h5ad")

    res = preprocess_counts(path=path, output_path=out, **thresholds)
    assert res["status"] == "error", res
    assert res["error_type"] == "EmptyAfterFilteringError"
    expected = {"min_genes": thresholds["min_genes"], "min_cells": thresholds.get("min_cells", 3)}
    assert res["thresholds"] == expected
    assert f"every {emptied}" in res["error"]
    assert not os.path.exists(out)


# --- compute_umap / benchmark_scib --------------------------------------------------------


def _embedded_cells(nan_rows=()) -> AnnData:
    rng = np.random.default_rng(2)
    adata = _cells(rng.poisson(2.0, size=(60, 30)))
    centers = rng.normal(scale=4.0, size=(3, 8))
    for key, seed in (("X_emb", 0), ("X_pca", 1), ("X_prior", 2)):
        noise = np.random.default_rng(seed).normal(size=(60, 8))
        adata.obsm[key] = (centers[np.arange(60) % 3] + noise).astype(np.float32)
    adata.obsm["X_emb"][list(nan_rows)] = np.nan
    return adata


def _same(a, b) -> bool:
    import scipy.sparse as sp

    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if sp.issparse(a):
        return sp.issparse(b) and a.dtype == b.dtype and a.shape == b.shape and (a != b).nnz == 0
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def test_umap_keeps_the_existing_umap_and_neighbors(tmp_path):
    """Only the new coordinates and this embedding's neighbors graph are added; nothing else changes."""
    import anndata
    import scanpy as sc

    adata = _embedded_cells()
    sc.pp.neighbors(adata, use_rep="X_prior", n_neighbors=5)
    sc.tl.umap(adata, random_state=0)
    path = str(tmp_path / "with_umap.h5ad")
    adata.write_h5ad(path)
    before = anndata.read_h5ad(path)

    out = str(tmp_path / "umap_pca.h5ad")
    res = compute_umap(path=path, embedding_key="X_pca", n_neighbors=5, output_path=out)
    assert res["status"] == "success", res
    after = anndata.read_h5ad(out)

    for slot in ("obsm", "obsp", "uns"):
        old, new = getattr(before, slot), getattr(after, slot)
        changed = [key for key in old if key not in new or not _same(old[key], new[key])]
        assert not changed, f"{slot} entries changed: {changed}"
    nkey = res["neighbors_key"]
    assert res["umap_key"] == "X_umap_pca" and nkey == "neighbors_pca"
    assert set(after.obsm) - set(before.obsm) == {"X_umap_pca"}
    assert set(after.obsp) - set(before.obsp) == {f"{nkey}_connectivities", f"{nkey}_distances"}
    assert set(after.uns) - set(before.uns) == {nkey}
    assert after.uns[nkey]["params"]["use_rep"] == "X_pca"
    assert after.obsm["X_umap_pca"].shape == (60, 2)


def test_umap_leaves_out_rows_without_a_finite_embedding(tmp_path):
    """Two NaN rows (as missing='nan' leaves them): excluded, counted, named, NaN coordinates."""
    import anndata

    adata = _embedded_cells(nan_rows=(3, 10))
    adata.obsm["X_all_nan"] = np.full((60, 4), np.nan, dtype=np.float32)
    path = str(tmp_path / "with_nan.h5ad")
    adata.write_h5ad(path)

    out = str(tmp_path / "umap.h5ad")
    res = compute_umap(path=path, embedding_key="X_emb", n_neighbors=5, output_path=out)
    assert res["status"] == "success", res
    assert res["n_excluded_nonfinite"] == 2
    assert "cell_3" in res["warning"] and "cell_10" in res["warning"] and "missing='nan'" in res["warning"]
    coords = np.asarray(anndata.read_h5ad(out).obsm[res["umap_key"]])
    assert coords.shape == (60, 2)
    assert np.isnan(coords[[3, 10]]).all()
    assert np.isfinite(np.delete(coords, [3, 10], axis=0)).all()

    none_left = compute_umap(path=path, embedding_key="X_all_nan", output_path=str(tmp_path / "x.h5ad"))
    assert none_left["status"] == "error" and "cell_0" in none_left["error"]
    assert not os.path.exists(tmp_path / "x.h5ad")


def test_scib_leaves_out_rows_without_a_finite_embedding(tmp_path):
    """benchmark_scib scores the other cells and reports the excluded ones like compute_umap."""
    pytest.importorskip("scib", reason="scib is not installed")
    path = str(tmp_path / "with_nan.h5ad")
    _embedded_cells(nan_rows=(3, 10)).write_h5ad(path)

    res = benchmark_scib(path=path, embedding_keys=["X_emb", "X_pca"], label_key="cell_type", batch_key="batch")
    assert res["status"] == "success", res
    assert res["n_excluded_nonfinite"] == 2
    assert res["n_cells_scored"] == 58
    assert "cell_3" in res["warning"] and "cell_10" in res["warning"] and "missing='nan'" in res["warning"]
    assert set(res["metrics"]) == {"X_emb", "X_pca"}
