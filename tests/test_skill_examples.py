"""Run the code examples of the Agent Skills on small datasets.

Each skill's ```python blocks are executed in order in one namespace, inside a
temporary directory holding toy versions of the files the examples read
(``data/cells.h5ad``, ``data/screen.h5ad``, ...). MCP tool calls in the
examples go to the real tool functions and fail the test if a tool answers
``status: "error"``. Blocks marked ``# Requires:`` need resources a test cannot
provide (e.g. a whole-genome FASTA) and are only checked statically, by
``test_skills.py``.

The examples download small models (gene2vec, ESM-2 8M, HyenaDNA tiny) and call
web services (Ensembl, UniProt, PubChem, ChEMBL), so this runs only with
``EMBPY_RUN_SKILL_EXAMPLES=1``.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.mcp import tools as mcp_tools

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBPY_RUN_SKILL_EXAMPLES") != "1",
    reason="downloads models and calls web services; set EMBPY_RUN_SKILL_EXAMPLES=1",
)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
# Skills whose examples run on the toy data below. The first three skills reference
# datasets and models (scVI, foundation models) that are checked statically and were
# exercised in live assistant sessions instead.
RUNNABLE_SKILLS = [
    "embpy-similarity-search",
    "embpy-embedding-clustering",
    "embpy-compare-embedding-models",
    "embpy-embedding-visualization",
    "embpy-perturbation-screen-analysis",
    "embpy-compound-screen-analysis",
    "embpy-perturbation-prediction-metrics",
    "embpy-embedding-task-benchmark",
    "embpy-cross-species-proteins",
    "embpy-variant-effects",
    "embpy-mcp-setup",
]
TOOL_NAMES = (
    "inspect_data",
    "resolve_identifiers",
    "preprocess_counts",
    "list_model_registry",
    "compute_embeddings",
    "annotate_metadata",
    "compute_umap",
    "benchmark_scib",
    "benchmark_perturbations",
)


def _strict(fn):
    @functools.wraps(fn)
    def call(**kwargs):
        result = fn(**kwargs)
        assert result.get("status") != "error", f"{fn.__name__} failed: {result}"
        return result

    return call


def _pca(x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=n, random_state=seed).fit_transform(np.log1p(x)).astype(np.float32)


def _counts(rng, groups: list[str], n_per: int, n_genes: int, effect: float) -> np.ndarray:
    base = rng.gamma(2.0, 1.5, size=n_genes)
    rows = []
    for i, _ in enumerate(groups):
        shift = np.ones(n_genes)
        shift[(i * 7) % n_genes : (i * 7) % n_genes + 7] = effect if i else 1.0
        rows.append(rng.poisson(base * shift, size=(n_per, n_genes)))
    return np.vstack(rows).astype(np.float32)


def _make_data(data: Path) -> None:
    data.mkdir()
    rng = np.random.default_rng(0)

    types = ["T_cell", "B_cell", "Monocyte"]
    x = _counts(rng, types, 50, 60, 6.0)
    cells = AnnData(X=x)
    cells.obs_names = [f"cell_{i}" for i in range(cells.n_obs)]
    cells.var_names = [f"gene_{j}" for j in range(cells.n_vars)]
    cells.obs["cell_type"] = np.repeat(types, 50)
    cells.obs["batch"] = np.tile(["b1", "b2"], cells.n_obs // 2)
    cells.write_h5ad(data / "cells.h5ad")
    embedded = cells.copy()
    embedded.obsm["X_pca"] = _pca(x, 20)
    embedded.obsm["X_scvi"] = _pca(x, 10, seed=1) + rng.normal(scale=0.3, size=(cells.n_obs, 10)).astype(np.float32)
    embedded.write_h5ad(data / "cells_embedded.h5ad")

    targets = ["control", "TP53", "MDM2", "CDKN1A", "GAPDH", "ACTB"]
    xs = _counts(rng, targets, 25, 60, 5.0)
    screen = AnnData(X=xs)
    screen.obs_names = [f"cell_{i}" for i in range(screen.n_obs)]
    screen.var_names = [f"gene_{j}" for j in range(screen.n_vars)]
    screen.obs["target_gene"] = np.repeat(targets, 25)
    screen.write_h5ad(data / "screen.h5ad")
    screen_emb = screen.copy()
    screen_emb.obsm["X_pca"] = _pca(xs, 20)
    screen_emb.write_h5ad(data / "screen_embedded.h5ad")

    compounds = AnnData(
        X=np.zeros((12, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"compound": np.repeat(["aspirin", "ibuprofen", "caffeine", "DMSO"], 3)},
            index=[f"well_{i}" for i in range(12)],
        ),
    )
    compounds.write_h5ad(data / "compounds.h5ad")

    smiles = ["C" * n + "O" for n in range(1, 21)] + ["C" * n + "N" for n in range(1, 21)]
    ic50 = [0.3 * len(s) + (0.8 if s.endswith("N") else 0.0) + rng.normal(scale=0.2) for s in smiles]
    AnnData(
        X=np.zeros((len(smiles), 2), dtype=np.float32),
        obs=pd.DataFrame({"smiles": smiles, "ic50": ic50}, index=[f"mol_{i}" for i in range(len(smiles))]),
    ).write_h5ad(data / "ic50.h5ad")

    perts = ["control", "TP53", "MDM2", "CDKN1A", "GAPDH", "ACTB"]
    xo = np.log1p(_counts(rng, perts, 30, 40, 4.0))
    observed = AnnData(X=xo)
    observed.obs_names = [f"cell_{i}" for i in range(observed.n_obs)]
    observed.var_names = [f"gene_{j}" for j in range(observed.n_vars)]
    observed.obs["perturbation"] = np.repeat(perts, 30)
    observed.write_h5ad(data / "observed.h5ad")
    predicted = observed.copy()
    predicted.X = np.clip(xo + rng.normal(scale=0.4, size=xo.shape), 0, None).astype(np.float32)
    predicted.obs_names = [f"pred_{i}" for i in range(predicted.n_obs)]
    predicted.write_h5ad(data / "predicted.h5ad")


@pytest.mark.parametrize("skill", RUNNABLE_SKILLS)
def test_skill_examples_run(skill, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    _make_data(tmp_path / "data")
    monkeypatch.chdir(tmp_path)

    text = (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, f"{skill} has no python examples"
    namespace = {name: _strict(getattr(mcp_tools, name)) for name in TOOL_NAMES}
    ran = 0
    for i, block in enumerate(blocks):
        if "# Requires:" in block:
            continue
        exec(compile(block, f"{skill} example {i + 1}", "exec"), namespace)  # noqa: S102
        ran += 1
    assert ran, f"{skill}: every example needs external resources"
