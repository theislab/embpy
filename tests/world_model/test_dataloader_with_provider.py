"""End-to-end test that ``build_dataloaders`` honours AnnData action embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from world_model.configs import (  # noqa: E402
    ActionEmbeddingConfig,
    DataConfig,
    SplitConfig,
)
from world_model.data.dataloader import build_dataloaders  # noqa: E402


def _make_tiny_adata(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_cells, n_genes = 80, 30
    x = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    perts = ["non-targeting"] * 30 + ["GENE_000"] * 20 + ["GENE_001"] * 15 + ["GENE_002"] * 15
    obs = {"perturbation": perts}
    adata = anndata.AnnData(X=x, obs=obs, var={"gene_symbols": var_names})
    adata.var.index = var_names
    adata.obsm["X_state_tiny"] = x[:, :6].astype(np.float32)
    by_pert = {
        "non-targeting": np.zeros(4, dtype=np.float32),
        "GENE_000": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "GENE_001": np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "GENE_002": np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32),
    }
    adata.obsm["X_pert_tiny"] = np.stack([by_pert[p] for p in perts], axis=0)
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


def test_build_dataloaders_with_anndata_obsm_provider_writes_meta(tmp_path: Path):
    h5ad = _make_tiny_adata(tmp_path)

    data_cfg = DataConfig(
        dataset="tiny_any_adata",
        h5ad_path=str(h5ad),
        state_obsm_key="X_state_tiny",
        sequence_length=4,
        stack_size=2,
        n_pert=2,
        batch_size=4,
        num_workers=0,
        n_sequences_per_epoch=16,
    )
    split_cfg = SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0)
    action_cfg = ActionEmbeddingConfig(source="anndata_obsm", obsm_key="X_pert_tiny")

    output_dir = tmp_path / "run"
    artifacts = build_dataloaders(
        data_cfg,
        split_cfg=split_cfg,
        action_cfg=action_cfg,
        seed=0,
        output_dir=output_dir,
    )

    assert artifacts.gene_table.shape == (4, 4)
    np.testing.assert_array_equal(
        artifacts.gene_table[0],
        np.zeros(4, dtype=np.float32),
    )

    meta_path = output_dir / "action_embedding_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "anndata_obsm"
    assert meta["embedding_dim"] == 4
    assert meta["n_symbols"] == 3
    assert (output_dir / "action_embedding_status.json").exists()
