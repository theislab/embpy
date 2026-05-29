"""Dataloader contract for pre-attached AnnData state embeddings."""

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
    StateBackboneConfig,
)
from world_model.data.dataloader import build_dataloaders  # noqa: E402


def _make_tiny_adata(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 24
    x = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    perts = ["non-targeting"] * 24 + ["GENE_000"] * 12 + ["GENE_001"] * 12 + ["GENE_002"] * 12
    obs = {"perturbation": perts}
    adata = anndata.AnnData(X=x, obs=obs, var={"gene_symbols": var_names})
    adata.var.index = var_names
    adata.obsm["X_state_tiny"] = rng.standard_normal((n_cells, 24)).astype(np.float32)
    rows = {
        "non-targeting": np.zeros(4, dtype=np.float32),
        "GENE_000": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "GENE_001": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "GENE_002": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    adata.obsm["X_pert_tiny"] = np.stack([rows[p] for p in perts], axis=0)
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


def _data_cfg(h5ad: Path) -> DataConfig:
    return DataConfig(
        dataset="tiny_any_adata",
        h5ad_path=str(h5ad),
        state_obsm_key="X_state_tiny",
        sequence_length=4,
        stack_size=2,
        n_pert=2,
        batch_size=4,
        num_workers=0,
        n_sequences_per_epoch=8,
    )


def test_build_dataloaders_reads_state_and_action_from_obsm(tmp_path: Path) -> None:
    h5ad = _make_tiny_adata(tmp_path)
    output_dir = tmp_path / "run"

    artifacts = build_dataloaders(
        _data_cfg(h5ad),
        split_cfg=SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0),
        action_cfg=ActionEmbeddingConfig(source="anndata_obsm", obsm_key="X_pert_tiny"),
        state_backbone_cfg=StateBackboneConfig(kind="local", freeze=True),
        seed=0,
        output_dir=output_dir,
    )

    assert artifacts.state_backbone is None
    assert artifacts.state_backbone_embedding_dim is None
    assert artifacts.full_dataset.expression.shape == (60, 24)
    assert artifacts.full_dataset.n_genes == 24
    assert artifacts.gene_symbols[:2] == ["X_state_tiny_0", "X_state_tiny_1"]

    meta_path = output_dir / "state_embedding_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["kind"] == "anndata_obsm"
    assert meta["state_head_kind"] == "local"
    assert meta["obsm_key"] == "X_state_tiny"
    assert meta["embedding_dim"] == 24
    assert meta["n_cells_encoded"] == 60


def test_foreign_state_head_uses_obsm_dim_without_encoding(tmp_path: Path) -> None:
    h5ad = _make_tiny_adata(tmp_path)
    artifacts = build_dataloaders(
        _data_cfg(h5ad),
        split_cfg=SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0),
        action_cfg=ActionEmbeddingConfig(source="anndata_obsm", obsm_key="X_pert_tiny"),
        state_backbone_cfg=StateBackboneConfig(kind="state", freeze=True),
        seed=0,
        output_dir=tmp_path / "run",
    )

    assert artifacts.state_backbone is None
    assert artifacts.state_backbone_embedding_dim == 24
