from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from world_model.configs import ActionEmbeddingConfig, DataConfig  # noqa: E402
from world_model.data.embeddings import AnnDataObsmProvider, build_provider  # noqa: E402
from world_model.data.embeddings.sentinel import make_control_vector  # noqa: E402


def _write_adata(path: Path) -> Path:
    labels = ["non-targeting", "g1", "g1", "g2", "ghost"]
    adata = anndata.AnnData(
        X=np.zeros((len(labels), 3), dtype=np.float32),
        obs={"perturbation": labels},
        var={"gene": ["a", "b", "c"]},
    )
    rows = {
        "non-targeting": np.zeros(2, dtype=np.float32),
        "g1": np.array([1.0, 0.0], dtype=np.float32),
        "g2": np.array([0.0, 1.0], dtype=np.float32),
        "ghost": np.zeros(2, dtype=np.float32),
    }
    adata.obsm["X_pert_toy"] = np.stack([rows[label] for label in labels], axis=0)
    adata.write_h5ad(path)
    return path


def test_anndata_obsm_provider_status_aware_resolution(tmp_path: Path) -> None:
    path = _write_adata(tmp_path / "tiny.h5ad")
    provider = AnnDataObsmProvider(path, obsm_key="X_pert_toy")

    rows, statuses = provider.embed_with_status(["g1", "g2", "non-targeting", "ghost"])

    assert provider.embedding_dim == 2
    assert list(statuses) == ["RESOLVED", "RESOLVED", "CONTROL", "UNRESOLVED"]
    assert rows[0].tolist() == pytest.approx([1.0, 0.0])
    assert rows[1].tolist() == pytest.approx([0.0, 1.0])
    assert rows[2].tolist() == pytest.approx(make_control_vector(2, seed=0).tolist())
    assert rows[3].tolist() == [0.0, 0.0]
    assert provider._last_unresolved == ["ghost"]
    assert provider._last_controls == ["non-targeting"]


def test_anndata_obsm_provider_build_table_and_metadata(tmp_path: Path) -> None:
    path = _write_adata(tmp_path / "tiny.h5ad")
    provider = AnnDataObsmProvider(path, obsm_key="X_pert_toy")
    table, indexer = provider.build_table(["g1", "g2"])

    assert table.shape == (3, 2)
    assert indexer.symbol_to_index["g1"] == 1
    np.testing.assert_array_equal(table[1], np.array([1.0, 0.0], dtype=np.float32))

    meta = provider.metadata(n_symbols=2, n_unresolved=0)
    assert meta.source == "anndata_obsm"
    assert meta.embedding_dim == 2
    assert meta.n_resolved == 2
    assert meta.extras["obsm_key"] == "X_pert_toy"


def test_build_provider_anndata_obsm_source(tmp_path: Path) -> None:
    path = _write_adata(tmp_path / "tiny.h5ad")
    cfg = ActionEmbeddingConfig(source="anndata_obsm", obsm_key="X_pert_toy")
    provider = build_provider(cfg, data_cfg=DataConfig(h5ad_path=str(path)))
    assert isinstance(provider, AnnDataObsmProvider)
    rows, statuses = provider.embed_with_status(["g1", "non-targeting"])
    assert rows.shape == (2, 2)
    assert list(statuses) == ["RESOLVED", "CONTROL"]


def test_build_provider_store_source_removed() -> None:
    with pytest.raises(ValueError, match="source='store'.*removed"):
        build_provider(ActionEmbeddingConfig(source="store", store_path="old.emstore"))


def test_build_provider_precomputed_source_removed_from_training(tmp_path: Path) -> None:
    path = _write_adata(tmp_path / "tiny.h5ad")
    with pytest.raises(ValueError, match="not a training input"):
        build_provider(
            ActionEmbeddingConfig(source="precomputed", path="old.npz"),
            data_cfg=DataConfig(h5ad_path=str(path)),
        )
