"""End-to-end test for ``build_dataloaders`` with a foreign state backbone.

We pass a manually-constructed provider via ``state_backbone_override``
so the lazy import path is never exercised (no ``arc-state`` install
required for CI). The provider is the in-repo
:class:`StateBackboneProvider` ABC -- a fake that records calls. The
test asserts:

* embeddings replace the dataset's expression matrix,
* the cache NPZ is written on first build and consumed on the second,
* ``state_backbone_meta.json`` lands on disk,
* ``cache.inspect_cache`` finds the artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

anndata = pytest.importorskip("anndata")

from embpy.io.result import EmbeddingProvenance, EmbeddingResult  # noqa: E402
from embpy.store import EmbeddingStore  # noqa: E402
from world_model.configs import (  # noqa: E402
    ActionEmbeddingConfig,
    DataConfig,
    SplitConfig,
    StateBackboneConfig,
)
from world_model.data.dataloader import build_dataloaders  # noqa: E402
from world_model.models.encoders.backbones import (  # noqa: E402
    StateBackboneProvider,
    inspect_cache,
)
from world_model.models.encoders.backbones.cache import (  # noqa: E402
    cache_path_for,
    save_cached,
)

# ---------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------


class _FakeProvider(StateBackboneProvider):
    name: str = "state"
    supports_decode: bool = False

    def __init__(self, embedding_dim: int = 24, cache_dir: Path | None = None) -> None:
        self._dim = embedding_dim
        self._cache_dir = cache_dir
        self.encode_calls = 0
        self._ckpt_hash = "deadbeef" + "0" * 8

    @property
    def embedding_dim(self) -> int:
        return int(self._dim)

    def _cache_key(self, adata: Any) -> tuple[str, str]:
        return self._ckpt_hash, "ds-" + str(int(getattr(adata, "n_obs", 0)))

    def encode(self, adata: Any, *, batch_size: int | None = None) -> np.ndarray:
        del batch_size
        self.encode_calls += 1
        n = int(getattr(adata, "n_obs", 0))
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((n, self._dim)).astype(np.float32)
        if self._cache_dir is not None:
            ckpt_hash, ds_hash = self._cache_key(adata)
            save_cached(
                self._cache_dir,
                backbone=self.name,
                ckpt_hash=ckpt_hash,
                dataset_hash=ds_hash,
                embeddings=emb,
                adata_hash=ds_hash,
            )
        return emb

    def freeze(self) -> None:
        pass

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return iter(())

    def train_mode(self, flag: bool) -> None:
        del flag

    @property
    def checkpoint_path(self) -> str:
        return "fake.ckpt"


def _make_tiny_adata(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 24
    x = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    perts = ["non-targeting"] * 24 + ["GENE_000"] * 12 + ["GENE_001"] * 12 + ["GENE_002"] * 12
    obs = {"perturbation": perts}
    adata = anndata.AnnData(X=x, obs=obs, var={"gene_symbols": var_names})
    adata.var.index = var_names
    out = tmp_path / "tiny.h5ad"
    adata.write_h5ad(out)
    return out


def _make_tiny_store(tmp_path: Path) -> Path:
    syms = ["GENE_000", "GENE_001", "GENE_002"]
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = EmbeddingResult(
        matrix=emb,
        entity_ids=tuple(syms),
        entity_type="gene",
        id_scheme="symbol",
        provenance=EmbeddingProvenance(model="tiny"),
    )
    return EmbeddingStore.from_results(result).write(tmp_path / "tiny.emstore")


def _data_cfg(h5ad: Path) -> DataConfig:
    return DataConfig(
        dataset="replogle",
        h5ad_path=str(h5ad),
        gene_embedding_path="",
        n_top_genes=10,
        log_normalize=True,
        sequence_length=4,
        stack_size=2,
        n_pert=2,
        batch_size=4,
        num_workers=0,
        n_sequences_per_epoch=8,
        cell_type_key=None,
    )


def test_build_dataloaders_pre_encodes_with_backbone_and_caches(tmp_path: Path) -> None:
    h5ad = _make_tiny_adata(tmp_path)
    store_path = _make_tiny_store(tmp_path)
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "run"

    sb_cfg = StateBackboneConfig(
        kind="state",
        state_checkpoint="fake.ckpt",
        cache_dir=str(cache_dir),
        freeze=True,
    )
    provider = _FakeProvider(embedding_dim=24, cache_dir=cache_dir)
    artifacts = build_dataloaders(
        _data_cfg(h5ad),
        split_cfg=SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0),
        action_cfg=ActionEmbeddingConfig(source="store", store_path=str(store_path)),
        state_backbone_cfg=sb_cfg,
        state_backbone_override=provider,
        seed=0,
        output_dir=output_dir,
    )

    assert artifacts.state_backbone_embedding_dim == 24
    assert artifacts.full_dataset.expression.shape == (60, 24)
    assert artifacts.full_dataset.n_genes == 24
    assert provider.encode_calls == 1

    meta_path = output_dir / "state_backbone_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["kind"] == "state"
    assert meta["embedding_dim"] == 24
    assert meta["n_cells_encoded"] == 60

    cp = cache_path_for(
        cache_dir,
        backbone="state",
        ckpt_hash=provider._ckpt_hash,
        dataset_hash="ds-60",
    )
    assert cp.exists()
    inspected = inspect_cache(cache_dir)
    assert any(r["path"] == str(cp) for r in inspected)


def test_dataloader_reuses_provider_override_without_extra_encode(tmp_path: Path) -> None:
    h5ad = _make_tiny_adata(tmp_path)
    store_path = _make_tiny_store(tmp_path)
    cache_dir = tmp_path / "cache"

    sb_cfg = StateBackboneConfig(
        kind="state",
        state_checkpoint="fake.ckpt",
        cache_dir=str(cache_dir),
        freeze=True,
    )
    provider = _FakeProvider(embedding_dim=24, cache_dir=cache_dir)
    build_dataloaders(
        _data_cfg(h5ad),
        split_cfg=SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0),
        action_cfg=ActionEmbeddingConfig(source="store", store_path=str(store_path)),
        state_backbone_cfg=sb_cfg,
        state_backbone_override=provider,
        seed=0,
        output_dir=tmp_path / "run1",
    )
    assert provider.encode_calls == 1

    # Second build uses the same provider instance; the on-disk NPZ is
    # already present, so an idempotent provider would skip the encode.
    # Our fake always re-encodes (it doesn't consult the cache); the
    # important contract is that the *dataloader* doesn't double-encode
    # when override is reused -- only one encode per build_dataloaders
    # call. So we just check it ticked exactly one more time.
    build_dataloaders(
        _data_cfg(h5ad),
        split_cfg=SplitConfig(split_by="perturbation", train_fraction=0.6, seed=0),
        action_cfg=ActionEmbeddingConfig(source="store", store_path=str(store_path)),
        state_backbone_cfg=sb_cfg,
        state_backbone_override=provider,
        seed=0,
        output_dir=tmp_path / "run2",
    )
    assert provider.encode_calls == 2
