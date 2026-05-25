from __future__ import annotations

import numpy as np
import pytest

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.store import EmbeddingStore, control_sentinel_vector
from world_model.configs import ActionEmbeddingConfig
from world_model.data.embeddings import build_provider
from world_model.data.embeddings.sentinel import make_control_vector
from world_model.data.embeddings.store_provider import StoreProvider


def _store() -> EmbeddingStore:
    result = EmbeddingResult(
        matrix=np.array([[1.0, 0.0], [0.0, 1.0], [0.25, 0.25]], dtype=np.float32),
        entity_ids=("g1", "g2", "g3"),
        entity_type="gene",
        id_scheme="symbol",
        provenance=EmbeddingProvenance(model="toy"),
    )
    return EmbeddingStore.from_results(result)


def test_store_provider_status_aware_resolution_in_memory():
    p = StoreProvider(store=_store())
    rows, statuses = p.embed_with_status(["g1", "g1+g2", "non-targeting", "ghost"])

    assert p.embedding_dim == 2
    assert list(statuses) == ["RESOLVED", "RESOLVED", "CONTROL", "UNRESOLVED"]
    assert rows[0].tolist() == pytest.approx([1.0, 0.0])
    assert rows[1].tolist() == pytest.approx([0.5, 0.5])
    assert rows[2].tolist() == pytest.approx(make_control_vector(2, seed=0).tolist())
    assert rows[3].tolist() == [0.0, 0.0]
    assert p._last_unresolved == ["ghost"]
    assert p._last_controls == ["non-targeting"]


def test_store_provider_partial_combo_resolves_and_records_missing():
    p = StoreProvider(store=_store())
    rows, statuses = p.embed_with_status(["g1+ghost"])
    assert list(statuses) == ["RESOLVED"]
    assert rows[0].tolist() == pytest.approx([1.0, 0.0])
    assert p._last_unresolved == ["ghost"]


def test_store_provider_build_table_and_metadata_from_disk(tmp_path):
    path = _store().write(tmp_path / "genes.emstore")
    p = StoreProvider(store_path=path)
    table, indexer = p.build_table(["g1", "g2"])

    assert table.shape == (3, 2)  # row 0 control + g1 + g2
    assert indexer.symbol_to_index["g1"] == 1
    assert np.allclose(table[1], [1.0, 0.0])

    meta = p.metadata(n_symbols=2, n_unresolved=0)
    assert meta.source == "store"
    assert meta.embedding_dim == 2
    assert meta.model_name == "toy"
    assert meta.n_resolved == 2


def test_store_provider_auto_selects_gene_block_or_errors_on_ambiguity():
    store = EmbeddingStore()
    store.add_embedding("gene:a", np.eye(2, dtype=np.float32), ["g1", "g2"], entity_type="gene", id_scheme="symbol")
    store.add_embedding("prot:b", np.eye(2, dtype=np.float32), ["p1", "p2"], entity_type="protein", id_scheme="uniprot")
    # Two blocks but a single gene block -> auto-selected.
    assert StoreProvider(store=store).embedding_dim == 2

    store.add_embedding("gene:c", np.eye(2, dtype=np.float32), ["g3", "g4"], entity_type="gene", id_scheme="symbol")
    with pytest.raises(ValueError, match="multiple embeddings"):
        StoreProvider(store=store).embed_with_status(["g1"])


def test_build_provider_store_source(tmp_path):
    path = _store().write(tmp_path / "genes.emstore")
    cfg = ActionEmbeddingConfig(source="store", store_path=str(path))
    provider = build_provider(cfg)
    assert isinstance(provider, StoreProvider)
    rows, statuses = provider.embed_with_status(["g1", "non-targeting"])
    assert list(statuses) == ["RESOLVED", "CONTROL"]


def test_build_provider_store_requires_path():
    with pytest.raises(ValueError, match="requires action_embedding.store_path"):
        build_provider(ActionEmbeddingConfig(source="store"))


def test_make_control_vector_bit_compatible_with_embpy():
    # StoreProvider keeps world_model.sentinel dependency-light (no embpy import);
    # this guards that the two control sentinels stay bit-identical.
    for dim, seed in [(2, 0), (8, 0), (16, 3), (128, 7)]:
        assert np.array_equal(
            make_control_vector(dim, seed=seed),
            control_sentinel_vector(dim, seed=seed),
        )
