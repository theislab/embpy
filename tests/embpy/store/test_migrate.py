from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embpy.store import EmbeddingStore
from embpy.store.migrate import (
    gene_store_from_table,
    migrate_table_to_emstore,
    read_embedding_table,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        index=["TP53", "MYC"],
        columns=["d0", "d1", "d2"],
    )


def test_read_embedding_table_csv_npz_dataframe(tmp_path):
    frame = _frame()
    ids_df, mat_df = read_embedding_table(frame)
    assert ids_df == ["TP53", "MYC"]
    assert mat_df.shape == (2, 3)

    csv = tmp_path / "emb.csv"
    frame.to_csv(csv)
    ids_csv, mat_csv = read_embedding_table(csv)
    assert ids_csv == ["TP53", "MYC"]
    assert np.allclose(mat_csv, frame.to_numpy(dtype=np.float32))

    npz = tmp_path / "emb.npz"
    np.savez(
        npz,
        symbols=np.array(["TP53", "MYC"], dtype=object),
        embeddings=frame.to_numpy(dtype=np.float32),
    )
    ids_npz, mat_npz = read_embedding_table(npz)
    assert ids_npz == ["TP53", "MYC"]
    assert np.allclose(mat_npz, frame.to_numpy(dtype=np.float32))


def test_gene_store_from_table_builds_store_with_provenance():
    store = gene_store_from_table(_frame(), model="genept", id_scheme="symbol")
    assert store.keys() == ["gene:genept"]
    block = store.embedding("gene:genept")
    assert block.entity_ids == ("TP53", "MYC")
    assert block.entity_type == "gene"
    assert block.n_dims == 3
    assert block.provenance["model"] == "genept"
    assert block.provenance["extra"]["migrated_from"] == "<dataframe>"


def test_migrate_table_to_emstore_roundtrip(tmp_path):
    csv = tmp_path / "emb.csv"
    _frame().to_csv(csv)
    dest = tmp_path / "gene_genept.emstore"

    path = migrate_table_to_emstore(csv, dest, model="genept")
    assert path == dest

    loaded = EmbeddingStore.read(path)
    block = loaded.embedding("gene:genept")
    assert block.entity_ids == ("TP53", "MYC")
    assert np.allclose(block.matrix, _frame().to_numpy(dtype=np.float32))


def test_migrate_requires_emstore_suffix(tmp_path):
    csv = tmp_path / "emb.csv"
    _frame().to_csv(csv)
    with pytest.raises(ValueError, match="must end with '.emstore'"):
        migrate_table_to_emstore(csv, tmp_path / "bad_dest", model="genept")


def test_read_embedding_table_missing_and_bad_npz(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_embedding_table(tmp_path / "nope.csv")
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    with pytest.raises(KeyError, match="symbols"):
        read_embedding_table(bad)
