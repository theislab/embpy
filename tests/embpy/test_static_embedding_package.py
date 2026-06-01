from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embpy.pp.static_embeddings import (
    StaticEmbeddingSource,
    StaticEmbeddingStore,
    load_static_embedding_package,
    prepare_static_embedding_package,
    read_static_embedding_table,
    validate_static_embedding_package,
    write_static_embedding_package,
)


def test_prepare_validate_and_query_static_embedding_package(tmp_path):
    source_root = tmp_path / "sources"
    source_dir = source_root / "genept" / "scaled"
    source_dir.mkdir(parents=True)
    path = source_dir / "embeddings_3072.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["TP53", "MYC"]),
        columns=["0", "1"],
    ).to_csv(path)

    package_root = tmp_path / "package"
    manifest = prepare_static_embedding_package(source_root, package_root)

    assert sorted(manifest["embeddings"]) == ["genept_scaled"]
    assert (package_root / "embeddings" / "genept_scaled" / "values.zarr").is_dir()
    assert (package_root / "embeddings" / "genept_scaled" / "metadata" / "index.parquet").is_file()
    assert (package_root / "embeddings" / "genept_scaled" / "metadata" / "uns.json").is_file()

    validation = validate_static_embedding_package(package_root)
    assert validation[0].key == "genept_scaled"
    assert validation[0].n_entities == 2
    assert validation[0].n_dims == 2

    store = load_static_embedding_package(package_root, key="genept_scaled")
    assert isinstance(store, StaticEmbeddingStore)
    assert np.array_equal(store.get("TP53"), np.array([1.0, 2.0], dtype=np.float32))

    frame = store.query(["TP53", "MYC"])
    assert list(frame.index) == ["TP53", "MYC"]
    assert list(frame.columns) == ["dim_0", "dim_1"]


def test_static_embedding_query_missing_policies(tmp_path):
    source = StaticEmbeddingSource(
        key="toy",
        path=tmp_path / "toy.csv",
        id_type="symbol",
    )
    pd.DataFrame([[1.0, 2.0]], index=pd.Index(["TP53"]), columns=["0", "1"]).to_csv(source.path)

    table = read_static_embedding_table(source)
    package_root = tmp_path / "package"
    write_static_embedding_package(table, package_root)
    store = load_static_embedding_package(package_root, key="toy")

    with pytest.raises(KeyError, match="not present"):
        store.get(["TP53", "MYC"])

    dropped = store.query(["TP53", "MYC"], missing="drop")
    assert list(dropped.index) == ["TP53"]

    nan_rows = store.get(["TP53", "MYC"], missing="nan")
    assert nan_rows.shape == (2, 2)
    assert np.isnan(nan_rows[1]).all()


def test_read_static_embedding_table_duplicate_policy(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["TP53", "TP53"]),
        columns=["0", "1"],
    ).to_csv(path)
    source = StaticEmbeddingSource(key="dup", path=path, id_type="symbol")

    with pytest.raises(ValueError, match="duplicate identifiers"):
        read_static_embedding_table(source)

    table = read_static_embedding_table(source, duplicate_policy="first")
    assert table.entity_ids == ("TP53",)
    assert table.n_duplicate_input_ids == 1


def test_read_static_embedding_table_missing_ids_are_explicit(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("gene_id,0,1\nTP53,1,2\n,3,4\n")
    source = StaticEmbeddingSource(key="missing", path=path, id_type="symbol")

    with pytest.raises(ValueError, match="missing/blank identifier"):
        read_static_embedding_table(source)

    table = read_static_embedding_table(source, drop_missing_ids=True)
    assert table.entity_ids == ("TP53",)
    assert table.n_missing_input_ids == 1
