from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from embpy.pp.static_embeddings import (
    StaticEmbeddingSource,
    StaticEmbeddingStore,
    load_static_embedding_source_config,
    load_static_embedding_package,
    prepare_static_embedding_package,
    read_static_embedding_table,
    render_static_embedding_dataset_card,
    validate_static_embedding_package,
    write_static_embedding_dataset_card,
    write_static_embedding_package,
)


class FakeGeneResolver:
    def __init__(self, mapping: dict[str, str | None] | None = None) -> None:
        self.mapping = mapping or {
            "TP53": "ENSG00000141510",
            "MYC": "ENSG00000136997",
            "EGFR": "ENSG00000146648",
        }

    def symbols_to_ensembl_batch(self, symbols: list[str], organism: str = "human") -> dict[str, str | None]:
        return {symbol: self.mapping.get(symbol) for symbol in symbols}


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
    manifest = prepare_static_embedding_package(source_root, package_root, gene_resolver=FakeGeneResolver())

    assert sorted(manifest["embeddings"]) == ["genept_scaled"]
    assert manifest["species_keys"] == ["human_9606"]
    assert (package_root / "embeddings" / "genept_scaled" / "values.zarr").is_dir()
    assert (package_root / "embeddings" / "genept_scaled" / "metadata" / "index.parquet").is_file()
    assert (package_root / "embeddings" / "genept_scaled" / "metadata" / "uns.json").is_file()

    validation = validate_static_embedding_package(package_root)
    assert validation[0].key == "genept_scaled"
    assert validation[0].n_entities == 2
    assert validation[0].n_dims == 2
    assert validation[0].id_type == "ensembl_id"

    store = load_static_embedding_package(package_root, key="genept_scaled")
    assert isinstance(store, StaticEmbeddingStore)
    assert store.id_type == "ensembl_id"
    assert store.metadata["species"] == "human"
    assert store.metadata["taxonomy_id"] == "9606"
    assert store.metadata["species_key"] == "human_9606"
    assert list(store.index.columns) == ["entity_id", "source_id", "gene_symbol"]
    assert np.array_equal(store.get("ENSG00000141510"), np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(
        store.get("ENSG00000141510", id_type="ensembl_gene_id"),
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert np.array_equal(store.get("TP53", id_type="symbol"), np.array([1.0, 2.0], dtype=np.float32))

    frame = store.query(["TP53", "MYC"], id_type="symbol")
    assert list(frame.index) == ["TP53", "MYC"]
    assert frame.index.name == "symbol"
    assert list(frame.columns) == ["dim_0", "dim_1"]


def test_static_embedding_query_missing_policies(tmp_path):
    source = StaticEmbeddingSource(
        key="toy",
        path=tmp_path / "toy.csv",
        id_type="symbol",
    )
    pd.DataFrame([[1.0, 2.0]], index=pd.Index(["TP53"]), columns=["0", "1"]).to_csv(source.path)

    table = read_static_embedding_table(source, harmonize_ids=False)
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
        read_static_embedding_table(source, harmonize_ids=False)

    table = read_static_embedding_table(source, duplicate_policy="first", harmonize_ids=False)
    assert table.entity_ids == ("TP53",)
    assert table.n_duplicate_input_ids == 1


def test_read_static_embedding_table_missing_ids_are_explicit(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("gene_id,0,1\nTP53,1,2\n,3,4\n")
    source = StaticEmbeddingSource(key="missing", path=path, id_type="symbol")

    with pytest.raises(ValueError, match="missing/blank identifier"):
        read_static_embedding_table(source, harmonize_ids=False)

    table = read_static_embedding_table(source, drop_missing_ids=True, harmonize_ids=False)
    assert table.entity_ids == ("TP53",)
    assert table.n_missing_input_ids == 1


def test_read_static_embedding_table_explicit_id_and_metadata_columns(tmp_path):
    path = tmp_path / "messy.csv"
    pd.DataFrame(
        {
            "gene": ["TP53", "MYC"],
            "description": ["tumor protein p53", "MYC proto-oncogene"],
            "dim_0": [1.0, 3.0],
            "dim_1": [2.0, 4.0],
        }
    ).to_csv(path, index=False)
    source = StaticEmbeddingSource(
        key="messy",
        path=path,
        id_type="symbol",
        id_column="gene",
        metadata_columns=("description",),
    )

    table = read_static_embedding_table(source, harmonize_ids=False)

    assert table.entity_ids == ("TP53", "MYC")
    assert np.array_equal(table.matrix[0], np.array([1.0, 2.0], dtype=np.float32))
    assert table.alias_columns["description"] == ("tumor protein p53", "MYC proto-oncogene")

    package_root = tmp_path / "package"
    write_static_embedding_package(table, package_root)
    store = load_static_embedding_package(package_root, key="messy")
    assert list(store.index.columns) == ["entity_id", "description"]


def test_read_static_embedding_table_nonnumeric_error_suggests_layout_flags(tmp_path):
    path = tmp_path / "messy.csv"
    pd.DataFrame(
        {
            "gene": ["TP53"],
            "description": ["not an embedding dimension"],
            "dim_0": [1.0],
        }
    ).to_csv(path, index=False)
    source = StaticEmbeddingSource(key="messy", path=path, id_column="gene", id_type="symbol")

    with pytest.raises(ValueError, match="metadata_columns"):
        read_static_embedding_table(source, harmonize_ids=False)


def test_read_static_embedding_table_nan_policy_drop_rows(tmp_path):
    path = tmp_path / "nan.csv"
    pd.DataFrame(
        [[1.0, 2.0], [np.nan, 4.0]],
        index=pd.Index(["TP53", "MYC"]),
        columns=["0", "1"],
    ).to_csv(path)
    source = StaticEmbeddingSource(key="nan", path=path, id_type="symbol")

    with pytest.raises(ValueError, match="column='0'"):
        read_static_embedding_table(source, harmonize_ids=False)

    table = read_static_embedding_table(source, nan_policy="drop-rows", harmonize_ids=False)
    assert table.entity_ids == ("TP53",)
    assert table.n_nan_input_rows == 1
    assert table.n_nan_input_values == 1


def test_read_static_embedding_table_nan_policy_fill_zero(tmp_path):
    path = tmp_path / "nan.csv"
    pd.DataFrame(
        [[1.0, 2.0], [np.nan, 4.0]],
        index=pd.Index(["TP53", "MYC"]),
        columns=["0", "1"],
    ).to_csv(path)
    source = StaticEmbeddingSource(key="nan", path=path, id_type="symbol")

    table = read_static_embedding_table(source, nan_policy="fill-zero", harmonize_ids=False)
    assert table.entity_ids == ("TP53", "MYC")
    assert np.array_equal(table.matrix[1], np.array([0.0, 4.0], dtype=np.float32))


def test_read_static_embedding_table_harmonizes_symbols_to_ensembl(tmp_path):
    path = tmp_path / "symbols.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["TP53", "MYC"]),
        columns=["0", "1"],
    ).to_csv(path)
    source = StaticEmbeddingSource(key="symbols", path=path, id_type="symbol")

    table = read_static_embedding_table(source, gene_resolver=FakeGeneResolver())

    assert table.entity_ids == ("ENSG00000141510", "ENSG00000136997")
    assert table.id_type == "ensembl_id"
    assert table.source_id_type == "symbol"
    assert table.target_id_type == "ensembl_id"
    assert table.id_harmonized is True
    assert table.alias_columns["source_id"] == ("TP53", "MYC")
    assert table.alias_columns["gene_symbol"] == ("TP53", "MYC")


def test_read_static_embedding_table_harmonization_drops_unresolved_and_duplicates(tmp_path):
    path = tmp_path / "symbols.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=pd.Index(["A", "B", "A_ALIAS"]),
        columns=["0", "1"],
    ).to_csv(path)
    source = StaticEmbeddingSource(key="symbols", path=path, id_type="symbol")
    resolver = FakeGeneResolver({"A": "ENSGA", "B": None, "A_ALIAS": "ENSGA"})

    table = read_static_embedding_table(source, gene_resolver=resolver)

    assert table.entity_ids == ("ENSGA",)
    assert np.array_equal(table.matrix[0], np.array([1.0, 2.0], dtype=np.float32))
    assert table.n_unresolved_harmonization_ids == 1
    assert table.n_duplicate_harmonized_ids == 1
    assert table.alias_columns["gene_symbol"] == ("A",)

    with pytest.raises(ValueError, match="could not be harmonized"):
        read_static_embedding_table(source, gene_resolver=resolver, unresolved_id_policy="error")


def test_prepare_static_embedding_package_from_source_config(tmp_path):
    source_path = tmp_path / "custom.tsv"
    pd.DataFrame(
        {
            "gene_id": ["TP53", "MYC"],
            "batch": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        }
    ).to_csv(source_path, sep="\t", index=False)
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "key": "custom",
                        "path": "custom.tsv",
                        "id_column": "gene_id",
                        "id_type": "symbol",
                        "sep": "\t",
                        "metadata_columns": ["batch"],
                        "metadata": {
                            "species": "mouse",
                            "taxonomy_id": "10090",
                            "source": "unit-test",
                        },
                    }
                ]
            }
        )
    )

    sources = load_static_embedding_source_config(config_path)
    assert sources[0].key == "custom"
    assert sources[0].path == source_path
    assert sources[0].species == "mouse"
    assert sources[0].taxonomy_id == "10090"

    package_root = tmp_path / "package"
    manifest = prepare_static_embedding_package(
        None,
        package_root,
        source_config=config_path,
        harmonize_ids=False,
    )

    assert sorted(manifest["embeddings"]) == ["custom"]
    assert manifest["species_keys"] == ["mouse_10090"]
    assert manifest["planned_embeddings"][0]["species_key"] == "mouse_10090"
    store = load_static_embedding_package(package_root, key="custom")
    assert store.metadata["species"] == "mouse"
    assert store.metadata["taxonomy_id"] == "10090"
    assert store.metadata["species_key"] == "mouse_10090"
    assert store.uns["embpy_static_embedding"]["species_key"] == "mouse_10090"
    assert list(store.entity_ids) == ["TP53", "MYC"]
    assert list(store.index["batch"]) == ["a", "b"]


def test_source_config_rejects_species_taxonomy_mismatch(tmp_path):
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "key": "bad_species",
                        "path": "custom.tsv",
                        "species": "human",
                        "taxonomy_id": "10090",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="species and taxonomy_id"):
        load_static_embedding_source_config(config_path)


def test_static_embedding_dataset_card_from_manifest(tmp_path):
    source_root = tmp_path / "sources"
    source_dir = source_root / "genept" / "scaled"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["TP53"]),
        columns=["0", "1"],
    ).to_csv(source_dir / "embeddings_3072.csv")

    package_root = tmp_path / "package"
    manifest = prepare_static_embedding_package(source_root, package_root, gene_resolver=FakeGeneResolver())

    markdown = render_static_embedding_dataset_card(manifest, repo_id="theislab/Embpy_Data")
    assert "pretty_name: Embpy Static Embeddings" in markdown
    assert "HFHandler(\"theislab/Embpy_Data\")" in markdown
    assert "| genept_scaled | gene | human_9606 | ensembl_id | 1 | 2 |" in markdown
    assert str(tmp_path) not in markdown

    card_path = write_static_embedding_dataset_card(package_root, repo_id="theislab/Embpy_Data")
    assert card_path == package_root / "README.md"
    assert "Species keys: `human_9606`" in card_path.read_text()
    with pytest.raises(FileExistsError, match="Dataset card already exists"):
        write_static_embedding_dataset_card(package_root)


def test_read_static_embedding_table_hdf5_string_source(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "9606.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("embeddings", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))
        handle.create_dataset(
            "proteins",
            data=np.array([b"9606.ENSP000001", b"9606.ENSP000002"], dtype=object),
        )
        meta = handle.create_group("metadata")
        meta.attrs["embedding_dim"] = 2

    source = StaticEmbeddingSource(
        key="string_node2vec_9606",
        path=path,
        entity_type="protein",
        id_type="string_protein_id",
        h5_embedding_dataset="embeddings",
        h5_id_dataset="proteins",
    )
    table = read_static_embedding_table(source)

    assert table.entity_ids == ("9606.ENSP000001", "9606.ENSP000002")
    assert table.entity_type == "protein"
    assert table.id_type == "string_protein_id"
    assert table.target_id_type == "source"
    assert table.species == "human"
    assert table.taxonomy_id == "9606"
    assert table.species_key == "human_9606"
    assert table.alias_columns["ensembl_protein_id"] == ("ENSP000001", "ENSP000002")
    assert np.array_equal(table.matrix[1], np.array([3.0, 4.0], dtype=np.float32))


def test_prepare_static_embedding_package_discovers_known_string_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    source_dir = tmp_path / "precomputed_embeddings_string" / "node2vec" / "node2vec"
    source_dir.mkdir(parents=True)
    with h5py.File(source_dir / "9606.h5", "w") as handle:
        handle.create_dataset("embeddings", data=np.array([[1.0, 2.0]], dtype=np.float16))
        handle.create_dataset("proteins", data=np.array([b"9606.ENSP000001"], dtype=object))

    package_root = tmp_path / "package"
    manifest = prepare_static_embedding_package(tmp_path, package_root)

    assert sorted(manifest["embeddings"]) == ["string_node2vec_9606"]
    assert manifest["species_keys"] == ["human_9606"]
    store = load_static_embedding_package(package_root, key="string_node2vec_9606")
    assert store.entity_type == "protein"
    assert store.id_type == "string_protein_id"
    assert store.metadata["taxonomy_id"] == "9606"
    assert store.metadata["species_key"] == "human_9606"
    assert np.array_equal(store.get("9606.ENSP000001"), np.array([1.0, 2.0], dtype=np.float32))


def test_prepare_static_embedding_package_requires_explicit_string_species(tmp_path):
    h5py = pytest.importorskip("h5py")
    human_dir = tmp_path / "precomputed_embeddings_string" / "node2vec" / "node2vec"
    mouse_dir = human_dir
    human_dir.mkdir(parents=True)
    for species in ("9606", "10090"):
        with h5py.File(mouse_dir / f"{species}.h5", "w") as handle:
            handle.create_dataset("embeddings", data=np.array([[1.0, 2.0]], dtype=np.float16))
            handle.create_dataset("proteins", data=np.array([f"{species}.ENSP000001".encode()], dtype=object))

    default_package = tmp_path / "package_default"
    default_manifest = prepare_static_embedding_package(tmp_path, default_package)
    assert sorted(default_manifest["embeddings"]) == ["string_node2vec_9606"]

    explicit_package = tmp_path / "package_explicit"
    explicit_manifest = prepare_static_embedding_package(
        tmp_path,
        explicit_package,
        string_species=["10090"],
    )
    assert sorted(explicit_manifest["embeddings"]) == ["string_node2vec_10090", "string_node2vec_9606"]
    assert explicit_manifest["species_keys"] == ["human_9606", "mouse_10090"]
    mouse_store = load_static_embedding_package(explicit_package, key="string_node2vec_10090")
    assert mouse_store.metadata["species"] == "mouse"
    assert mouse_store.metadata["taxonomy_id"] == "10090"
    assert mouse_store.metadata["species_key"] == "mouse_10090"

    with pytest.raises(FileNotFoundError, match="Requested STRING species"):
        prepare_static_embedding_package(tmp_path, tmp_path / "package_missing", string_species=["999999"])
