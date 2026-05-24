from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.io.normalize import normalize_embedding_input


def test_list_numpy_series_inputs():
    assert normalize_embedding_input(["TP53", "MYC"]).identifiers == ("TP53", "MYC")
    assert normalize_embedding_input(np.array(["TP53", "MYC"])).identifiers == ("TP53", "MYC")
    assert normalize_embedding_input(pd.Series(["TP53", "MYC"], name="gene")).id_column == "gene"


def test_dataframe_requires_identifier_column_when_ambiguous():
    df = pd.DataFrame({"gene": ["TP53"], "name": ["tumor protein p53"]})
    with pytest.raises(ValueError, match=r"ambiguous table identifier column.*gene.*name"):
        normalize_embedding_input(df)

    norm = normalize_embedding_input(df, identifier_column="gene")
    assert norm.identifiers == ("TP53",)
    assert norm.alias_columns["name"] == ("tumor protein p53",)


def test_file_paths_csv_tsv_parquet(tmp_path):
    df = pd.DataFrame({"gene": ["TP53", "MYC"]})
    csv = tmp_path / "genes.csv"
    tsv = tmp_path / "genes.tsv"
    pq = tmp_path / "genes.parquet"
    df.to_csv(csv, index=False)
    df.to_csv(tsv, sep="\t", index=False)
    df.to_parquet(pq)

    assert normalize_embedding_input(csv).identifiers == ("TP53", "MYC")
    assert normalize_embedding_input(tsv).identifiers == ("TP53", "MYC")
    assert normalize_embedding_input(pq).identifiers == ("TP53", "MYC")


def test_invalid_path_and_suffix_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"input loading: input path does not exist"):
        normalize_embedding_input(tmp_path / "missing.csv")

    bad = tmp_path / "ids.txt"
    bad.write_text("TP53\n")
    with pytest.raises(ValueError, match=r"input loading: unsupported input file suffix"):
        normalize_embedding_input(bad)


def test_invalid_numpy_shape_and_dtype_errors():
    with pytest.raises(ValueError, match=r"NumPy array must be 1D"):
        normalize_embedding_input(np.zeros((2, 2, 2), dtype=object))
    with pytest.raises(ValueError, match=r"identifiers must have string"):
        normalize_embedding_input(np.array([1, 2, 3]))


def test_anndata_sources():
    adata = AnnData(X=np.zeros((2, 3), dtype=np.float32))
    adata.obs_names = ["cell1", "cell2"]
    adata.var_names = ["ENSG1", "ENSG2", "ENSG3"]
    adata.obs["perturbation"] = ["drugA", "drugB"]
    adata.var["symbol"] = ["TP53", "MYC", "EGFR"]

    assert normalize_embedding_input(adata, entity_type="molecule").identifiers == ("cell1", "cell2")
    assert normalize_embedding_input(adata, entity_type="gene").identifiers == (
        "ENSG1",
        "ENSG2",
        "ENSG3",
    )
    assert normalize_embedding_input(adata, obs_column="perturbation").identifiers == (
        "drugA",
        "drugB",
    )
    assert normalize_embedding_input(adata, var_column="symbol").identifiers == (
        "TP53",
        "MYC",
        "EGFR",
    )
