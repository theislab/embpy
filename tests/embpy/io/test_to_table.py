"""to_table: index/column conventions + parquet & csv round-trips."""

from __future__ import annotations

import json

import numpy as np
import pytest

from embpy.io.exporters import to_table
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(aliases=None) -> EmbeddingResult:
    return EmbeddingResult(
        matrix=np.arange(6, dtype=np.float32).reshape(3, 2),
        entity_ids=("ENSG1", "ENSG2", "ENSG3"),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="m", pooling="mean"),
        aliases=aliases,
    )


def test_index_name_and_dim_columns():
    df = to_table(_result())
    assert df.index.name == "ensembl_gene_id"
    assert list(df.index) == ["ENSG1", "ENSG2", "ENSG3"]
    assert list(df.columns) == ["dim_0", "dim_1"]


def test_alias_columns_precede_dims_and_carry_values():
    aliases = {
        "ENSG1": {"gene_symbol": "TP53"},
        "ENSG2": {"gene_symbol": "MYC"},
        # ENSG3 intentionally missing -> None
    }
    df = to_table(_result(aliases))
    assert list(df.columns) == ["gene_symbol", "dim_0", "dim_1"]
    assert df.loc["ENSG1", "gene_symbol"] == "TP53"
    assert df.loc["ENSG3", "gene_symbol"] is None or df.loc["ENSG3", "gene_symbol"] != df.loc["ENSG3", "gene_symbol"]  # None/NaN


def test_parquet_roundtrip_with_sidecar(tmp_path):
    import pandas as pd

    res = _result()
    p = tmp_path / "emb.parquet"
    to_table(res, path=p, fmt="parquet")
    assert p.exists()
    side = tmp_path / "emb.parquet.meta.json"
    assert side.exists()

    back = pd.read_parquet(p)
    assert back.index.name == "ensembl_gene_id"
    assert list(back.columns) == ["dim_0", "dim_1"]
    np.testing.assert_allclose(back.to_numpy(), res.matrix)

    meta = json.loads(side.read_text())
    assert meta["entity_type"] == "gene"
    assert meta["id_scheme"] == "ensembl_gene_id"
    assert meta["provenance"]["model"] == "m"
    assert meta["n_entities"] == 3 and meta["n_dims"] == 2


def test_csv_roundtrip(tmp_path):
    import pandas as pd

    res = _result()
    p = tmp_path / "emb.csv"
    to_table(res, path=p, fmt="csv")
    assert (tmp_path / "emb.csv.meta.json").exists()

    back = pd.read_csv(p, index_col=0)
    assert back.index.name == "ensembl_gene_id"
    assert list(back.columns) == ["dim_0", "dim_1"]
    np.testing.assert_allclose(back.to_numpy(), res.matrix)


def test_bad_fmt_raises():
    with pytest.raises(ValueError, match=r"fmt must be"):
        to_table(_result(), fmt="hdf5")  # type: ignore[arg-type]
