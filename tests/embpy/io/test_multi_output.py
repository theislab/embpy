from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.io.exporters import route_output
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(model: str, ids=("ENSG1", "ENSG2"), entity_type="gene"):
    return EmbeddingResult(
        matrix=np.arange(len(ids) * 3, dtype=np.float32).reshape(len(ids), 3),
        entity_ids=tuple(ids),
        entity_type=entity_type,
        id_scheme="ensembl_gene_id" if entity_type == "gene" else "canonical_smiles",
        provenance=EmbeddingProvenance(model=model, pooling="mean"),
    )


def test_multi_table_returns_keyed_frames_and_rejects_single_file(tmp_path):
    results = [_result("m1"), _result("m2")]
    out = route_output(results, output="table")
    assert isinstance(out, dict)
    assert list(out) == ["X_emb__gene__m1__pool_mean", "X_emb__gene__m2__pool_mean"]
    assert all(isinstance(v, pd.DataFrame) for v in out.values())

    with pytest.raises(ValueError, match=r"multiple embedding results.*single file path"):
        route_output(results, output="table", path=tmp_path / "all.npz")


def test_multi_table_writes_directory(tmp_path):
    results = [_result("m1"), _result("m2")]
    route_output(results, output="table", path=tmp_path)
    assert (tmp_path / "X_emb__gene__m1__pool_mean.npz").exists()
    assert (tmp_path / "X_emb__gene__m1__pool_mean.npz.meta.json").exists()
    assert (tmp_path / "X_emb__gene__m2__pool_mean.npz").exists()


def test_multi_anndata_standalone_stores_each_key_outside_x():
    out = route_output([_result("m1"), _result("m2")], output="anndata")
    assert isinstance(out, AnnData)
    assert set(out.varm.keys()) == {
        "X_emb__gene__m1__pool_mean",
        "X_emb__gene__m2__pool_mean",
    }
    assert out.X.nnz == 0
