"""Standalone AnnData export: rows = entities, cols = dims, uns provenance."""

from __future__ import annotations

import numpy as np

from embpy.io.exporters import to_anndata
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(aliases=None):
    return EmbeddingResult(
        matrix=np.arange(6, dtype=np.float32).reshape(3, 2),
        entity_ids=("ENSG1", "ENSG2", "ENSG3"),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="m", pooling="mean"),
        aliases=aliases,
    )


def test_obs_are_entities_var_are_dims():
    adata = to_anndata(_result())
    assert list(adata.obs_names) == ["ENSG1", "ENSG2", "ENSG3"]
    assert list(adata.var_names) == ["dim_0", "dim_1"]
    np.testing.assert_allclose(np.asarray(adata.X), _result().matrix)


def test_uns_carries_provenance_and_scheme():
    adata = to_anndata(_result())
    blk = adata.uns["embpy"]
    assert blk["entity_type"] == "gene"
    assert blk["id_scheme"] == "ensembl_gene_id"
    assert blk["provenance"]["model"] == "m"


def test_aliases_go_into_obs_not_index():
    adata = to_anndata(_result(aliases={"ENSG1": {"gene_symbol": "TP53"}}))
    assert "gene_symbol" in adata.obs.columns
    assert adata.obs.loc["ENSG1", "gene_symbol"] == "TP53"
    assert adata.obs_names.name == "ensembl_gene_id"
