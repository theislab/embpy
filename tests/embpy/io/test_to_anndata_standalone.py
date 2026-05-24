"""Standalone AnnData export: placeholder .X, embeddings outside .X."""

from __future__ import annotations

import numpy as np
from scipy import sparse

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


def test_gene_entities_are_vars_and_x_is_placeholder():
    adata = to_anndata(_result())
    key = "X_emb__gene__m__pool_mean"
    assert list(adata.obs_names) == ["embpy_placeholder_obs"]
    assert list(adata.var_names) == ["ENSG1", "ENSG2", "ENSG3"]
    assert sparse.issparse(adata.X)
    assert adata.X.shape == (1, 3)
    assert adata.X.nnz == 0
    assert key in adata.varm
    np.testing.assert_allclose(adata.varm[key], _result().matrix)


def test_uns_carries_provenance_and_scheme():
    adata = to_anndata(_result())
    blk = adata.uns["embpy"]
    assert blk["entity_type"] == "gene"
    assert blk["id_scheme"] == "ensembl_gene_id"
    assert blk["provenance"]["model"] == "m"
    assert blk["placeholder_X"]["is_placeholder"] is True


def test_gene_aliases_go_into_var_not_index():
    adata = to_anndata(_result(aliases={"ENSG1": {"gene_symbol": "TP53"}}))
    assert "gene_symbol" in adata.var.columns
    assert adata.var.loc["ENSG1", "gene_symbol"] == "TP53"
    assert adata.var_names.name == "ensembl_gene_id"


def test_custom_key_standalone():
    adata = to_anndata(_result(), key="X_custom")
    assert "X_custom" in adata.varm
    assert "X_custom" in adata.uns["embpy"]["embeddings"]
