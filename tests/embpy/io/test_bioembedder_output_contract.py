"""The output contract shared by BioEmbedder.embed (via route_output).

Tested at the ``route_output`` level so the contract (anndata-standalone
vs attach, table, the warn rule, harmonize_dim) is verified without
loading any model. ``BioEmbedder.embed`` is a thin wrapper that builds an
EmbeddingResult and calls this same function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.io.exporters import route_output
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(ids=("ENSG1", "ENSG2", "ENSG3"), n_dims=4):
    return EmbeddingResult(
        matrix=np.arange(len(ids) * n_dims, dtype=np.float32).reshape(len(ids), n_dims),
        entity_ids=tuple(ids),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="m"),
    )


def test_anndata_standalone_when_no_target():
    out = route_output(_result(), output="anndata")
    assert isinstance(out, AnnData)
    assert list(out.var_names) == ["ENSG1", "ENSG2", "ENSG3"]
    assert "X_emb__gene__m" in out.varm


def test_anndata_attach_when_target_given():
    tgt = AnnData(X=np.zeros((3, 2), dtype=np.float32))
    tgt.obs_names = ["ENSG1", "ENSG2", "ENSG3"]
    tgt.var_names = ["g1", "g2"]
    out = route_output(_result(), output="anndata", target=tgt)
    assert "X_emb__gene__m" in out.obsm


def test_table_output_returns_dataframe():
    out = route_output(_result(), output="table")
    assert isinstance(out, pd.DataFrame)
    assert out.index.name == "ensembl_gene_id"


def test_payload_output_returns_entity_aligned_metadata():
    out = route_output(_result(), output="payload")
    assert out["schema_version"] == "embpy.uns_embedding.v1"
    assert out["entity_ids"] == ["ENSG1", "ENSG2", "ENSG3"]
    assert out["id_scheme"] == "ensembl_gene_id"
    assert out["model"]["name"] == "m"
    assert out["matrix"].shape == (3, 4)


def test_table_ignores_target_with_warning(caplog):
    tgt = AnnData(X=np.zeros((3, 2), dtype=np.float32))
    import logging

    with caplog.at_level(logging.WARNING):
        out = route_output(_result(), output="table", target=tgt)
    assert isinstance(out, pd.DataFrame)
    assert any("ignores the provided target" in r.message for r in caplog.records)


def test_harmonize_dim_applied_before_export():
    out = route_output(_result(n_dims=4), output="table", harmonize_dim=2)
    assert list(out.columns) == ["dim_0", "dim_1"]  # projected to 2 dims


def test_bad_output_raises():
    with pytest.raises(ValueError, match=r"output must be 'anndata', 'table', or 'payload'"):
        route_output(_result(), output="zarr")  # type: ignore[arg-type]


def test_table_written_to_path(tmp_path):
    p = tmp_path / "out.npz"
    route_output(_result(), output="table", path=p)
    assert p.exists() and (tmp_path / "out.npz.meta.json").exists()


def test_bioembedder_embed_supports_perturbation_morphology_payload(monkeypatch):
    from embpy.embedder import BioEmbedder

    def _fake_batch(self, perturbations, **kwargs):
        assert kwargs["dataset"] == "hpa"
        assert kwargs["source"] == "subcell"
        return np.ones((1, 3), dtype=np.float32), [perturbations[0]]

    monkeypatch.setattr(
        BioEmbedder,
        "embed_perturbation_morphology_batch",
        _fake_batch,
    )

    out = BioEmbedder(device="cpu").embed(
        ["TP53", "MISSING"],
        entity_type="perturbation",
        model="subcell_mae_rybg",
        output="payload",
        morphology_dataset="hpa",
        morphology_source="subcell",
        max_images=1,
        verbose=False,
    )

    assert out["entity_type"] == "perturbation"
    assert out["id_scheme"] == "perturbation_label"
    assert out["entity_ids"] == ["TP53"]
    assert out["matrix"].shape == (1, 3)
    assert out["model_config"]["morphology_dataset"] == "hpa"
