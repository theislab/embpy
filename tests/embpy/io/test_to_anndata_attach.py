"""Attach export: the obsm-vs-varm resolution rule and its failure modes."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from embpy.io.exporters import to_anndata
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(ids, n_dims=4):
    return EmbeddingResult(
        matrix=np.arange(len(ids) * n_dims, dtype=np.float32).reshape(len(ids), n_dims),
        entity_ids=tuple(ids),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="m"),
    )


def _make_target(obs_names, var_names):
    ad = AnnData(X=np.zeros((len(obs_names), len(var_names)), dtype=np.float32))
    ad.obs_names = list(obs_names)
    ad.var_names = list(var_names)
    return ad


def test_obs_aligned_goes_to_obsm():
    res = _result(["pertA", "pertB", "pertC"])
    tgt = _make_target(["pertA", "pertB", "pertC"], ["g1", "g2"])
    out = to_anndata(res, target=tgt)
    key = "X_emb__gene__m"
    assert key in out.obsm
    assert out.obsm[key].shape == (3, 4)
    np.testing.assert_allclose(out.obsm[key], res.matrix)


def test_var_aligned_goes_to_varm():
    res = _result(["pertA", "pertB", "pertC"])
    tgt = _make_target(["c1", "c2"], ["pertA", "pertB", "pertC"])
    out = to_anndata(res, target=tgt)
    key = "X_emb__gene__m"
    assert key in out.varm
    assert out.varm[key].shape == (3, 4)


def test_no_overlap_raises_with_counts():
    res = _result(["x", "y", "z"])
    tgt = _make_target(["a", "b"], ["g1", "g2"])
    with pytest.raises(ValueError, match=r"0/3 entities match target.obs_names"):
        to_anndata(res, target=tgt)


def test_both_axes_match_is_ambiguous_then_override():
    # entities present on BOTH axes -> auto must refuse, override resolves it.
    res = _result(["s1", "s2"])
    tgt = _make_target(["s1", "s2"], ["s1", "s2"])
    with pytest.raises(ValueError, match=r"Cannot auto-resolve attach axis"):
        to_anndata(res, target=tgt)
    out = to_anndata(res, target=tgt, attach_to="obs")
    assert "X_emb__gene__m" in out.obsm


def test_missing_error_vs_nan():
    res = _result(["pertA", "pertB", "pertC"])
    tgt = _make_target(["pertA", "pertB", "pertC", "pertD"], ["g1"])
    with pytest.raises(ValueError, match=r"pertD|no embedding"):
        to_anndata(res, target=tgt, missing="error")

    out = to_anndata(res, target=tgt, missing="nan")
    m = out.obsm["X_emb__gene__m"]
    assert m.shape == (4, 4)
    assert np.isnan(m[3]).all()  # pertD NaN-filled
    assert not np.isnan(m[:3]).any()  # A,B,C intact


def test_custom_key():
    res = _result(["pertA", "pertB"])
    tgt = _make_target(["pertA", "pertB"], ["g1"])
    out = to_anndata(res, target=tgt, key="X_myemb")
    assert "X_myemb" in out.obsm
