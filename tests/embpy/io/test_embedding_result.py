"""Validation + immutability contract for :class:`EmbeddingResult`.

These pin the invariants the rest of the io layer relies on, so the
exporters never have to re-check shape / uniqueness / finiteness.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _prov() -> EmbeddingProvenance:
    # Deterministic: build directly rather than via create().
    return EmbeddingProvenance(model="test_model", pooling="mean")


def _result(matrix, ids, **kw) -> EmbeddingResult:
    return EmbeddingResult(
        matrix=matrix,
        entity_ids=ids,
        entity_type=kw.get("entity_type", "gene"),
        id_scheme=kw.get("id_scheme", "ensembl_gene_id"),
        provenance=_prov(),
        aliases=kw.get("aliases"),
    )


def test_valid_construction_and_shapes():
    res = _result(np.zeros((3, 4)), ("ENSG1", "ENSG2", "ENSG3"))
    assert res.n_entities == 3
    assert res.n_dims == 4
    assert res.dim_names == ["dim_0", "dim_1", "dim_2", "dim_3"]
    assert res.entity_ids == ("ENSG1", "ENSG2", "ENSG3")


def test_matrix_coerced_to_float32():
    res = _result(np.ones((2, 2), dtype=np.float64), ("a", "b"))
    assert res.matrix.dtype == np.float32
    assert res.matrix.flags["C_CONTIGUOUS"]


def test_duplicate_ids_raise_naming_the_duplicate():
    with pytest.raises(ValueError, match=r"unique.*'ENSG1'"):
        _result(np.zeros((3, 2)), ("ENSG1", "ENSG2", "ENSG1"))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match=r"length 2 but matrix has 3 rows"):
        _result(np.zeros((3, 2)), ("a", "b"))


def test_non_2d_matrix_raises():
    with pytest.raises(ValueError, match=r"must be 2D"):
        _result(np.zeros((5,)), ("a", "b", "c", "d", "e"))


def test_nan_raises_naming_row():
    m = np.zeros((3, 2))
    m[1, 0] = np.nan
    with pytest.raises(ValueError, match=r"NaN/Inf.*index 1.*'b'"):
        _result(m, ("a", "b", "c"))


def test_inf_raises():
    m = np.zeros((2, 2))
    m[0, 1] = np.inf
    with pytest.raises(ValueError, match=r"NaN/Inf"):
        _result(m, ("a", "b"))


def test_alias_key_not_in_ids_raises():
    with pytest.raises(ValueError, match=r"aliases key 'ENSGX'"):
        _result(np.zeros((2, 2)), ("a", "b"), aliases={"ENSGX": {"gene_symbol": "TP53"}})


def test_frozen_immutability():
    res = _result(np.zeros((2, 2)), ("a", "b"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.entity_type = "molecule"  # type: ignore[misc]


def test_provenance_create_autofills():
    p = EmbeddingProvenance.create("m", pooling="cls")
    assert p.model == "m"
    assert p.pooling == "cls"
    assert p.timestamp  # non-empty ISO timestamp
    d = p.to_dict()
    assert d["model"] == "m" and "timestamp" in d
