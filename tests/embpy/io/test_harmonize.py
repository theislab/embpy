"""harmonize: different-width results -> same width; variance recorded."""

from __future__ import annotations

import numpy as np

from embpy.io.harmonize import harmonize
from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def _result(n_dims, model, seed=0):
    rng = np.random.default_rng(seed)
    return EmbeddingResult(
        matrix=rng.normal(size=(20, n_dims)).astype(np.float32),
        entity_ids=tuple(f"ENSG{i}" for i in range(20)),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model=model),
    )


def test_two_widths_become_same_width():
    a = harmonize(_result(64, "modelA"), 8)
    b = harmonize(_result(128, "modelB"), 8)
    assert a.n_dims == 8 and b.n_dims == 8
    assert a.entity_ids == b.entity_ids


def test_provenance_records_variance_and_components():
    out = harmonize(_result(32, "m"), 5, random_state=7)
    assert out.provenance.harmonized_n_components == 5
    assert out.provenance.explained_variance_ratio is not None
    assert len(out.provenance.explained_variance_ratio) == 5
    assert out.provenance.random_state == 7


def test_ids_and_aliases_carried_over():
    res = EmbeddingResult(
        matrix=np.random.default_rng(0).normal(size=(10, 16)).astype(np.float32),
        entity_ids=tuple(f"g{i}" for i in range(10)),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="m"),
        aliases={"g0": {"gene_symbol": "TP53"}},
    )
    out = harmonize(res, 4)
    assert out.entity_ids == res.entity_ids
    assert out.aliases == res.aliases


def test_determinism():
    r = _result(32, "m", seed=1)
    o1 = harmonize(r, 6, random_state=42)
    o2 = harmonize(r, 6, random_state=42)
    np.testing.assert_allclose(o1.matrix, o2.matrix)
