from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.store import EmbeddingStore


def _result() -> EmbeddingResult:
    return EmbeddingResult(
        matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        entity_ids=("ENSG1", "ENSG2"),
        entity_type="gene",
        id_scheme="ensembl_gene_id",
        provenance=EmbeddingProvenance(model="geneformer"),
        aliases={"ENSG1": {"symbol": "A"}, "ENSG2": {"symbol": "B"}},
    )


def test_add_result_stores_matrix_ids_and_metadata():
    store = EmbeddingStore()
    block = store.add_result(_result())

    assert block.key == "gene:geneformer"
    assert block.entity_ids == ("ENSG1", "ENSG2")
    assert block.n_dims == 2
    assert store.embedding("gene:geneformer").aliases["ENSG1"]["symbol"] == "A"
    assert store.entity_table("gene").index.tolist() == ["ENSG1", "ENSG2"]


def test_store_write_read_roundtrip_and_backed(tmp_path):
    store = EmbeddingStore.from_results(_result())
    store.add_relation(
        "gene_interacts_gene",
        pd.DataFrame({"source_id": ["ENSG1"], "target_id": ["ENSG2"], "weight": [0.7]}),
        source_type="gene",
        target_type="gene",
    )

    path = store.write(tmp_path / "test.emstore")
    loaded = EmbeddingStore.read(path)
    backed = EmbeddingStore.read(path, backed=True)

    assert loaded.keys() == ["gene:geneformer"]
    assert np.allclose(loaded.embedding("gene:geneformer").matrix, store.embedding("gene:geneformer").matrix)
    assert loaded.relation("gene_interacts_gene").n_edges == 1
    assert isinstance(backed.embedding("gene:geneformer").matrix, np.memmap)


def test_duplicate_and_invalid_matrices_raise():
    store = EmbeddingStore()
    with pytest.raises(ValueError, match="not unique"):
        store.add_embedding(
            "bad",
            np.ones((2, 2)),
            ["a", "a"],
            entity_type="gene",
            id_scheme="symbol",
        )
    with pytest.raises(ValueError, match="must be 2D"):
        store.add_embedding(
            "bad2",
            np.ones((2, 2, 2)),
            ["a", "b"],
            entity_type="gene",
            id_scheme="symbol",
        )
    with pytest.raises(ValueError, match="NaN/Inf"):
        store.add_embedding(
            "bad3",
            np.array([[np.nan]]),
            ["a"],
            entity_type="gene",
            id_scheme="symbol",
        )


def test_relation_registration_and_audit_reports_missing_targets():
    store = EmbeddingStore.from_results(_result())
    store.add_relation(
        "perturbation_targets_gene",
        pd.DataFrame({"source_id": ["pertA"], "target_id": ["ENSG_MISSING"]}),
        source_type="perturbation",
        target_type="gene",
    )

    audit = store.audit()

    assert {"missing_source_ids", "missing_target_ids"} <= set(audit["issue"])
    assert audit.loc[audit["issue"] == "missing_target_ids", "count"].iloc[0] == 1
