"""Tests for the dataset / preprocessing layer.

Avoids touching the real .h5ad files by constructing synthetic
expression matrices and a synthetic gene-embedding table in-memory.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from world_model.data.datasets.base import (
    GeneIndexer,
    PerturbationSequenceDataset,
)
from world_model.data.preprocessing import sequence_collate_fn

N_CELLS = 200
N_GENES = 64
PERTURBATIONS = ["GENE_A", "GENE_B", "GENE_C"]
CONTROL = "non-targeting"


def _make_synthetic() -> tuple[np.ndarray, np.ndarray, GeneIndexer]:
    rng = np.random.default_rng(0)
    x = rng.poisson(lam=2.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    labels = np.array(
        [CONTROL] * 50 + PERTURBATIONS * ((N_CELLS - 50) // len(PERTURBATIONS) + 1),
        dtype=object,
    )[:N_CELLS]
    indexer = GeneIndexer.from_symbols(PERTURBATIONS)
    return x, labels, indexer


def test_indexer_encodes_multi_gene_perturbations() -> None:
    indexer = GeneIndexer.from_symbols(PERTURBATIONS)
    assert indexer.encode("GENE_A", CONTROL) == [indexer.symbol_to_index["GENE_A"]]
    assert indexer.encode(CONTROL, CONTROL) == [0]
    multi = indexer.encode("GENE_A+GENE_B", CONTROL)
    assert sorted(multi) == sorted(
        [indexer.symbol_to_index["GENE_A"], indexer.symbol_to_index["GENE_B"]]
    )


def test_dataset_yields_correct_shapes() -> None:
    x, labels, indexer = _make_synthetic()
    ds = PerturbationSequenceDataset(
        expression=x,
        perturbation_labels=labels,
        indexer=indexer,
        sequence_length=4,
        stack_size=3,
        n_pert=2,
        control_label=CONTROL,
        rng=np.random.default_rng(0),
        n_sequences_per_epoch=8,
    )
    assert len(ds) == 8
    sample = ds[0]
    assert sample["obs_stack"].shape == (4, 3, N_GENES)
    assert sample["next_obs_stack"].shape == (4, 3, N_GENES)
    assert sample["action_indices"].shape == (4, 2)
    assert sample["next_expression"].shape == (4, N_GENES)


def test_collate_fn_stacks_correctly() -> None:
    x, labels, indexer = _make_synthetic()
    ds = PerturbationSequenceDataset(
        expression=x,
        perturbation_labels=labels,
        indexer=indexer,
        sequence_length=4,
        stack_size=3,
        n_pert=2,
        control_label=CONTROL,
        rng=np.random.default_rng(0),
        n_sequences_per_epoch=8,
    )
    samples = [ds[i] for i in range(2)]
    batch = sequence_collate_fn(samples)
    assert batch["obs_stack"].shape == (2, 4, 3, N_GENES)
    assert batch["action_indices"].shape == (2, 4, 2)


def test_incontext_action_similarity_support_ranks_nearest_embedding() -> None:
    x, labels, indexer = _make_synthetic()
    table = np.zeros((len(PERTURBATIONS) + 1, 3), dtype=np.float32)
    table[indexer.symbol_to_index["GENE_A"]] = [1.0, 0.0, 0.0]
    table[indexer.symbol_to_index["GENE_B"]] = [0.9, 0.1, 0.0]
    table[indexer.symbol_to_index["GENE_C"]] = [0.0, 1.0, 0.0]
    ds = PerturbationSequenceDataset(
        expression=x,
        perturbation_labels=labels,
        indexer=indexer,
        sequence_length=4,
        stack_size=3,
        n_pert=2,
        control_label=CONTROL,
        rng=np.random.default_rng(0),
        n_sequences_per_epoch=8,
        context_mode="incontext_set",
        incontext_support_size=1,
        incontext_support_strategy="action_similarity",
        action_embedding_table=table,
    )

    labels_out = ds._select_incontext_support_labels(
        label_pool=list(PERTURBATIONS),
        query_label="GENE_A",
        local=np.random.default_rng(0),
        size=1,
    )

    assert labels_out == ["GENE_B"]
