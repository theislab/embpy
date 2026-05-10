"""Tests for the AnnData prep helpers consumed by cell-eval."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def expression_data():
    rng = np.random.default_rng(0)
    n_cells = 30
    n_genes = 6
    expression = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    labels = np.array(["non-targeting"] * 10 + ["P0"] * 10 + ["P1"] * 10)
    test_indices = np.arange(10, 30)
    return {
        "expression": expression,
        "labels": labels,
        "test_indices": test_indices,
        "test_perturbations": ["P0", "P1"],
        "gene_symbols": [f"g{i}" for i in range(n_genes)],
    }


def test_build_real_anndata_obs_and_vars(expression_data):
    pytest.importorskip("anndata")
    from embpy.world_model.evaluation.prep import build_real_anndata

    real = build_real_anndata(
        expression=expression_data["expression"],
        perturbation_labels=expression_data["labels"],
        test_indices=expression_data["test_indices"],
        test_perturbations=expression_data["test_perturbations"],
        gene_symbols=expression_data["gene_symbols"],
    )
    assert real.n_obs == 20
    assert real.n_vars == 6
    assert "perturbation" in real.obs.columns
    assert set(real.obs["perturbation"].unique().tolist()) == {"P0", "P1"}
    assert list(real.var_names) == expression_data["gene_symbols"]


def test_build_pred_anndata_aligns_to_real(expression_data):
    pytest.importorskip("anndata")
    from embpy.world_model.evaluation.prep import (
        build_pred_anndata,
        build_real_anndata,
    )

    real = build_real_anndata(
        expression=expression_data["expression"],
        perturbation_labels=expression_data["labels"],
        test_indices=expression_data["test_indices"],
        test_perturbations=expression_data["test_perturbations"],
        gene_symbols=expression_data["gene_symbols"],
    )
    n_genes = real.n_vars
    predictions = {
        "P0": np.full((n_genes,), 0.1, dtype=np.float32),
        "P1": np.full((n_genes,), 0.2, dtype=np.float32),
    }
    pred = build_pred_anndata(predictions, real)
    assert pred.n_obs == real.n_obs
    assert pred.n_vars == real.n_vars
    assert list(pred.var_names) == list(real.var_names)
    p0_mask = pred.obs["perturbation"].values == "P0"
    p1_mask = pred.obs["perturbation"].values == "P1"
    assert np.allclose(np.asarray(pred.X)[p0_mask], 0.1)
    assert np.allclose(np.asarray(pred.X)[p1_mask], 0.2)


def test_build_pred_anndata_handles_missing_predictions(expression_data):
    pytest.importorskip("anndata")
    from embpy.world_model.evaluation.prep import (
        build_pred_anndata,
        build_real_anndata,
    )

    real = build_real_anndata(
        expression=expression_data["expression"],
        perturbation_labels=expression_data["labels"],
        test_indices=expression_data["test_indices"],
        test_perturbations=expression_data["test_perturbations"],
        gene_symbols=expression_data["gene_symbols"],
    )
    n_genes = real.n_vars
    predictions = {"P0": np.full((n_genes,), 0.1, dtype=np.float32)}
    pred = build_pred_anndata(predictions, real)
    p1_mask = pred.obs["perturbation"].values == "P1"
    assert np.all(np.asarray(pred.X)[p1_mask] == 0.0)


def test_build_real_anndata_raises_when_no_test_cells(expression_data):
    pytest.importorskip("anndata")
    from embpy.world_model.evaluation.prep import build_real_anndata

    with pytest.raises(ValueError):
        build_real_anndata(
            expression=expression_data["expression"],
            perturbation_labels=expression_data["labels"],
            test_indices=np.array([0]),
            test_perturbations=["P0", "P1"],
            gene_symbols=expression_data["gene_symbols"],
        )
