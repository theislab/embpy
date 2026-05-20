"""Tests for the gene-expression-space metrics helpers.

The numpy helpers (mse, mae, r2_score, pearson_corr, spearman_corr,
deg_overlap_top_k) are exercised on toy data with closed-form expected
values so any future tweak to the implementation is caught.
"""

from __future__ import annotations

import numpy as np

from world_model.evaluation.metrics import (
    deg_overlap_top_k,
    mae,
    mse,
    pearson_corr,
    r2_score,
    spearman_corr,
)


def test_mse_zero_for_identical_arrays():
    a = np.linspace(0, 1, 32, dtype=np.float32)
    assert mse(a, a) == 0.0


def test_mse_matches_manual_formula():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    assert abs(mse(a, b) - (4.0 / 3.0)) < 1e-6


def test_mae_zero_for_identical_arrays():
    a = np.random.default_rng(0).normal(size=(16,)).astype(np.float32)
    assert mae(a, a) == 0.0


def test_r2_perfect_prediction():
    a = np.linspace(0, 1, 64, dtype=np.float32)
    assert r2_score(a, a) == 1.0


def test_r2_returns_nan_for_constant_target():
    a = np.zeros(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    out = r2_score(a, b)
    assert np.isnan(out)


def test_pearson_corr_perfect_linear():
    a = np.linspace(-1, 1, 64, dtype=np.float32)
    b = 3.0 * a + 2.0
    assert abs(pearson_corr(a, b) - 1.0) < 1e-5


def test_spearman_corr_monotone_nonlinear():
    a = np.linspace(0.1, 5.0, 64, dtype=np.float32)
    b = np.exp(a).astype(np.float32)
    assert abs(spearman_corr(a, b) - 1.0) < 1e-5


def test_deg_overlap_perfect_when_inputs_match():
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(128,)).astype(np.float32)
    assert deg_overlap_top_k(delta, delta, k=20) == 1.0


def test_deg_overlap_random_baseline():
    rng = np.random.default_rng(0)
    n_genes = 200
    real = rng.normal(size=(n_genes,)).astype(np.float32)
    pred = rng.normal(size=(n_genes,)).astype(np.float32)
    overlap = deg_overlap_top_k(real, pred, k=20)
    expected = 20 / n_genes  # random expectation
    assert overlap < 0.4  # very loose; the random sample is small
    assert overlap >= 0.0


def test_deg_overlap_handles_k_larger_than_array():
    delta = np.array([0.1, 0.5, -0.3], dtype=np.float32)
    overlap = deg_overlap_top_k(delta, delta, k=999)
    assert overlap == 1.0
