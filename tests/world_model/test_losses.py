"""Tests for loss functions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from world_model.training.losses import (
    delta_mse,
    gaussian_nll,
    info_nce,
    latent_mse,
)


def test_latent_mse_zero_when_equal() -> None:
    x = torch.randn(8, 16)
    assert latent_mse(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_delta_mse_matches_latent_mse_when_basal_zero() -> None:
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    basal = torch.zeros(4, 8)
    a = delta_mse(basal, pred, target)
    b = latent_mse(pred, target)
    assert torch.allclose(a, b, atol=1e-6)


def test_gaussian_nll_decreases_with_log_var() -> None:
    target = torch.zeros(4, 8)
    mu = torch.zeros(4, 8)
    nll_low = gaussian_nll(target, mu, torch.full_like(mu, -2.0))
    nll_high = gaussian_nll(target, mu, torch.full_like(mu, 2.0))
    assert nll_low < nll_high


def test_info_nce_diagonal_optimum() -> None:
    """When pred == target, InfoNCE should be near its lower bound (~0).

    Use orthogonal rows so off-diagonal cosine similarities are exactly
    lower than the matched diagonal; random vectors can occasionally
    produce close negatives and make this test depend on luck.
    """
    x = torch.eye(8, 16)
    val = info_nce(x, x, temperature=0.1).item()
    assert val < 1e-2
