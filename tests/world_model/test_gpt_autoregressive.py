"""Tests for the GPT-style autoregressive dynamics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from world_model.models.dynamics.gpt_autoregressive import GPTAutoregressiveDynamics

B, T, D = 2, 5, 16


def _build(use_action: bool = True) -> GPTAutoregressiveDynamics:
    return GPTAutoregressiveDynamics(
        d_model=D,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_sequence_length=4 * T,
        use_action_token=use_action,
    )


def test_dynamics_shape_with_actions() -> None:
    model = _build(use_action=True)
    s = torch.randn(B, T, D)
    a = torch.randn(B, T, D)
    out = model(s, a)
    assert out.shape == (B, T, D)


def test_dynamics_shape_without_actions() -> None:
    model = _build(use_action=False)
    s = torch.randn(B, T, D)
    out = model(s, None)
    assert out.shape == (B, T, D)


def test_dynamics_is_causal() -> None:
    """Mutating future tokens must not affect past predictions."""
    model = _build(use_action=True).eval()
    s = torch.randn(B, T, D)
    a = torch.randn(B, T, D)
    out_baseline = model(s, a)

    s_perturbed = s.clone()
    s_perturbed[:, T - 1] = torch.randn(B, D)
    out_perturbed = model(s_perturbed, a)

    # All but the last position must be byte-equal up to floating point.
    diff = (out_baseline[:, :-1] - out_perturbed[:, :-1]).abs().max()
    assert diff.item() < 1e-5


def test_dynamics_rejects_action_when_disabled() -> None:
    model = _build(use_action=True)
    s = torch.randn(B, T, D)
    with pytest.raises(ValueError):
        model(s, None)
