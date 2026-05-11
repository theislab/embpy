"""Smoke tests for every baseline.

We construct a tiny synthetic dataset where the perturbation effect is
a deterministic per-perturbation shift; this lets us assert that
baselines that *should* recover the effect (additive on seen perts,
linear when actions are informative) do so within tolerance, while
the identity baseline does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from world_model.evaluation.baselines import (
    AdditiveBaseline,
    BaselineTrainData,
    ControlMeanBaseline,
    IdentityBaseline,
    LinearRegressionBaseline,
    MeanBaseline,
)


CONTROL = "non-targeting"


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    n_genes = 16
    n_perts = 4
    cells_per = 32
    pert_labels = [CONTROL] + [f"P{i}" for i in range(n_perts)]
    rows = []
    labels = []
    actions: dict[str, np.ndarray] = {}
    pert_effects: dict[str, np.ndarray] = {}
    base = rng.normal(size=(n_genes,)).astype(np.float32)
    for p in pert_labels:
        if p == CONTROL:
            effect = np.zeros(n_genes, dtype=np.float32)
        else:
            effect = rng.normal(scale=0.5, size=(n_genes,)).astype(np.float32)
        pert_effects[p] = effect
        actions[p] = effect.copy()
        for _ in range(cells_per):
            rows.append(base + effect + rng.normal(scale=0.05, size=(n_genes,)).astype(np.float32))
            labels.append(p)
    expression = np.stack(rows, axis=0)
    perturbation_labels = np.array(labels)
    return {
        "expression": expression,
        "perturbation_labels": perturbation_labels,
        "actions": actions,
        "effects": pert_effects,
        "n_genes": n_genes,
        "control_mean": expression[perturbation_labels == CONTROL].mean(axis=0),
    }


def _train_data(td, train_indices):
    return BaselineTrainData(
        expression=td["expression"],
        perturbation_labels=td["perturbation_labels"],
        train_indices=train_indices,
        control_label=CONTROL,
        gene_symbols=[f"g{i}" for i in range(td["n_genes"])],
        perturbation_to_action=td["actions"],
    )


def _make_request(td, perts: list[str]):
    """Return ``(state, action_labels)`` with control_template as state."""
    n = len(perts)
    state = np.broadcast_to(td["control_mean"], (n, td["n_genes"])).astype(np.float32, copy=True)
    action = np.array(perts)
    return state, action


def test_identity_baseline(tiny_dataset):
    b = IdentityBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1"])
    out = b.predict(state, action)
    assert out.shape == (2, tiny_dataset["n_genes"])
    np.testing.assert_allclose(out[0], tiny_dataset["control_mean"])


def test_control_mean_baseline_returns_train_control_mean(tiny_dataset):
    b = ControlMeanBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1", "P2"])
    out = b.predict(state, action)
    assert out.shape == (3, tiny_dataset["n_genes"])
    np.testing.assert_allclose(out[0], out[1])


def test_control_mean_passes_controls_through(tiny_dataset):
    b = ControlMeanBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    custom_state = np.full((1, tiny_dataset["n_genes"]), 7.0, dtype=np.float32)
    out = b.predict(custom_state, np.array([CONTROL]))
    np.testing.assert_allclose(out[0], custom_state[0])


def test_mean_baseline(tiny_dataset):
    b = MeanBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1"])
    out = b.predict(state, action)
    assert out.shape == (2, tiny_dataset["n_genes"])
    np.testing.assert_allclose(out[0], out[1])


def test_additive_baseline_seen_pert(tiny_dataset):
    b = AdditiveBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1"])
    out = b.predict(state, action)
    expected_p0 = tiny_dataset["control_mean"] + tiny_dataset["effects"]["P0"]
    np.testing.assert_allclose(out[0], expected_p0, atol=0.1)


def test_additive_baseline_unseen_pert_falls_back_to_identity(tiny_dataset):
    b = AdditiveBaseline()
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["UNSEEN"])
    out = b.predict(state, action)
    np.testing.assert_allclose(out[0], state[0])


def test_linear_baseline_recovers_effect(tiny_dataset):
    b = LinearRegressionBaseline(alpha=1e-3, target="delta")
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1"])
    out = b.predict(state, action)
    expected_p0 = tiny_dataset["control_mean"] + tiny_dataset["effects"]["P0"]
    diff_linear = np.linalg.norm(out[0] - expected_p0)
    diff_identity = np.linalg.norm(state[0] - expected_p0)
    assert diff_linear < diff_identity, "Linear baseline must beat identity given true action embeddings."


def test_linear_baseline_next_state_target(tiny_dataset):
    b = LinearRegressionBaseline(alpha=1e-3, target="next_state")
    train_indices = np.arange(tiny_dataset["expression"].shape[0])
    b.fit(_train_data(tiny_dataset, train_indices))
    state, action = _make_request(tiny_dataset, ["P0", "P1"])
    out = b.predict(state, action)
    assert out.shape == (2, tiny_dataset["n_genes"])
    assert np.isfinite(out).all()
