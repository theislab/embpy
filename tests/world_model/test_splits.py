"""Determinism + correctness tests for train/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model.data.splits import (
    load_split,
    make_split,
    save_split,
    split_or_load,
    subsample_train_perturbations,
)


def _toy_labels(n_perts: int = 6, cells_per: int = 10) -> np.ndarray:
    out = ["non-targeting"] * cells_per
    for i in range(n_perts):
        out += [f"P{i}"] * cells_per
    return np.array(out)


def test_perturbation_split_is_disjoint_on_perts():
    labels = _toy_labels()
    spec = make_split(labels, control_label="non-targeting", split_by="perturbation",
                      train_fraction=0.5, seed=0)
    train_perts = set(spec.train_perturbations or [])
    test_perts = set(spec.test_perturbations or [])
    assert train_perts.isdisjoint(test_perts)
    assert train_perts | test_perts == {f"P{i}" for i in range(6)}


def test_split_is_deterministic():
    labels = _toy_labels()
    a = make_split(labels, control_label="non-targeting", seed=42)
    b = make_split(labels, control_label="non-targeting", seed=42)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.test_indices, b.test_indices)


def test_split_save_load_roundtrip(tmp_path: Path):
    labels = _toy_labels()
    spec = make_split(labels, control_label="non-targeting", seed=0)
    p = tmp_path / "split.npz"
    save_split(spec, p)
    loaded = load_split(p)
    np.testing.assert_array_equal(spec.train_indices, loaded.train_indices)
    np.testing.assert_array_equal(spec.test_indices, loaded.test_indices)
    assert loaded.split_by == "perturbation"
    assert loaded.train_perturbations == spec.train_perturbations


def test_split_or_load_caches(tmp_path: Path):
    labels = _toy_labels()
    p = tmp_path / "cache.npz"
    a = split_or_load(labels, cache_path=p, control_label="non-targeting", seed=7)
    assert p.exists()
    b = split_or_load(labels, cache_path=p, control_label="non-targeting", seed=7)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)


def test_cell_split_is_full_partition():
    labels = _toy_labels()
    spec = make_split(labels, control_label="non-targeting", split_by="cell",
                      train_fraction=0.7, seed=0)
    union = np.concatenate([spec.train_indices, spec.test_indices])
    assert len(set(union.tolist())) == labels.size


def test_subsample_keeps_fraction():
    labels = _toy_labels(n_perts=10)
    spec = make_split(labels, control_label="non-targeting", seed=0)
    sub = subsample_train_perturbations(spec, labels, fraction=0.2, seed=0)
    assert len(sub.train_perturbations or []) <= len(spec.train_perturbations or [])
    assert len(sub.train_perturbations or []) >= 1
