"""Tests for the action-encoder ablation grid loader and filters."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from world_model.evaluation.ablation.grid import (
    DEFAULT_ACTION_ENCODER_GRID,
    ActionEncoderSpec,
    filter_grid,
    load_grid,
    resolve_grid,
)


def test_default_grid_contents():
    keys = [s.key for s in DEFAULT_ACTION_ENCODER_GRID]
    assert keys == ["borzoi", "enformer", "nt_v2_500m", "esm2_650m", "minilm"]
    for spec in DEFAULT_ACTION_ENCODER_GRID:
        assert spec.id_type == "symbol"
        assert spec.region == "full"
        assert spec.pooling == "mean"
        assert isinstance(spec.extra_kwargs, dict)


def test_load_grid_yaml(tmp_path: Path):
    payload = {
        "grid": [
            {"key": "k1", "model_name": "m1"},
            {
                "key": "k2",
                "model_name": "m2",
                "id_type": "ensembl_id",
                "region": "exons",
                "pooling": "max",
                "extra_kwargs": {"chunk_size": 4},
                "notes": "custom",
            },
        ],
    }
    p = tmp_path / "grid.yaml"
    p.write_text(yaml.safe_dump(payload))
    specs = load_grid(p)
    assert [s.key for s in specs] == ["k1", "k2"]
    assert specs[1].region == "exons"
    assert specs[1].pooling == "max"
    assert specs[1].extra_kwargs == {"chunk_size": 4}


def test_load_grid_top_level_list(tmp_path: Path):
    payload = [{"key": "x", "model_name": "m"}]
    p = tmp_path / "grid.yaml"
    p.write_text(yaml.safe_dump(payload))
    specs = load_grid(p)
    assert [s.key for s in specs] == ["x"]


def test_load_grid_missing_required_field(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"grid": [{"key": "x"}]}))
    with pytest.raises(ValueError, match="missing required field"):
        load_grid(p)


def test_load_grid_duplicate_keys(tmp_path: Path):
    payload = {"grid": [
        {"key": "dup", "model_name": "a"},
        {"key": "dup", "model_name": "b"},
    ]}
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="Duplicate grid keys"):
        load_grid(p)


def test_filter_only_and_skip():
    specs = [
        ActionEncoderSpec(key="a", model_name="m"),
        ActionEncoderSpec(key="b", model_name="m"),
        ActionEncoderSpec(key="c", model_name="m"),
    ]
    out = filter_grid(specs, only="a,b")
    assert [s.key for s in out] == ["a", "b"]
    out = filter_grid(specs, skip="b")
    assert [s.key for s in out] == ["a", "c"]
    out = filter_grid(specs, only=["a", "b", "c"], skip=["c"])
    assert [s.key for s in out] == ["a", "b"]


def test_filter_only_unknown_raises():
    specs = [ActionEncoderSpec(key="a", model_name="m")]
    with pytest.raises(KeyError, match="unknown grid keys"):
        filter_grid(specs, only="b")


def test_filter_empty_result_raises():
    specs = [ActionEncoderSpec(key="a", model_name="m")]
    with pytest.raises(ValueError, match="empty grid"):
        filter_grid(specs, skip="a")


def test_resolve_grid_uses_default_when_path_none():
    specs = resolve_grid(grid_path=None, only=None, skip=None)
    assert [s.key for s in specs] == [s.key for s in DEFAULT_ACTION_ENCODER_GRID]
