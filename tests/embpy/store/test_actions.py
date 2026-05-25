from __future__ import annotations

import numpy as np
import pytest

from embpy.resources.gene.control import ControlPolicy
from embpy.store.actions import (
    ActionStatus,
    compile_action_table,
    control_sentinel_vector,
    split_targets,
)


def _genes() -> tuple[np.ndarray, list[str]]:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.25, 0.25]], dtype=np.float32)
    return matrix, ["g1", "g2", "g3"]


def test_split_targets_handles_separators_and_singletons():
    assert split_targets("g1+g2") == ["g1", "g2"]
    assert split_targets("g1, g2 ; g3 / g4 | g5") == ["g1", "g2", "g3", "g4", "g5"]
    assert split_targets("g1") == ["g1"]


def test_control_sentinel_vector_is_deterministic_nonzero_and_scaled():
    v1 = control_sentinel_vector(8, seed=0)
    v2 = control_sentinel_vector(8, seed=0)
    assert v1.dtype == np.float32
    assert v1.shape == (8,)
    assert np.array_equal(v1, v2)  # deterministic across calls
    assert np.any(v1 != 0.0)  # non-zero: never confused with an UNRESOLVED row
    assert float(np.linalg.norm(v1)) == pytest.approx(float(np.sqrt(8)), rel=1e-5)
    assert not np.array_equal(v1, control_sentinel_vector(8, seed=1))
    with pytest.raises(ValueError, match="dim must be > 0"):
        control_sentinel_vector(0)


def test_compile_action_table_mean_and_sum_pooling():
    matrix, ids = _genes()
    mean_tbl = compile_action_table(["g1", "g2", "g1+g2"], matrix, ids)
    assert list(mean_tbl.table.index) == ["g1", "g2", "g1+g2"]
    assert mean_tbl.table.loc["g1+g2"].to_numpy().tolist() == pytest.approx([0.5, 0.5])
    assert mean_tbl.statuses == {
        "g1": ActionStatus.RESOLVED.value,
        "g2": ActionStatus.RESOLVED.value,
        "g1+g2": ActionStatus.RESOLVED.value,
    }
    assert mean_tbl.n_resolved == 3

    sum_tbl = compile_action_table(["g1+g2"], matrix, ids, aggregation="sum")
    assert sum_tbl.table.loc["g1+g2"].to_numpy().tolist() == pytest.approx([1.0, 1.0])


def test_compile_action_table_controls_use_sentinel():
    matrix, ids = _genes()
    out = compile_action_table(
        ["g1", "non-targeting", "DMSO"],
        matrix,
        ids,
        control_policy=ControlPolicy.default(),
        control_values=["DMSO"],
    )
    assert out.statuses["non-targeting"] == ActionStatus.CONTROL.value
    assert out.statuses["DMSO"] == ActionStatus.CONTROL.value
    assert out.statuses["g1"] == ActionStatus.RESOLVED.value
    sentinel = control_sentinel_vector(2, seed=0)
    assert out.table.loc["non-targeting"].to_numpy().tolist() == pytest.approx(sentinel.tolist())
    assert out.n_control == 2


def test_compile_action_table_mixed_drops_control_component():
    matrix, ids = _genes()
    out = compile_action_table(["g1+non-targeting"], matrix, ids, control_policy=ControlPolicy.default())
    # The mixed label keeps only its gene component g1.
    assert out.statuses["g1+non-targeting"] == ActionStatus.RESOLVED.value
    assert out.table.loc["g1+non-targeting"].to_numpy().tolist() == pytest.approx([1.0, 0.0])


def test_compile_action_table_unresolved_raise_zero_control():
    matrix, ids = _genes()
    with pytest.raises(ValueError, match="no target embeddings"):
        compile_action_table(["ghost"], matrix, ids)

    zero = compile_action_table(["ghost"], matrix, ids, on_unresolved="zero")
    assert zero.statuses["ghost"] == ActionStatus.UNRESOLVED.value
    assert zero.table.loc["ghost"].to_numpy().tolist() == [0.0, 0.0]
    assert zero.missing_targets["ghost"] == ["ghost"]
    assert zero.unresolved_conditions == ["ghost"]

    ctrl = compile_action_table(["ghost"], matrix, ids, on_unresolved="control")
    assert ctrl.statuses["ghost"] == ActionStatus.CONTROL.value
    assert np.any(ctrl.table.loc["ghost"].to_numpy() != 0.0)


def test_compile_action_table_partial_combo_resolves_and_records_missing():
    matrix, ids = _genes()
    out = compile_action_table(["g1+ghost"], matrix, ids)
    assert out.statuses["g1+ghost"] == ActionStatus.RESOLVED.value
    assert out.table.loc["g1+ghost"].to_numpy().tolist() == pytest.approx([1.0, 0.0])
    assert out.missing_targets["g1+ghost"] == ["ghost"]


def test_compile_action_table_relation_overrides_splitting():
    matrix, ids = _genes()
    out = compile_action_table(["pertA"], matrix, ids, relation={"pertA": ["g1", "g2"]})
    assert out.table.loc["pertA"].to_numpy().tolist() == pytest.approx([0.5, 0.5])


def test_compile_action_table_dedupes_and_counts():
    matrix, ids = _genes()
    out = compile_action_table(["g1", "g1", "g2"], matrix, ids)
    assert list(out.table.index) == ["g1", "g2"]
    assert out.counts() == {"RESOLVED": 2, "CONTROL": 0, "UNRESOLVED": 0}


def test_compile_action_table_validates_inputs():
    matrix, ids = _genes()
    with pytest.raises(ValueError, match="must be 2D"):
        compile_action_table(["g1"], np.ones((3,), dtype=np.float32), ids)
    with pytest.raises(ValueError, match="target_ids has length"):
        compile_action_table(["g1"], matrix, ["only_one"])
    with pytest.raises(ValueError, match="aggregation must be"):
        compile_action_table(["g1"], matrix, ids, aggregation="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_unresolved must be"):
        compile_action_table(["g1"], matrix, ids, on_unresolved="silent")  # type: ignore[arg-type]
