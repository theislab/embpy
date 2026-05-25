from __future__ import annotations

import datetime as dt
from pathlib import Path

from world_model.utils.run_identity import (
    append_run_suffix,
    has_run_suffix,
    make_run_suffix,
    resolve_latest_suffixed_run_dir,
    strip_run_suffix,
)


def test_make_run_suffix_uses_slurm_or_local():
    now = dt.datetime(2026, 1, 2, 3, 4, 5)
    assert make_run_suffix(environ={"SLURM_JOB_ID": "123"}, now=now) == "__job123__20260102_030405"
    assert make_run_suffix(environ={}, now=now) == "__local__20260102_030405"


def test_append_strip_and_detect_suffix():
    suffixed = append_run_suffix("runs/world_model/single", "__job7__20260102_030405")
    assert suffixed.endswith("single__job7__20260102_030405")
    assert has_run_suffix(suffixed)
    assert strip_run_suffix(suffixed) == "single"
    assert append_run_suffix(suffixed, "__job8__20260102_030406") == suffixed


def test_resolve_latest_suffixed_run_dir(tmp_path: Path):
    base = tmp_path / "run"
    older = tmp_path / "run__job1__20260102_030405"
    newer = tmp_path / "run__job2__20260102_030406"
    older.mkdir()
    newer.mkdir()
    assert resolve_latest_suffixed_run_dir(base) == newer
    base.mkdir()
    assert resolve_latest_suffixed_run_dir(base) == base
