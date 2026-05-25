"""Run-directory identity helpers for world-model jobs."""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Mapping
from pathlib import Path

_RUN_SUFFIX_RE = re.compile(r"__(?:job(?P<jobid>[^_]+)|local)__(?P<ts>\d{8}_\d{6})$")


def make_run_suffix(
    *,
    environ: Mapping[str, str] | None = None,
    now: dt.datetime | None = None,
) -> str:
    """Return ``__job{id}__YYYYmmdd_HHMMSS`` or ``__local__YYYYmmdd_HHMMSS``."""
    env = os.environ if environ is None else environ
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
    job_id = (env.get("SLURM_JOB_ID") or "").strip()
    return f"__job{job_id}__{stamp}" if job_id else f"__local__{stamp}"


def has_run_suffix(path: str | Path) -> bool:
    """Whether the final path component already has a world-model run suffix."""
    return bool(_RUN_SUFFIX_RE.search(Path(path).name))


def append_run_suffix(path: str | Path, suffix: str) -> str:
    """Append ``suffix`` to a run directory path unless it is already suffixed."""
    path_str = str(path)
    if has_run_suffix(path_str):
        return path_str
    return f"{path_str}{suffix}"


def strip_run_suffix(name_or_path: str | Path) -> str:
    """Strip a trailing job/local suffix from a run directory name."""
    name = Path(name_or_path).name
    return _RUN_SUFFIX_RE.sub("", name)


def resolve_latest_suffixed_run_dir(base: str | Path) -> Path:
    """Resolve ``base`` or the newest ``base__job*`` / ``base__local*`` dir.

    This is useful for chained jobs that receive a logical run directory from
    a submit script while the training job itself wrote a suffixed directory.
    """
    base_path = Path(base)
    if base_path.exists():
        return base_path
    parent = base_path.parent if str(base_path.parent) else Path(".")
    matches = [p for p in parent.glob(f"{base_path.name}__*") if p.is_dir()]
    if not matches:
        return base_path
    return max(matches, key=lambda p: p.stat().st_mtime)


__all__ = [
    "append_run_suffix",
    "has_run_suffix",
    "make_run_suffix",
    "resolve_latest_suffixed_run_dir",
    "strip_run_suffix",
]
