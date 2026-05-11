"""Byte-equivalent regression for the smoke pipeline across the Part C split.

Skipped unless invoked with ``--run-smoke-regression`` (see conftest).
The flag exists so CI does not pay the smoke-pipeline cost on every
push, but a release gate can demand byte equivalence against the
pre-split snapshot.

Workflow
--------

1. Capture the pre-split snapshot (one-shot, against commit ``f1b51cb``)::

       git checkout f1b51cb -- src/embpy/world_model    # pre-split tree
       pixi run -e gpu python -m embpy.world_model.scripts.smoke_test \\
           --config src/embpy/world_model/configs/experiments/smoke.yaml \\
           --output-dir tests/_snapshots/pre_split/smoke
       git checkout HEAD -- src/embpy/world_model       # back to the split tree

   then commit ``tests/_snapshots/pre_split/smoke/``.

2. After the split, run::

       pixi run -e gpu pytest tests/world_model/test_post_split_smoke.py \\
           --run-smoke-regression

   This re-runs the smoke pipeline on the current tree, then compares
   CSV / JSON outputs byte-for-byte and PNG outputs pixel-for-pixel
   against the snapshot. Any divergence is a failure.

The test skips with a helpful message if the snapshot does not exist
yet (initial bootstrap is a manual step -- see
``docs/audit/package_split.md``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "tests" / "_snapshots" / "pre_split" / "smoke"
SMOKE_CONFIG = REPO_ROOT / "src" / "world_model" / "configs" / "experiments" / "smoke.yaml"

# Files the smoke pipeline produces under outputs/world_model/smoke/.
EXACT_MATCH_FILES = (
    "comparison.csv",
    "baselines.csv",
    "action_embedding_meta.json",
)
PIXEL_MATCH_FILES = (
    "plots/loss_curves.png",
)


def _file_bytes_equal(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def _csv_records_equal(a: Path, b: Path) -> bool:
    """Compare two CSVs row-by-row so we get a useful diff on failure."""
    with a.open() as fa, b.open() as fb:
        ra = list(csv.reader(fa))
        rb = list(csv.reader(fb))
    return ra == rb


def _json_payload_equal(a: Path, b: Path) -> bool:
    return json.loads(a.read_text()) == json.loads(b.read_text())


def _png_pixels_equal(a: Path, b: Path) -> bool:
    """Compare PNGs by decoded pixel array (PNG metadata may differ)."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        pytest.skip(f"Pillow / numpy unavailable: {e}")
    arr_a = np.asarray(Image.open(a))
    arr_b = np.asarray(Image.open(b))
    if arr_a.shape != arr_b.shape:
        return False
    return bool((arr_a == arr_b).all())


@pytest.mark.smoke_regression
def test_smoke_outputs_byte_equivalent_against_pre_split_snapshot(tmp_path):
    """End-to-end: smoke run on current tree must match pre-split snapshot."""
    if not SNAPSHOT_DIR.exists():
        pytest.skip(
            f"No pre-split snapshot at {SNAPSHOT_DIR.relative_to(REPO_ROOT)}. "
            "Capture it once with the procedure in docs/audit/package_split.md, "
            "then re-run this test."
        )

    out_dir = tmp_path / "smoke_out"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    cmd = [
        sys.executable,
        "-m",
        "world_model.scripts.smoke_test",
        "--config",
        str(SMOKE_CONFIG),
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Smoke pipeline crashed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    failures: list[str] = []
    for rel in EXACT_MATCH_FILES:
        post = out_dir / rel
        pre = SNAPSHOT_DIR / rel
        if not post.exists():
            failures.append(f"missing post-split output: {rel}")
            continue
        if not pre.exists():
            failures.append(f"missing pre-split snapshot: {rel}")
            continue
        if rel.endswith(".csv") and not _csv_records_equal(pre, post):
            failures.append(f"CSV diverged: {rel}")
        elif rel.endswith(".json") and not _json_payload_equal(pre, post):
            failures.append(f"JSON diverged: {rel}")
        elif not _file_bytes_equal(pre, post):
            failures.append(f"raw bytes diverged: {rel}")
    for rel in PIXEL_MATCH_FILES:
        post = out_dir / rel
        pre = SNAPSHOT_DIR / rel
        if not post.exists():
            failures.append(f"missing post-split output: {rel}")
            continue
        if not pre.exists():
            failures.append(f"missing pre-split snapshot: {rel}")
            continue
        if not _png_pixels_equal(pre, post):
            failures.append(f"PNG pixels diverged: {rel} (hash pre={hashlib.sha1(pre.read_bytes()).hexdigest()[:8]} post={hashlib.sha1(post.read_bytes()).hexdigest()[:8]})")

    assert not failures, (
        "Post-split smoke outputs diverged from pre-split snapshot.\n"
        + "\n".join(f"  - {f}" for f in failures)
        + f"\nLogs: stdout={len(result.stdout)} bytes, stderr={len(result.stderr)} bytes."
    )
