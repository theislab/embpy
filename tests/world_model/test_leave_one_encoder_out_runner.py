"""Tests for the leave-one-encoder-out runner.

We monkeypatch ``train.main`` to a no-op that drops the minimal
artifacts the runner expects (``world_model_metrics.csv`` plus the
``<run_name>_pretrain_final.pt`` checkpoint stub).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from world_model.scripts import leave_one_encoder_out as runner


def _stub_train_factory(call_log: list[Path]):
    """Return a callable that mimics train.main(argv): writes minimal artifacts."""

    def stub_train_main(argv):
        cfg_path = None
        argv_list = list(argv)
        if "--config" in argv_list:
            i = argv_list.index("--config")
            cfg_path = Path(argv_list[i + 1])
        assert cfg_path is not None
        call_log.append(cfg_path)

        import yaml  # noqa: PLC0415

        cfg = yaml.safe_load(cfg_path.read_text())
        run_dir = Path(cfg["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        # Minimal world_model_metrics.csv -- enough for the aggregator.
        pd.DataFrame([{"r2": 0.42, "mse": 0.1}]).to_csv(
            run_dir / "world_model_metrics.csv", index=False
        )

        # Drop a fake pretrain checkpoint where the runner expects to
        # find it.  In transfer mode the runner queries
        # <output_dir>/pretrain/<run_name>_pretrain_final.pt so mirror
        # that layout.
        if cfg.get("mode") == "transfer":
            ck_dir = run_dir / "pretrain"
            ck_dir.mkdir(parents=True, exist_ok=True)
            (ck_dir / f"{cfg['run_name']}_pretrain_final.pt").write_bytes(b"\x00")

    return stub_train_main


def _write_grid_yaml(tmp_path: Path, n: int = 3) -> Path:
    keys = [f"enc{i}" for i in range(n)]
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text("\n".join(
        [f"- key: {k}\n  model_name: {k}_model\n" for k in keys]
    ))
    return grid_path


def _write_base_config(tmp_path: Path) -> Path:
    p = tmp_path / "transfer.yaml"
    p.write_text(
        "mode: transfer\n"
        "run_name: test_lone\n"
        f"output_dir: {tmp_path / '_unused'}\n"
        "transfer:\n"
        "  enabled: true\n"
        "  pretrain_dataset: nadig\n"
        "  pretrain_h5ad_path: /dev/null\n"
        "  pretrain_epochs: 1\n"
        "  finetune_epochs: 1\n"
        "  finetune_fraction: 0.1\n"
    )
    return p


def test_lone_diagonal_only_runs_once_per_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid_path = _write_grid_yaml(tmp_path, n=3)
    base_path = _write_base_config(tmp_path)
    output_root = tmp_path / "out"

    call_log: list[Path] = []
    monkeypatch.setattr(
        "world_model.scripts.leave_one_encoder_out.train_script.main",
        _stub_train_factory(call_log),
    )

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--strategy", "reset_adapter",
        "--output-root", str(output_root),
        "--diagonal-only",
    ])
    assert rc == 0

    # Each pretrain runs exactly once.
    assert len(call_log) == 3, f"expected 3 pretrain calls, got {len(call_log)}"

    # The Stage A run dirs exist with their checkpoints.
    for key in ("enc0", "enc1", "enc2"):
        run_dir = output_root / "_pretrain" / key
        assert (run_dir / "world_model_metrics.csv").exists()
        # Diagonal cell mirrored under <strategy>/<X>__to__<X>/.
        diag = output_root / "reset_adapter" / f"{key}__to__{key}"
        assert (diag / "world_model_metrics.csv").exists()

    long_csv = output_root / "reset_adapter" / "summary_long.csv"
    assert long_csv.exists()
    long_df = pd.read_csv(long_csv)
    # 3 encoders * 3 encoders = 9 cells with 2 metrics each = 18 rows
    # for the full grid; with --diagonal-only we still summarise the
    # full grid but off-diagonal cells lack metrics, so we expect at
    # least the diagonal rows to be present.
    diag_rows = long_df[long_df["pretrain"] == long_df["finetune"]]
    assert not diag_rows.empty


def test_lone_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    grid_path = _write_grid_yaml(tmp_path, n=3)
    base_path = _write_base_config(tmp_path)
    output_root = tmp_path / "dry"

    # Hard-fail if train.main is called -- dry-run should not invoke it.
    def _explode(argv):
        raise AssertionError("train.main must not be called in --dry-run")

    monkeypatch.setattr(
        "world_model.scripts.leave_one_encoder_out.train_script.main",
        _explode,
    )

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--strategy", "reset_adapter",
        "--output-root", str(output_root),
        "--diagonal-only",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "diagonal" in out.lower() or "cells_planned" in out.lower() or "cells" in out.lower()
    grid_resolved = output_root / "grid_resolved.json"
    assert grid_resolved.exists()
    payload = json.loads(grid_resolved.read_text())
    assert payload["strategy"] == "reset_adapter"
    assert len(payload["encoders"]) == 3


def test_lone_only_filter_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid_path = _write_grid_yaml(tmp_path, n=3)
    base_path = _write_base_config(tmp_path)
    output_root = tmp_path / "filt"

    call_log: list[Path] = []
    monkeypatch.setattr(
        "world_model.scripts.leave_one_encoder_out.train_script.main",
        _stub_train_factory(call_log),
    )

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--strategy", "reset_adapter",
        "--output-root", str(output_root),
        "--only", "enc0:enc1",
    ])
    assert rc == 0
    # Stage A runs only the X for which we requested any cell -- here enc0.
    # Stage B runs the single off-diagonal cell.
    # That is two train.main calls.
    assert len(call_log) == 2
