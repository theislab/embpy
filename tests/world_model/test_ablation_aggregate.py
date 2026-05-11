"""Aggregator tests on synthetic per-run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from world_model.evaluation.ablation.aggregate import (
    aggregate_ablation,
    persist_summary,
    render_plots,
    write_report,
)
from world_model.evaluation.ablation.grid import ActionEncoderSpec


def _seed_run_dir(
    run_dir: Path,
    *,
    model_name: str,
    r2: float,
    mse: float,
    embedding_dim: int,
    n_unresolved: int,
    status: str = "ok",
    error: str | None = None,
    wall_clock_s: float = 12.0,
    final_train_loss: float = 0.4,
    final_val_loss: float = 0.5,
    write_metrics: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_metrics:
        (run_dir / "world_model_metrics.csv").write_text(
            f"name,r2,mse\nworld_model,{r2},{mse}\n"
        )
        (run_dir / "comparison.csv").write_text(
            f"name,r2,mse\nworld_model,{r2},{mse}\nmean_baseline,0.1,0.2\n"
        )
    (run_dir / "action_embedding_meta.json").write_text(json.dumps({
        "source": "bio_embedder",
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "n_symbols": 100,
        "n_unresolved": n_unresolved,
    }))
    (run_dir / "train_log.csv").write_text(
        f"epoch,train_loss,val_loss\n1,1.0,1.2\n2,{final_train_loss},{final_val_loss}\n"
    )
    meta = {"status": status, "wall_clock_s": wall_clock_s, "key": run_dir.name,
            "model_name": model_name}
    if error is not None:
        meta["error"] = error
    (run_dir / "_ablation_run.json").write_text(json.dumps(meta))


@pytest.fixture
def fake_runs(tmp_path: Path):
    grid = [
        ActionEncoderSpec(key="borzoi", model_name="borzoi_v0"),
        ActionEncoderSpec(key="esm2", model_name="esm2_650M"),
        ActionEncoderSpec(key="minilm", model_name="minilm_l6_v2"),
    ]
    _seed_run_dir(tmp_path / "borzoi", model_name="borzoi_v0",
                  r2=0.55, mse=0.10, embedding_dim=2048, n_unresolved=0,
                  wall_clock_s=300.0)
    _seed_run_dir(tmp_path / "esm2", model_name="esm2_650M",
                  r2=0.60, mse=0.08, embedding_dim=1280, n_unresolved=2,
                  wall_clock_s=180.0)
    _seed_run_dir(tmp_path / "minilm", model_name="minilm_l6_v2",
                  r2=0.0, mse=0.0, embedding_dim=384, n_unresolved=0,
                  status="failed", error="OOM",
                  wall_clock_s=42.0,
                  write_metrics=False)
    return tmp_path, grid


def test_aggregate_long_and_wide_shapes(fake_runs):
    output_root, grid = fake_runs
    long_df, wide_df = aggregate_ablation(output_root, grid)

    assert len(wide_df) == 3
    assert set(wide_df["grid_key"]) == {"borzoi", "esm2", "minilm"}

    ok_specs = wide_df[wide_df["status"] == "ok"]
    assert set(ok_specs["grid_key"]) == {"borzoi", "esm2"}
    assert ok_specs.shape[0] == 2
    assert "r2" in wide_df.columns and "mse" in wide_df.columns

    failed = wide_df[wide_df["status"] == "failed"]
    assert len(failed) == 1
    assert failed.iloc[0]["error"] == "OOM"
    assert pd.isna(failed.iloc[0]["r2"])

    metric_long = long_df[long_df["metric"].notna()]
    assert set(metric_long["grid_key"]) == {"borzoi", "esm2"}
    assert set(metric_long["metric"]) == {"r2", "mse"}


def test_persist_summary_writes_csv_and_json(fake_runs):
    output_root, grid = fake_runs
    long_df, wide_df = aggregate_ablation(output_root, grid)
    paths = persist_summary(output_root, grid, long_df, wide_df)
    assert paths["long"].exists()
    assert paths["wide"].exists()
    assert paths["json"].exists()
    summary = json.loads(paths["json"].read_text())
    assert summary["n_specs"] == 3
    grid_keys = [s["key"] for s in summary["grid"]]
    assert grid_keys == ["borzoi", "esm2", "minilm"]


def test_render_plots_creates_expected_files(fake_runs):
    output_root, grid = fake_runs
    long_df, wide_df = aggregate_ablation(output_root, grid)
    plot_paths = render_plots(wide_df, output_root / "plots")
    names = {p.name for p in plot_paths}
    # Three categories per metric (r2 + mse): bar, scaling, pareto.
    for metric in ("r2", "mse"):
        assert f"metric_bar_{metric}.png" in names
        assert f"embedding_dim_vs_{metric}.png" in names
        assert f"pareto_{metric}_vs_walltime.png" in names
    for p in plot_paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_report_embeds_grid_and_plots(fake_runs):
    output_root, grid = fake_runs
    long_df, wide_df = aggregate_ablation(output_root, grid)
    plot_paths = render_plots(wide_df, output_root / "plots")
    rep_path = write_report(output_root, grid, long_df, wide_df, plot_paths)
    assert rep_path.exists()
    text = rep_path.read_text()
    assert "Resolved grid" in text
    assert "Wide summary" in text
    for spec in grid:
        assert spec.key in text
    assert "metric_bar_r2.png" in text


def test_aggregate_handles_missing_metrics_csv(tmp_path: Path):
    grid = [ActionEncoderSpec(key="only", model_name="m")]
    run_dir = tmp_path / "only"
    run_dir.mkdir()
    (run_dir / "_ablation_run.json").write_text(json.dumps(
        {"status": "failed", "error": "boom"}
    ))
    long_df, wide_df = aggregate_ablation(tmp_path, grid)
    assert len(wide_df) == 1
    assert wide_df.iloc[0]["status"] == "failed"
    assert pd.isna(wide_df.iloc[0]["embedding_dim"])
