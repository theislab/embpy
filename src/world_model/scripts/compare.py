"""Merge world-model + baseline metrics into a unified comparison.

Reads ``outputs/<run_id>/baselines.csv`` (produced by
``run_baselines.py``) and an optional world-model row from
``outputs/<run_id>/world_model_metrics.csv`` (produced by
``train.py`` / ``eval.py``), pivots them into a wide table with one
row per evaluator and one column per metric, and writes:

* ``outputs/<run_id>/comparison.csv``
* ``outputs/<run_id>/plots/comparison.png`` (+ .svg)

Usage:

    python -m world_model.scripts.compare \\
        --run-dir outputs/world_model/single_replogle
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from world_model.evaluation.plots import plot_baseline_comparison
from world_model.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison.csv + comparison.png.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to outputs/<run_id> (must contain baselines.csv).",
    )
    parser.add_argument(
        "--baselines-csv",
        type=str,
        default=None,
        help="Optional override; defaults to <run-dir>/baselines.csv.",
    )
    parser.add_argument(
        "--world-model-csv",
        type=str,
        default=None,
        help="Optional override; defaults to <run-dir>/world_model_metrics.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import pandas as pd  # noqa: PLC0415

    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    setup_logging(level=logging.INFO, log_file=run_dir / "compare.log")

    baselines_csv = Path(args.baselines_csv or run_dir / "baselines.csv")
    if not baselines_csv.exists():
        raise FileNotFoundError(
            f"Expected {baselines_csv}. Run scripts/run_baselines.py first."
        )
    long = pd.read_csv(baselines_csv)
    expected_cols = {"baseline", "metric", "value"}
    if not expected_cols.issubset(long.columns):
        raise ValueError(
            f"baselines.csv must have columns {expected_cols}, got {set(long.columns)}"
        )

    wide = long.pivot_table(index="baseline", columns="metric", values="value", aggfunc="first")
    wide = wide.reset_index().rename(columns={"baseline": "name"})

    wm_csv = Path(args.world_model_csv or run_dir / "world_model_metrics.csv")
    if wm_csv.exists():
        wm_row = pd.read_csv(wm_csv)
        if "name" not in wm_row.columns:
            wm_row.insert(0, "name", "world_model")
        wide = pd.concat([wide, wm_row], ignore_index=True)
        logger.info("Added world model row from %s", wm_csv)
    else:
        logger.info("No world_model_metrics.csv at %s; comparison includes baselines only.", wm_csv)

    out_csv = run_dir / "comparison.csv"
    wide.to_csv(out_csv, index=False)
    logger.info("Wrote %s with shape %s", out_csv, wide.shape)

    plot_path = run_dir / "plots" / "comparison.png"
    plot_baseline_comparison(wide, plot_path)


if __name__ == "__main__":  # pragma: no cover
    main()
