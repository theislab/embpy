r"""Render ``runs/<run_id>/report.md`` from existing artefacts.

Reads whatever the training / baseline / compare scripts have written
into ``run_dir`` and produces a single self-contained markdown file:

* ``config.yaml`` (or the dump in ``train.log`` config line) -> embedded
  YAML block,
* ``comparison.csv``                  -> aggregated metrics table,
* ``baselines.csv``                   -> per-(baseline, metric) long table,
* ``plots/*.png``                     -> embedded image links.

Usage:

    python -m world_model.scripts.make_report \\
        --run-dir runs/world_model/single_replogle
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from world_model.evaluation.report import write_report
from world_model.utils import setup_logging
from world_model.utils.run_identity import resolve_latest_suffixed_run_dir

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render report.md from a run directory.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config to embed; defaults to <run-dir>/config.yaml if present.",
    )
    return parser.parse_args(argv)


def _load_config_dict(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path) as fp:
            return yaml.safe_load(fp) or {}
    except ImportError:
        return {}


def _collect_plots(run_dir: Path) -> dict[str, str]:
    plot_dir = run_dir / "plots"
    if not plot_dir.exists():
        return {}
    plots: dict[str, str] = {}
    for png in sorted(plot_dir.glob("*.png")):
        title = png.stem.replace("_", " ").capitalize()
        plots[title] = f"plots/{png.name}"
    return plots


def main(argv: list[str] | None = None) -> None:
    """Render the report from CLI arguments."""
    import pandas as pd

    args = parse_args(argv)
    run_dir = resolve_latest_suffixed_run_dir(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    setup_logging(level=logging.INFO, log_file=run_dir / "report.log")

    cfg_path = Path(args.config) if args.config else run_dir / "config.yaml"
    cfg_dict = _load_config_dict(cfg_path)
    run_name = cfg_dict.get("run_name") or run_dir.name

    aggregate_table: pd.DataFrame
    comparison_csv = run_dir / "comparison.csv"
    if comparison_csv.exists():
        aggregate_table = pd.read_csv(comparison_csv)
    else:
        aggregate_table = pd.DataFrame()
        logger.warning("No comparison.csv at %s; report will lack the aggregate table.", comparison_csv)

    per_pert_tables: dict[str, pd.DataFrame] = {}
    for csv in (run_dir / "eval").glob("per_pert_*.csv"):
        per_pert_tables[csv.stem.removeprefix("per_pert_")] = pd.read_csv(csv)

    plots = _collect_plots(run_dir)
    write_report(
        output_dir=run_dir,
        run_name=run_name,
        cfg_dict=cfg_dict,
        aggregate_table=aggregate_table,
        per_pert_tables=per_pert_tables or None,
        plot_paths=plots or None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
