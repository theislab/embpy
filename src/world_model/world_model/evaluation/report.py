"""Generate ``runs/<run_id>/report.md`` from evaluation results.

The report is a single self-contained markdown file embedding:

* the run config,
* a metrics summary table (world model vs every baseline),
* the saved plots (loss curves + per-pert metric distribution + DEG
  overlap + baseline comparison + scatter).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_report(
    *,
    output_dir: str | Path,
    run_name: str,
    cfg_dict: dict[str, Any],
    aggregate_table: Any,
    per_pert_tables: dict[str, Any] | None = None,
    plot_paths: dict[str, str] | None = None,
) -> Path:
    """Write a markdown report under ``output_dir/report.md``.

    Returns
    -------
    pathlib.Path
        Path to the written report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines: list[str] = []
    lines.append(f"# Run report -- {run_name}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```yaml")
    lines.append(_to_yaml(cfg_dict))
    lines.append("```")
    lines.append("")

    lines.append("## Aggregated metrics")
    lines.append("")
    if aggregate_table is None or aggregate_table.empty:
        lines.append("_No aggregate metrics produced._")
    else:
        lines.append(aggregate_table.to_markdown(index=False))
    lines.append("")

    if per_pert_tables:
        lines.append("## Per-perturbation metrics")
        lines.append("")
        for name, df in per_pert_tables.items():
            lines.append(f"### {name}")
            lines.append("")
            if df is None or df.empty:
                lines.append("_(empty)_")
            else:
                lines.append(df.to_markdown(index=False))
            lines.append("")

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for name, rel in plot_paths.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"![{name}]({rel})")
            lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info("Wrote report to %s", report_path)
    return report_path


def _to_yaml(d: dict[str, Any]) -> str:
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_dump(d, sort_keys=False)
    except ImportError:
        import json  # noqa: PLC0415

        return json.dumps(d, indent=2, default=str)


__all__ = ["write_report"]
