"""Aggregate per-run artifacts produced by the ablation runner.

Reads one ``<output_root>/<grid_key>/`` per spec and stitches them
into long / wide summary CSVs plus comparison plots and a Markdown
report. Re-runnable on its own:

    python -m world_model.evaluation.ablation.aggregate \
        --output-root runs/ablation_action_replogle \
        --grid configs/experiments/ablation_action_encoder.yaml

The aggregator never touches network / GPU; it only reads CSV / JSON
already on disk, so re-rendering after a hot-fix takes a couple of
seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .grid import (
    ActionAdapterSpec,
    ActionEncoderSpec,
    load_adapter_grid,
    load_grid,
    resolve_adapter_grid,
    resolve_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Per-spec readers
# ---------------------------------------------------------------------


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    """Load ``_ablation_run.json`` if the runner wrote one; otherwise infer."""
    p = run_dir / "_ablation_run.json"
    if p.exists():
        return json.loads(p.read_text())
    # Best-effort fallback for runs created outside the ablation runner
    # (e.g. when comparing against an existing single-config run).
    return {"status": "ok" if (run_dir / "world_model_metrics.csv").exists() else "missing"}


def _read_action_meta(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "action_embedding_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse %s: %s", p, exc)
        return {}


def _read_world_model_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "world_model_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df


def _read_train_log(run_dir: Path) -> dict[str, float]:
    """Scrape final-epoch train / val loss + total wall-clock from the CSV log."""
    p = run_dir / "train_log.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.EmptyDataError) as exc:
        logger.warning("Could not parse %s: %s", p, exc)
        return {}
    out: dict[str, float] = {}
    for col_in, col_out in (("train_loss", "final_train_loss"), ("val_loss", "final_val_loss")):
        if col_in in df.columns:
            series = pd.to_numeric(df[col_in], errors="coerce").dropna()
            if not series.empty:
                out[col_out] = float(series.iloc[-1])
    if "wall_clock_s" in df.columns:
        wc = pd.to_numeric(df["wall_clock_s"], errors="coerce").dropna()
        if not wc.empty:
            out["train_log_wall_clock_s"] = float(wc.iloc[-1])
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def _spec_run_dir(output_root: Path, spec: ActionEncoderSpec) -> Path:
    return output_root / spec.key


def aggregate_ablation(
    output_root: Path | str,
    grid: list[ActionEncoderSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the long-form and wide-form summary frames.

    Returns
    -------
    long
        One row per ``(grid_key, metric)`` pair plus the spec metadata.
    wide
        One row per spec; each metric becomes a column. ``status``,
        ``embedding_dim``, ``n_unresolved``, ``final_val_loss``, and
        ``wall_clock_s`` are always present; missing values are NaN.
    """
    output_root = Path(output_root)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for spec in grid:
        run_dir = _spec_run_dir(output_root, spec)
        run_meta = _read_run_meta(run_dir)
        action_meta = _read_action_meta(run_dir)
        train_meta = _read_train_log(run_dir)
        wm_metrics = _read_world_model_metrics(run_dir)

        embedding_dim = action_meta.get("embedding_dim")
        n_unresolved = action_meta.get("n_unresolved")
        status = run_meta.get("status", "ok" if not wm_metrics.empty else "missing")
        wall_clock_s = run_meta.get("wall_clock_s") or train_meta.get("train_log_wall_clock_s")
        peak_gpu_mem_mb = run_meta.get("peak_gpu_mem_mb")
        error = run_meta.get("error")

        wide_row: dict[str, Any] = {
            "grid_key": spec.key,
            "model_name": spec.model_name,
            "id_type": spec.id_type,
            "region": spec.region,
            "pooling": spec.pooling,
            "status": status,
            "embedding_dim": embedding_dim,
            "n_unresolved": n_unresolved,
            "wall_clock_s": wall_clock_s,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
            "final_train_loss": train_meta.get("final_train_loss"),
            "final_val_loss": train_meta.get("final_val_loss"),
            "error": error,
            "run_dir": str(run_dir),
        }

        if wm_metrics.empty:
            long_rows.append({
                **wide_row,
                "metric": np.nan,
                "value": np.nan,
            })
        else:
            row0 = wm_metrics.iloc[0]
            for metric in wm_metrics.columns:
                value = row0[metric]
                long_rows.append({
                    "grid_key": spec.key,
                    "model_name": spec.model_name,
                    "id_type": spec.id_type,
                    "region": spec.region,
                    "pooling": spec.pooling,
                    "status": status,
                    "embedding_dim": embedding_dim,
                    "n_unresolved": n_unresolved,
                    "metric": metric,
                    "value": _to_float(value),
                    "run_dir": str(run_dir),
                })
                wide_row[str(metric)] = _to_float(value)
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def _save_fig(fig, path: Path) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _metric_columns(wide: pd.DataFrame) -> list[str]:
    skip = {
        "grid_key", "model_name", "id_type", "region", "pooling", "status",
        "embedding_dim", "n_unresolved", "wall_clock_s", "peak_gpu_mem_mb",
        "final_train_loss", "final_val_loss", "error", "run_dir",
    }
    out: list[str] = []
    for col in wide.columns:
        if col in skip:
            continue
        coerced = pd.to_numeric(wide[col], errors="coerce")
        if coerced.notna().any():
            out.append(col)
    return out


def render_plots(wide: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Generate the per-metric / Pareto / scaling plots. Returns their paths."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = _metric_columns(wide)
    out_paths: list[Path] = []

    ok = wide[wide["status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        logger.warning("No successful runs in the ablation summary; skipping plots.")
        return out_paths

    for metric in metric_cols:
        sub = ok.dropna(subset=[metric])
        if sub.empty:
            continue

        # Bar plot per spec.
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(sub) + 2), 3.5))
        ax.bar(sub["grid_key"].tolist(), sub[metric].tolist(), color="#4C72B0")
        ax.set_ylabel(metric)
        ax.set_title(f"World-model {metric} by action encoder")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
        bar_path = plots_dir / f"metric_bar_{metric}.png"
        _save_fig(fig, bar_path)
        plt.close(fig)
        out_paths.append(bar_path)

        # Scaling: embedding_dim vs metric.
        if sub["embedding_dim"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            for _, row in sub.iterrows():
                ax.scatter(row["embedding_dim"], row[metric])
                ax.annotate(
                    str(row["grid_key"]),
                    xy=(row["embedding_dim"], row[metric]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlabel("embedding_dim")
            ax.set_ylabel(metric)
            ax.set_title(f"Capacity scaling: embedding_dim vs {metric}")
            scaling_path = plots_dir / f"embedding_dim_vs_{metric}.png"
            _save_fig(fig, scaling_path)
            plt.close(fig)
            out_paths.append(scaling_path)

        # Pareto: wall_clock_s vs metric.
        if sub["wall_clock_s"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            for _, row in sub.iterrows():
                if pd.isna(row["wall_clock_s"]):
                    continue
                ax.scatter(row["wall_clock_s"], row[metric])
                ax.annotate(
                    str(row["grid_key"]),
                    xy=(row["wall_clock_s"], row[metric]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlabel("wall_clock_s")
            ax.set_ylabel(metric)
            ax.set_title(f"Pareto: {metric} vs wall_clock_s")
            pareto_path = plots_dir / f"pareto_{metric}_vs_walltime.png"
            _save_fig(fig, pareto_path)
            plt.close(fig)
            out_paths.append(pareto_path)

    return out_paths


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def _grid_resolved_table(grid: list[ActionEncoderSpec]) -> pd.DataFrame:
    return pd.DataFrame([s.as_dict() for s in grid])


def write_report(
    output_root: Path,
    grid: list[ActionEncoderSpec],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    plot_paths: Iterable[Path],
) -> Path:
    """Render ``<output_root>/report.md`` summarising the sweep."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rep_path = output_root / "report.md"

    lines: list[str] = []
    lines.append(f"# Action-encoder ablation: {output_root.name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Resolved grid")
    lines.append("")
    grid_df = _grid_resolved_table(grid)
    lines.append(grid_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Wide summary")
    lines.append("")
    lines.append(wide_df.to_markdown(index=False))
    lines.append("")
    if not long_df.empty:
        lines.append("## Long summary (head)")
        lines.append("")
        lines.append(long_df.head(20).to_markdown(index=False))
        lines.append("")
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for path in sorted(set(plot_paths)):
            rel = Path(path).relative_to(output_root)
            lines.append(f"### {rel.stem}")
            lines.append("")
            lines.append(f"![{rel.stem}]({rel.as_posix()})")
            lines.append("")
    rep_path.write_text("\n".join(lines))
    return rep_path


# ---------------------------------------------------------------------
# Persist + CLI
# ---------------------------------------------------------------------


def persist_summary(
    output_root: Path,
    grid: list[ActionEncoderSpec],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_path = output_root / "summary_long.csv"
    wide_path = output_root / "summary_wide.csv"
    json_path = output_root / "summary.json"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "grid": [s.as_dict() for s in grid],
        "n_specs": len(grid),
    }, indent=2, default=str))
    return {"long": long_path, "wide": wide_path, "json": json_path}


def run(
    *,
    output_root: Path | str,
    grid_path: Path | str | None,
    only: str | None = None,
    skip: str | None = None,
) -> dict[str, Path]:
    """End-to-end: aggregate -> persist -> plot -> report."""
    grid = resolve_grid(grid_path=grid_path, only=only, skip=skip) if (only or skip) else load_grid(grid_path)
    long_df, wide_df = aggregate_ablation(output_root, grid)
    paths = persist_summary(Path(output_root), grid, long_df, wide_df)
    plot_paths = render_plots(wide_df, Path(output_root) / "plots")
    rep_path = write_report(Path(output_root), grid, long_df, wide_df, plot_paths)
    paths["report"] = rep_path
    paths["plots"] = Path(output_root) / "plots"
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the action-encoder ablation outputs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--grid", default=None,
                        help="YAML grid file. Defaults to the package default grid.")
    parser.add_argument("--only", default=None, help="Comma-separated grid keys.")
    parser.add_argument("--skip", default=None, help="Comma-separated grid keys.")
    parser.add_argument(
        "--mode",
        default="encoder",
        choices=["encoder", "adapter"],
        help="Which sweep this output belongs to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.mode == "adapter":
        paths = run_adapter(
            output_root=args.output_root,
            grid_path=args.grid,
            only=args.only,
            skip=args.skip,
        )
    else:
        paths = run(
            output_root=args.output_root,
            grid_path=args.grid,
            only=args.only,
            skip=args.skip,
        )
    for tag, path in paths.items():
        logger.info("Wrote %s -> %s", tag, path)


# ---------------------------------------------------------------------
# Adapter ablation aggregation (Phase 3)
# ---------------------------------------------------------------------


def _read_per_run_yaml(run_dir: Path) -> dict[str, Any]:
    """Read the ``config.yaml`` the runner dumped per spec."""
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(p.read_text()) or {}
    except (OSError, ImportError) as exc:
        logger.warning("Could not parse %s: %s", p, exc)
        return {}


def _adapter_param_count(
    *,
    kind: str,
    d_in: int,
    d_model: int,
    hidden_dim: int,
    lora_rank: int,
) -> int | None:
    """Closed-form trainable param count for each adapter kind.

    Mirrors :mod:`models.action.adapters` exactly. We do not instantiate
    the module here because the aggregator must run without torch.
    """
    if d_in is None or d_model is None:
        return None
    try:
        d_in_i = int(d_in)
        d_model_i = int(d_model)
    except (TypeError, ValueError):
        return None
    if kind == "linear":
        # nn.Linear: d_in*d_model weights + d_model bias.
        return d_in_i * d_model_i + d_model_i
    if kind == "mlp":
        h = int(hidden_dim)
        return d_in_i * h + h + h * d_model_i + d_model_i
    if kind == "lora":
        r = int(lora_rank)
        if r <= 0:
            # rank=0 -> only the frozen W0; nothing trainable.
            return 0
        return d_in_i * r + r * d_model_i
    return None


def _spec_adapter_run_dir(output_root: Path, spec: ActionAdapterSpec) -> Path:
    return output_root / spec.key


def aggregate_adapter(
    output_root: Path | str,
    grid: list[ActionAdapterSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapter sweep counterpart of :func:`aggregate_ablation`.

    Adds adapter-specific columns extracted from each per-run
    ``config.yaml``: ``kind``, ``hidden_dim``, ``dropout``,
    ``lora_rank``, ``param_count``, plus the embedding ``d_in``
    and ``d_model`` so the param count can be cross-checked.
    """
    output_root = Path(output_root)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for spec in grid:
        run_dir = _spec_adapter_run_dir(output_root, spec)
        run_meta = _read_run_meta(run_dir)
        action_meta = _read_action_meta(run_dir)
        train_meta = _read_train_log(run_dir)
        wm_metrics = _read_world_model_metrics(run_dir)
        cfg_dict = _read_per_run_yaml(run_dir)

        adapter_block = (cfg_dict.get("action_adapter") or {}) if isinstance(cfg_dict, dict) else {}
        kind = str(adapter_block.get("kind", spec.kind))
        hidden_dim = int(adapter_block.get("hidden_dim", spec.hidden_dim))
        dropout = float(adapter_block.get("dropout", spec.dropout))
        lora_rank = int(adapter_block.get("lora_rank", spec.lora_rank))
        encoder_block = (cfg_dict.get("encoder") or {}) if isinstance(cfg_dict, dict) else {}
        d_model = encoder_block.get("d_model")
        d_in = action_meta.get("embedding_dim")
        param_count = _adapter_param_count(
            kind=kind,
            d_in=d_in,
            d_model=d_model,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
        )

        status = run_meta.get("status", "ok" if not wm_metrics.empty else "missing")
        wall_clock_s = run_meta.get("wall_clock_s") or train_meta.get("train_log_wall_clock_s")
        peak_gpu_mem_mb = run_meta.get("peak_gpu_mem_mb")
        error = run_meta.get("error")

        wide_row: dict[str, Any] = {
            "grid_key": spec.key,
            "kind": kind,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lora_rank": lora_rank,
            "embedding_dim": d_in,
            "d_model": d_model,
            "param_count": param_count,
            "status": status,
            "wall_clock_s": wall_clock_s,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
            "final_train_loss": train_meta.get("final_train_loss"),
            "final_val_loss": train_meta.get("final_val_loss"),
            "error": error,
            "run_dir": str(run_dir),
        }

        if wm_metrics.empty:
            long_rows.append({**wide_row, "metric": np.nan, "value": np.nan})
        else:
            row0 = wm_metrics.iloc[0]
            for metric in wm_metrics.columns:
                value = row0[metric]
                long_rows.append({
                    "grid_key": spec.key,
                    "kind": kind,
                    "hidden_dim": hidden_dim,
                    "lora_rank": lora_rank,
                    "param_count": param_count,
                    "embedding_dim": d_in,
                    "d_model": d_model,
                    "status": status,
                    "metric": metric,
                    "value": _to_float(value),
                    "run_dir": str(run_dir),
                })
                wide_row[str(metric)] = _to_float(value)
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


_ADAPTER_NON_METRIC_COLS = {
    "grid_key", "kind", "hidden_dim", "dropout", "lora_rank", "embedding_dim",
    "d_model", "param_count", "status", "wall_clock_s", "peak_gpu_mem_mb",
    "final_train_loss", "final_val_loss", "error", "run_dir",
}


def _adapter_metric_columns(wide: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in wide.columns:
        if col in _ADAPTER_NON_METRIC_COLS:
            continue
        coerced = pd.to_numeric(wide[col], errors="coerce")
        if coerced.notna().any():
            out.append(col)
    return out


def render_adapter_plots(wide: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Adapter-specific plots: per-metric bar, param_count scaling, Pareto."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = _adapter_metric_columns(wide)
    out_paths: list[Path] = []

    ok = wide[wide["status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        logger.warning("No successful adapter runs in summary; skipping plots.")
        return out_paths

    for metric in metric_cols:
        sub = ok.dropna(subset=[metric])
        if sub.empty:
            continue

        # Per-spec bar.
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(sub) + 2), 3.5))
        ax.bar(sub["grid_key"].tolist(), sub[metric].tolist(), color="#55A868")
        ax.set_ylabel(metric)
        ax.set_title(f"World-model {metric} by adapter")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
        bar_path = plots_dir / f"metric_bar_{metric}.png"
        _save_fig(fig, bar_path)
        plt.close(fig)
        out_paths.append(bar_path)

        # param_count vs metric (capacity scaling).
        if sub["param_count"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            for _, row in sub.iterrows():
                if pd.isna(row["param_count"]):
                    continue
                ax.scatter(row["param_count"], row[metric])
                ax.annotate(
                    str(row["grid_key"]),
                    xy=(row["param_count"], row[metric]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xscale("symlog")
            ax.set_xlabel("trainable param_count (adapter only)")
            ax.set_ylabel(metric)
            ax.set_title(f"Param count vs {metric}")
            scaling_path = plots_dir / f"param_count_vs_{metric}.png"
            _save_fig(fig, scaling_path)
            plt.close(fig)
            out_paths.append(scaling_path)

        # Pareto: param_count vs metric (visually identical to scaling
        # but the convention from Phase 2 expects an explicit pareto_*
        # filename; keep it for symmetry with the encoder sweep).
        if sub["param_count"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            for _, row in sub.iterrows():
                if pd.isna(row["param_count"]):
                    continue
                ax.scatter(row["param_count"], row[metric])
                ax.annotate(
                    str(row["grid_key"]),
                    xy=(row["param_count"], row[metric]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xscale("symlog")
            ax.set_xlabel("param_count")
            ax.set_ylabel(metric)
            ax.set_title(f"Pareto: {metric} vs param_count")
            pareto_path = plots_dir / f"pareto_{metric}_vs_param_count.png"
            _save_fig(fig, pareto_path)
            plt.close(fig)
            out_paths.append(pareto_path)

    return out_paths


def write_adapter_report(
    output_root: Path,
    grid: list[ActionAdapterSpec],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    plot_paths: Iterable[Path],
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rep_path = output_root / "report.md"
    lines: list[str] = []
    lines.append(f"# Action-adapter ablation: {output_root.name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Resolved grid")
    lines.append("")
    grid_df = pd.DataFrame([s.as_dict() for s in grid])
    lines.append(grid_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Wide summary")
    lines.append("")
    lines.append(wide_df.to_markdown(index=False))
    lines.append("")
    if not long_df.empty:
        lines.append("## Long summary (head)")
        lines.append("")
        lines.append(long_df.head(20).to_markdown(index=False))
        lines.append("")
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for path in sorted(set(plot_paths)):
            rel = Path(path).relative_to(output_root)
            lines.append(f"### {rel.stem}")
            lines.append("")
            lines.append(f"![{rel.stem}]({rel.as_posix()})")
            lines.append("")
    rep_path.write_text("\n".join(lines))
    return rep_path


def persist_adapter_summary(
    output_root: Path,
    grid: list[ActionAdapterSpec],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_path = output_root / "summary_long.csv"
    wide_path = output_root / "summary_wide.csv"
    json_path = output_root / "summary.json"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "mode": "adapter",
        "grid": [s.as_dict() for s in grid],
        "n_specs": len(grid),
    }, indent=2, default=str))
    return {"long": long_path, "wide": wide_path, "json": json_path}


def run_adapter(
    *,
    output_root: Path | str,
    grid_path: Path | str | None,
    only: str | None = None,
    skip: str | None = None,
) -> dict[str, Path]:
    """End-to-end adapter aggregation: load -> aggregate -> persist -> plot -> report."""
    grid = (
        resolve_adapter_grid(grid_path=grid_path, only=only, skip=skip)
        if (only or skip) else load_adapter_grid(grid_path)
    )
    long_df, wide_df = aggregate_adapter(output_root, grid)
    paths = persist_adapter_summary(Path(output_root), grid, long_df, wide_df)
    plot_paths = render_adapter_plots(wide_df, Path(output_root) / "plots")
    rep_path = write_adapter_report(Path(output_root), grid, long_df, wide_df, plot_paths)
    paths["report"] = rep_path
    paths["plots"] = Path(output_root) / "plots"
    return paths


if __name__ == "__main__":  # pragma: no cover
    main()
