"""Plotting helpers for the evaluation report.

Every figure is saved as both PNG (for embedding in markdown) and SVG
(for editable post-processing). Plots are skipped silently with a
warning if matplotlib is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_mpl():  # type: ignore[no-untyped-def]
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plotting.")
        return None


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    logger.info("Saved plot %s (+ .svg)", path)


def plot_pred_vs_real_scatter(
    real_adata: Any,
    pred_adata: Any,
    out_path: Path,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = "non-targeting",
    title: str = "Predicted vs real (mean per perturbation)",
) -> None:
    """Plot mean predicted expression against mean real expression."""
    plt = _try_mpl()
    if plt is None:
        return
    real_X = np.asarray(real_adata.X, dtype=np.float32)
    pred_X = np.asarray(pred_adata.X, dtype=np.float32)
    labels = np.asarray(real_adata.obs[perturbation_key].values).astype(str)

    real_means: list[np.ndarray] = []
    pred_means: list[np.ndarray] = []
    for p in sorted(set(labels) - {control_label}):
        m = labels == p
        if not m.any():
            continue
        real_means.append(real_X[m].mean(axis=0))
        pred_means.append(pred_X[m].mean(axis=0))
    if not real_means:
        return
    R = np.concatenate(real_means)
    P = np.concatenate(pred_means)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(R, P, s=2, alpha=0.4)
    lims = [min(R.min(), P.min()), max(R.max(), P.max())]
    ax.plot(lims, lims, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("real mean expression")
    ax.set_ylabel("predicted mean expression")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_per_perturbation_metric(
    per_pert: Any,
    out_path: Path,
    *,
    metric: str = "r2",
    title: str | None = None,
) -> None:
    """Box/violin distribution of a metric across perturbations."""
    plt = _try_mpl()
    if plt is None or per_pert is None or per_pert.empty or metric not in per_pert.columns:
        return
    values = per_pert[metric].dropna().to_numpy()
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_ylabel(metric)
    ax.set_title(title or f"per-perturbation {metric}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_deg_overlap_bar(
    per_pert: Any,
    out_path: Path,
    *,
    column_prefix: str = "deg_overlap@",
    title: str = "DEG overlap@K (per perturbation)",
    dataset: str = "unknown",
    model: str = "world_model",
    seed: int | str | None = None,
) -> None:
    """Plot DEG-overlap distributions and write the tidy CSV sidecar."""
    plt = _try_mpl()
    if plt is None or per_pert is None or per_pert.empty:
        return
    cols = [c for c in per_pert.columns if c.startswith(column_prefix)]
    if not cols:
        return
    long = _per_pert_to_long(
        per_pert,
        metrics=cols,
        dataset=dataset,
        model=model,
        seed=seed,
    )
    _write_plot_csv(long, out_path)
    if long.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * long["metric"].nunique()), 4))
    _boxplot_long(ax, long, group_col="metric", value_col="value", point_col="seed")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_baseline_comparison(
    aggregate_table: Any,
    out_path: Path,
    *,
    metrics: list[str] | None = None,
    title: str = "Baselines vs world model (aggregated)",
    per_perturbation_long: pd.DataFrame | None = None,
    dataset: str = "unknown",
    seed: int | str | None = None,
) -> None:
    """Bar comparison with uncertainty across perturbations.

    ``per_perturbation_long`` is preferred and should contain one row per
    ``dataset x model/baseline x seed x perturbation x metric``. When only an
    aggregate table is available, the helper falls back to one pseudo-point per
    evaluator/metric so legacy callers still get a reproducible figure plus CSV.

    Bars show the mean metric value for each evaluator. Error bars show the
    sample standard deviation across perturbations/seeds when more than one
    value is available. This keeps the report compact while exposing the
    variation behind a headline aggregate.
    """
    plt = _try_mpl()
    if plt is None or aggregate_table is None or aggregate_table.empty:
        return
    df = aggregate_table.copy()
    if metrics is None:
        metrics = [c for c in df.columns if c not in {"name", "n_perturbations"}]
    if not metrics:
        return
    long = (
        _normalize_long_metrics(per_perturbation_long, metrics=metrics)
        if per_perturbation_long is not None
        else pd.DataFrame()
    )
    if long.empty:
        long = _aggregate_to_long(df, metrics=metrics, dataset=dataset, seed=seed)
    _write_plot_csv(long, out_path)
    if long.empty:
        return
    summary = _summarize_long_metrics(long)
    _write_plot_summary_csv(summary, out_path)
    if summary.empty:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    for j, m in enumerate(metrics):
        ax = axes[0, j]
        sub = summary[summary["metric"] == m].dropna(subset=["mean"]).copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        raw = long[long["metric"] == m].dropna(subset=["value"]).copy()
        _barplot_summary(ax, sub, raw=raw, horizontal=True)
        ax.set_xlabel(m)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"{title} (mean +/- SD)")
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def _per_pert_to_long(
    per_pert: pd.DataFrame,
    *,
    metrics: list[str],
    dataset: str,
    model: str,
    seed: int | str | None,
) -> pd.DataFrame:
    if "perturbation" not in per_pert.columns:
        return pd.DataFrame(columns=_LONG_COLUMNS)
    keep = ["perturbation", *metrics]
    long = per_pert[keep].melt(
        id_vars=["perturbation"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long.insert(0, "seed", "unknown" if seed is None else str(seed))
    long.insert(0, "baseline", model)
    long.insert(0, "model", model)
    long.insert(0, "dataset", dataset)
    return _normalize_long_metrics(long, metrics=metrics)


def per_perturbation_tables_to_long(
    tables: dict[str, pd.DataFrame],
    *,
    dataset: str,
    seed: int | str | None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Convert evaluator-specific per-perturbation frames into tidy plot data."""
    frames: list[pd.DataFrame] = []
    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        available = [c for c in frame.columns if c != "perturbation" and pd.api.types.is_numeric_dtype(frame[c])]
        selected = [m for m in available if metrics is None or m in metrics]
        if selected:
            frames.append(
                _per_pert_to_long(
                    frame,
                    metrics=selected,
                    dataset=dataset,
                    model=str(name),
                    seed=seed,
                )
            )
    if not frames:
        return pd.DataFrame(columns=_LONG_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _aggregate_to_long(
    aggregate_table: pd.DataFrame,
    *,
    metrics: list[str],
    dataset: str,
    seed: int | str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in aggregate_table.to_dict(orient="records"):
        name = str(row.get("name", row.get("baseline", "unknown")))
        for metric in metrics:
            value = row.get(metric)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": name,
                    "baseline": name,
                    "seed": "unknown" if seed is None else str(seed),
                    "perturbation": "aggregate",
                    "metric": metric,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows, columns=_LONG_COLUMNS)


_LONG_COLUMNS = ["dataset", "model", "baseline", "seed", "perturbation", "metric", "value"]


def _normalize_long_metrics(
    frame: pd.DataFrame | None,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_LONG_COLUMNS)
    out = frame.copy()
    if "model" not in out.columns and "baseline" in out.columns:
        out["model"] = out["baseline"].astype(str)
    if "baseline" not in out.columns and "model" in out.columns:
        out["baseline"] = out["model"].astype(str)
    for col, default in (
        ("dataset", "unknown"),
        ("model", "unknown"),
        ("baseline", "unknown"),
        ("seed", "unknown"),
        ("perturbation", "unknown"),
    ):
        if col not in out.columns:
            out[col] = default
    if metrics is not None:
        out = out[out["metric"].isin(metrics)]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out[_LONG_COLUMNS].dropna(subset=["value"]).reset_index(drop=True)


def _write_plot_csv(long: pd.DataFrame, out_path: Path) -> None:
    csv_path = out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(csv_path, index=False)
    logger.info("Saved plot data %s", csv_path)


def _write_plot_summary_csv(summary: pd.DataFrame, out_path: Path) -> None:
    csv_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    logger.info("Saved plot summary %s", csv_path)


def _summarize_long_metrics(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "baseline",
                "metric",
                "mean",
                "std",
                "sem",
                "n",
                "min",
                "max",
            ]
        )
    grouped = (
        long.groupby(["dataset", "model", "baseline", "metric"], dropna=False)["value"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    return grouped[
        [
            "dataset",
            "model",
            "baseline",
            "metric",
            "mean",
            "std",
            "sem",
            "n",
            "min",
            "max",
        ]
    ]


def _barplot_summary(
    ax: Any,
    summary: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    horizontal: bool = False,
) -> None:
    rows = summary.dropna(subset=["mean"]).copy()
    if rows.empty:
        ax.set_axis_off()
        return
    reverse = not _metric_lower_is_better(str(rows["metric"].iloc[0]))
    rows = rows.sort_values("mean", ascending=not reverse)
    labels = rows["model"].astype(str).tolist()
    positions = np.arange(len(rows), dtype=float)
    means = rows["mean"].astype(float).to_numpy()
    errors = rows["std"].astype(float).to_numpy()
    colors = ["#4C78A8" if label == "world_model" else "#B7C9DC" for label in labels]
    edge_colors = ["#244A73" if label == "world_model" else "#6E839A" for label in labels]

    if horizontal:
        ax.barh(
            positions,
            means,
            xerr=errors,
            height=0.68,
            color=colors,
            edgecolor=edge_colors,
            linewidth=1.0,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capsize": 3, "capthick": 1.0},
        )
        _overlay_raw_points(ax, raw, labels=labels, positions=positions, horizontal=True)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{label} (n={n})" for label, n in zip(labels, rows["n"], strict=False)])
        ax.axvline(0.0, color="#222222", linewidth=0.8, alpha=0.45)
    else:
        ax.bar(
            positions,
            means,
            yerr=errors,
            width=0.68,
            color=colors,
            edgecolor=edge_colors,
            linewidth=1.0,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capsize": 3, "capthick": 1.0},
        )
        _overlay_raw_points(ax, raw, labels=labels, positions=positions, horizontal=False)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{label}\nn={n}" for label, n in zip(labels, rows["n"], strict=False)])
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.45)


def _overlay_raw_points(
    ax: Any,
    raw: pd.DataFrame | None,
    *,
    labels: list[str],
    positions: np.ndarray,
    horizontal: bool,
) -> None:
    if raw is None or raw.empty:
        return
    rng = np.random.default_rng(0)
    lookup = dict(zip(labels, positions, strict=False))
    for label, sub in raw.groupby("model"):
        pos = lookup.get(str(label))
        if pos is None:
            continue
        vals = sub["value"].astype(float).to_numpy()
        if vals.size <= 1:
            continue
        jitter = rng.normal(0.0, 0.035, size=vals.size)
        if horizontal:
            ax.scatter(vals, pos + jitter, s=12, alpha=0.35, color="#333333", linewidths=0, zorder=3)
        else:
            ax.scatter(pos + jitter, vals, s=12, alpha=0.35, color="#333333", linewidths=0, zorder=3)


def _boxplot_long(
    ax: Any,
    long: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    point_col: str,
    horizontal: bool = False,
) -> None:
    grouped = [(name, values[value_col].astype(float).to_numpy()) for name, values in long.groupby(group_col)]
    grouped = [(name, values) for name, values in grouped if values.size]
    if not grouped:
        ax.set_axis_off()
        return
    reverse = not _metric_lower_is_better(str(long["metric"].iloc[0]))
    grouped.sort(key=lambda item: float(np.nanmedian(item[1])), reverse=reverse)
    labels = [str(name) for name, _ in grouped]
    values = [vals for _, vals in grouped]
    positions = np.arange(1, len(values) + 1)
    ax.boxplot(
        values,
        positions=positions,
        orientation="horizontal" if horizontal else "vertical",
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#D8E7F5", "edgecolor": "#365F7D"},
        medianprops={"color": "#9B2D20", "linewidth": 1.5},
    )
    rng = np.random.default_rng(0)
    lookup = dict(zip(labels, positions, strict=False))
    for label, sub in long.groupby(group_col):
        pos = lookup.get(str(label))
        if pos is None:
            continue
        vals = sub[value_col].astype(float).to_numpy()
        jitter = rng.normal(0.0, 0.035, size=vals.size)
        if horizontal:
            ax.scatter(vals, pos + jitter, s=14, alpha=0.65, color="#333333", linewidths=0)
        else:
            ax.scatter(pos + jitter, vals, s=14, alpha=0.65, color="#333333", linewidths=0)
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{label} (n={len(vals)})" for label, vals in zip(labels, values, strict=False)])
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{label}\nn={len(vals)}" for label, vals in zip(labels, values, strict=False)])


def _metric_lower_is_better(metric: str) -> bool:
    return metric in {"mse", "mae"}


__all__ = [
    "per_perturbation_tables_to_long",
    "plot_baseline_comparison",
    "plot_deg_overlap_bar",
    "plot_per_perturbation_metric",
    "plot_pred_vs_real_scatter",
]
