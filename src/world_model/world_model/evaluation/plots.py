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

logger = logging.getLogger(__name__)


def _try_mpl():  # type: ignore[no-untyped-def]
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

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
) -> None:
    plt = _try_mpl()
    if plt is None or per_pert is None or per_pert.empty:
        return
    cols = [c for c in per_pert.columns if c.startswith(column_prefix)]
    if not cols:
        return
    col = cols[0]
    series = per_pert.set_index("perturbation")[col].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * series.size), 4))
    ax.bar(range(series.size), series.values, color="steelblue")
    ax.set_xticks(range(series.size))
    ax.set_xticklabels(series.index, rotation=90, fontsize=6)
    ax.set_ylabel(col)
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
) -> None:
    """Bar chart with one bar per (model/baseline, metric)."""
    plt = _try_mpl()
    if plt is None or aggregate_table is None or aggregate_table.empty:
        return
    df = aggregate_table.copy()
    if metrics is None:
        metrics = [c for c in df.columns if c not in {"name", "n_perturbations"}]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    for j, m in enumerate(metrics):
        ax = axes[0, j]
        sub = df.dropna(subset=[m]).set_index("name")[m].sort_values()
        ax.barh(range(sub.size), sub.values)
        ax.set_yticks(range(sub.size))
        ax.set_yticklabels(sub.index)
        ax.set_xlabel(m)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


__all__ = [
    "plot_baseline_comparison",
    "plot_deg_overlap_bar",
    "plot_per_perturbation_metric",
    "plot_pred_vs_real_scatter",
]
