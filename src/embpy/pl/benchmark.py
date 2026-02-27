"""Benchmark result visualizations.

Provides bar-chart visualizations for :func:`embpy.tl.benchmark_embeddings`
results, with automatic error-bar support when results come from
``mode="rigorous"`` (5-fold CV).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_METRICS = ["r2", "pearson", "spearman"]


def _is_rigorous(df: pd.DataFrame) -> bool:
    """Detect whether a results DataFrame came from rigorous mode."""
    return "r2_mean" in df.columns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_benchmark(
    results: pd.DataFrame,
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> Figure:
    """Bar chart of benchmark results.

    Automatically detects whether the DataFrame comes from
    ``mode="quick"`` or ``mode="rigorous"`` and adds error bars
    (mean +/- 1 std) for the rigorous case.

    Parameters
    ----------
    results
        DataFrame returned by :func:`embpy.tl.benchmark_embeddings`.
    metrics
        Which metrics to plot.  Defaults to ``["r2", "pearson", "spearman"]``.
        For rigorous mode, pass the base names (e.g. ``"r2"``), not
        ``"r2_mean"``.
    figsize
        Figure size.
    title
        Optional plot title.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    metrics = metrics or list(_BASE_METRICS)
    rigorous = _is_rigorous(results)

    model_names = list(results.index)
    n_metrics = len(metrics)
    n_models = len(model_names)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", n_colors=n_models)

    for i, model in enumerate(model_names):
        vals = []
        errs = []
        for m in metrics:
            if rigorous:
                mean_col = f"{m}_mean"
                std_col = f"{m}_std"
                vals.append(float(results.loc[model, mean_col]) if mean_col in results.columns else 0.0)
                errs.append(float(results.loc[model, std_col]) if std_col in results.columns else 0.0)
            else:
                vals.append(float(results.loc[model, m]) if m in results.columns else 0.0)
                errs.append(0.0)

        offset = (i - n_models / 2 + 0.5) * width
        bar_kw: dict[str, Any] = {
            "color": palette[i],
            "edgecolor": "k",
            "linewidth": 0.5,
            "label": model,
        }
        if rigorous:
            ax.bar(x + offset, vals, width, yerr=errs, capsize=3, **bar_kw)
        else:
            ax.bar(x + offset, vals, width, **bar_kw)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylabel("Score")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    default_title = "Benchmark Results"
    if rigorous:
        default_title += " (5-fold CV)"
    ax.set_title(title or default_title)

    fig.tight_layout()
    return fig


def plot_benchmark_comparison(
    results_dict: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare benchmark results across multiple embeddings.

    Produces one subplot per metric, with bars grouped by embedding
    and model.  Error bars are shown when rigorous-mode DataFrames
    are provided.

    Parameters
    ----------
    results_dict
        Mapping of ``{embedding_label: results_DataFrame}``.
        E.g. ``{"ChemBERTa": df1, "MolFormer": df2}``.
    metrics
        Metrics to plot.  Defaults to ``["r2", "pearson", "spearman"]``.
    figsize
        Overall figure size.  Defaults to ``(6*n_metrics, 5)``.

    Returns
    -------
    The matplotlib ``Figure``.
    """
    metrics = metrics or list(_BASE_METRICS)
    emb_names = list(results_dict.keys())
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (6 * n_metrics, 5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)

    first_df = next(iter(results_dict.values()))
    model_names = list(first_df.index)
    rigorous = _is_rigorous(first_df)

    n_groups = len(emb_names)
    n_bars = len(model_names)
    group_width = 0.8
    bar_width = group_width / n_bars

    palette = sns.color_palette("husl", n_colors=n_bars)

    for col_idx, metric in enumerate(metrics):
        ax = axes[0, col_idx]
        x = np.arange(n_groups)

        for bar_idx, model in enumerate(model_names):
            vals = []
            errs = []
            for emb_name in emb_names:
                df = results_dict[emb_name]
                if model not in df.index:
                    vals.append(0.0)
                    errs.append(0.0)
                    continue

                if rigorous:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    vals.append(float(df.loc[model, mean_col]) if mean_col in df.columns else 0.0)
                    errs.append(float(df.loc[model, std_col]) if std_col in df.columns else 0.0)
                else:
                    vals.append(float(df.loc[model, metric]) if metric in df.columns else 0.0)
                    errs.append(0.0)

            offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
            bar_kw: dict[str, Any] = {
                "color": palette[bar_idx],
                "edgecolor": "k",
                "linewidth": 0.5,
                "label": model if col_idx == 0 else None,
            }
            if rigorous:
                ax.bar(x + offset, vals, bar_width, yerr=errs, capsize=3, **bar_kw)
            else:
                ax.bar(x + offset, vals, bar_width, **bar_kw)

        ax.set_xticks(x)
        ax.set_xticklabels(emb_names, fontsize=9)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper right", fontsize=8,
                   bbox_to_anchor=(1.0, 0.98))

    fig.suptitle(
        "Embedding Comparison" + (" (5-fold CV)" if rigorous else ""),
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig
