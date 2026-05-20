"""Wrapper around the ``cell_eval`` package (STATE / cell-eval).

The ``cell_eval`` API is imported lazily so the module is usable on
machines that do not have the package installed (e.g. a laptop running
the smoke test). When ``cell_eval`` is missing, a built-in fallback
computes the same headline metrics (MSE, MAE, R^2, Pearson, Spearman,
DEG overlap@K) on the supplied AnnData objects.

The wrapper accepts the AnnData pair built by
:mod:`world_model.evaluation.prep` and returns two pandas DataFrames:

* ``per_perturbation`` -- one row per perturbation, one column per metric.
* ``aggregate``       -- a single-row summary (mean over perturbations).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_cell_eval():  # type: ignore[no-untyped-def]
    try:
        import cell_eval  # noqa: PLC0415

        return cell_eval
    except ImportError:
        return None


def run_cell_eval(
    real_adata: Any,
    pred_adata: Any,
    *,
    perturbation_key: str = "perturbation",
    control_label: str = "non-targeting",
    deg_top_k: int = 50,
    use_cell_eval: bool = True,
):
    """Compute per-perturbation and aggregated metrics.

    Returns
    -------
    per_perturbation : pandas.DataFrame
    aggregate : pandas.DataFrame
    """
    import pandas as pd  # noqa: PLC0415

    if use_cell_eval:
        ce = _try_import_cell_eval()
        if ce is not None:
            try:
                evaluator_cls = getattr(ce, "MetricsEvaluator", None) or getattr(ce, "CellEval", None)
                if evaluator_cls is None:
                    raise AttributeError("cell_eval has no MetricsEvaluator / CellEval class.")
                evaluator = evaluator_cls(
                    real=real_adata,
                    pred=pred_adata,
                    pert_col=perturbation_key,
                    control_pert=control_label,
                    profile="full",
                )
                if hasattr(evaluator, "compute"):
                    result = evaluator.compute()
                elif hasattr(evaluator, "evaluate"):
                    result = evaluator.evaluate()
                else:
                    raise AttributeError("cell_eval evaluator has no compute()/evaluate() method.")
                if isinstance(result, tuple) and len(result) == 2:
                    per_pert, agg = result
                else:
                    per_pert = result
                    agg = pd.DataFrame([per_pert.mean(numeric_only=True)])
                return per_pert, agg
            except Exception as e:
                logger.warning(
                    "cell_eval invocation failed (%s); falling back to internal metrics.", e,
                )
        else:
            logger.warning("cell_eval not installed; using internal fallback metrics.")

    return _internal_metrics(
        real_adata, pred_adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        deg_top_k=deg_top_k,
    )


def _internal_metrics(
    real_adata: Any,
    pred_adata: Any,
    *,
    perturbation_key: str,
    control_label: str,
    deg_top_k: int,
):
    """Internal fallback metrics that mirror the ``cell_eval`` headline numbers."""
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    if real_adata.shape != pred_adata.shape:
        raise ValueError(
            f"real ({real_adata.shape}) and pred ({pred_adata.shape}) must share shape."
        )

    real_X = np.asarray(real_adata.X, dtype=np.float32)
    pred_X = np.asarray(pred_adata.X, dtype=np.float32)
    labels = np.asarray(real_adata.obs[perturbation_key].values).astype(str)

    is_control = labels == control_label
    if is_control.any():
        control_real_mean = real_X[is_control].mean(axis=0)
    else:
        control_real_mean = real_X.mean(axis=0)

    rows: list[dict[str, Any]] = []
    for pert in sorted(set(labels) - {control_label}):
        mask = labels == pert
        if not mask.any():
            continue
        real_mean = real_X[mask].mean(axis=0)
        pred_mean = pred_X[mask].mean(axis=0)

        mse = float(np.mean((real_mean - pred_mean) ** 2))
        mae = float(np.mean(np.abs(real_mean - pred_mean)))

        ss_res = float(np.sum((real_mean - pred_mean) ** 2))
        ss_tot = float(np.sum((real_mean - real_mean.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        pr = float(pearsonr(real_mean, pred_mean)[0]) if real_mean.size > 1 else float("nan")
        sp = float(spearmanr(real_mean, pred_mean)[0]) if real_mean.size > 1 else float("nan")

        # Delta-space metrics: predicted vs true perturbation effect.
        delta_real = real_mean - control_real_mean
        delta_pred = pred_mean - control_real_mean
        denom = max(float(np.linalg.norm(delta_real) * np.linalg.norm(delta_pred)), 1e-8)
        delta_cosine = float(np.dot(delta_real, delta_pred) / denom)

        # DEG overlap@K based on absolute delta magnitude.
        k = min(deg_top_k, real_mean.size)
        top_real = np.argsort(-np.abs(delta_real))[:k]
        top_pred = np.argsort(-np.abs(delta_pred))[:k]
        deg_overlap = len(set(top_real.tolist()) & set(top_pred.tolist())) / max(k, 1)

        rows.append({
            "perturbation": pert,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "pearson": pr,
            "spearman": sp,
            "delta_cosine": delta_cosine,
            f"deg_overlap@{deg_top_k}": deg_overlap,
            "n_cells": int(mask.sum()),
        })

    per_pert = pd.DataFrame(rows)
    if per_pert.empty:
        agg = pd.DataFrame()
    else:
        agg = pd.DataFrame([per_pert.drop(columns=["perturbation"]).mean(numeric_only=True)])
        agg["n_perturbations"] = len(per_pert)
    return per_pert, agg


__all__ = ["run_cell_eval"]
