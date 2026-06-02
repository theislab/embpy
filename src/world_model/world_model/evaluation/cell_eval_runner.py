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
import tempfile
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CELL_EVAL_INSTALL_HINT = (
    "Install the ArcInstitute cell-eval package with `pixi install -e gpu` "
    "inside this repository, or with `pip install cell-eval` in a plain Python environment."
)


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
    require_cell_eval: bool = False,
    profile: str = "full",
    num_threads: int = 1,
    outdir: str | None = None,
):
    """Compute per-perturbation and aggregated metrics.

    Returns
    -------
    per_perturbation : pandas.DataFrame
    aggregate : pandas.DataFrame
    """
    if use_cell_eval:
        ce = _try_import_cell_eval()
        if ce is not None:
            try:
                per_pert, agg = _run_external_cell_eval(
                    ce,
                    real_adata,
                    pred_adata,
                    perturbation_key=perturbation_key,
                    control_label=control_label,
                    profile=profile,
                    num_threads=num_threads,
                    outdir=outdir,
                )
                logger.info(
                    "cell_eval finished with profile=%r: %d per-perturbation rows, %d aggregate rows.",
                    profile,
                    len(per_pert),
                    len(agg),
                )
                return per_pert, agg
            except Exception as e:
                message = (
                    f"cell_eval is installed but failed to run with profile={profile!r}: {e}. "
                    "Check that real/pred AnnData have matching shapes, compatible obs/var, "
                    f"and perturbation column {perturbation_key!r}."
                )
                if require_cell_eval:
                    raise RuntimeError(message) from e
                logger.warning("%s Falling back to internal metrics.", message)
        else:
            message = f"cell_eval not installed; using internal fallback metrics. {CELL_EVAL_INSTALL_HINT}"
            if require_cell_eval:
                raise RuntimeError(
                    "cell-eval is required for this evaluation but is not installed. "
                    f"{CELL_EVAL_INSTALL_HINT}"
                )
            logger.warning(message)

    return _internal_metrics(
        real_adata, pred_adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        deg_top_k=deg_top_k,
    )


def _run_external_cell_eval(
    cell_eval_module: Any,
    real_adata: Any,
    pred_adata: Any,
    *,
    perturbation_key: str,
    control_label: str,
    profile: str,
    num_threads: int,
    outdir: str | None,
):
    """Run ArcInstitute cell-eval and normalize its output to pandas."""
    evaluator_cls = getattr(cell_eval_module, "MetricsEvaluator", None) or getattr(cell_eval_module, "CellEval", None)
    if evaluator_cls is None:
        raise AttributeError("cell_eval has no MetricsEvaluator / CellEval class.")

    if outdir is None:
        with tempfile.TemporaryDirectory(prefix="embpy-cell-eval-") as tmpdir:
            result = _compute_external_cell_eval(
                evaluator_cls,
                real_adata=real_adata,
                pred_adata=pred_adata,
                perturbation_key=perturbation_key,
                control_label=control_label,
                profile=profile,
                num_threads=num_threads,
                outdir=f"{tmpdir}/cell_eval",
            )
    else:
        result = _compute_external_cell_eval(
            evaluator_cls,
            real_adata=real_adata,
            pred_adata=pred_adata,
            perturbation_key=perturbation_key,
            control_label=control_label,
            profile=profile,
            num_threads=num_threads,
            outdir=outdir,
        )

    return _normalize_cell_eval_result(result)


def _compute_external_cell_eval(
    evaluator_cls: Any,
    *,
    real_adata: Any,
    pred_adata: Any,
    perturbation_key: str,
    control_label: str,
    profile: str,
    num_threads: int,
    outdir: str,
) -> Any:
    evaluator = _build_cell_eval_evaluator(
        evaluator_cls,
        real_adata=real_adata,
        pred_adata=pred_adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        num_threads=num_threads,
        outdir=outdir,
    )

    if hasattr(evaluator, "compute"):
        return _call_cell_eval_compute(evaluator.compute, profile=profile)
    if hasattr(evaluator, "evaluate"):
        return evaluator.evaluate()
    raise AttributeError("cell_eval evaluator has no compute()/evaluate() method.")


def _build_cell_eval_evaluator(
    evaluator_cls: Any,
    *,
    real_adata: Any,
    pred_adata: Any,
    perturbation_key: str,
    control_label: str,
    num_threads: int,
    outdir: str,
) -> Any:
    """Instantiate current cell-eval API, with a legacy fallback."""
    try:
        return evaluator_cls(
            adata_pred=pred_adata,
            adata_real=real_adata,
            control_pert=control_label,
            pert_col=perturbation_key,
            num_threads=num_threads,
            outdir=outdir,
        )
    except TypeError as current_api_error:
        try:
            return evaluator_cls(
                pred=pred_adata,
                real=real_adata,
                control_pert=control_label,
                pert_col=perturbation_key,
            )
        except TypeError:
            raise current_api_error


def _call_cell_eval_compute(compute: Any, *, profile: str) -> Any:
    """Call ``MetricsEvaluator.compute`` across supported cell-eval versions."""
    try:
        return compute(profile=profile, write_csv=False)
    except TypeError as current_api_error:
        try:
            return compute(profile=profile)
        except TypeError:
            try:
                return compute()
            except TypeError:
                raise current_api_error


def _normalize_cell_eval_result(result: Any):
    """Return ``(per_perturbation, aggregate)`` as pandas DataFrames."""
    import pandas as pd  # noqa: PLC0415

    if isinstance(result, tuple) and len(result) == 2:
        per_pert, agg = result
    else:
        per_pert = result
        per_pert_pd = _to_pandas_frame(per_pert)
        agg = pd.DataFrame([per_pert_pd.mean(numeric_only=True)])
        return per_pert_pd, agg

    return _to_pandas_frame(per_pert), _to_pandas_frame(agg)


def _to_pandas_frame(value: Any):
    """Convert pandas / polars / lazy-polars results to pandas."""
    import pandas as pd  # noqa: PLC0415

    if isinstance(value, pd.DataFrame):
        return value
    if hasattr(value, "collect"):
        value = value.collect()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


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
