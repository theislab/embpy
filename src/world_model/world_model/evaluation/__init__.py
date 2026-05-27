"""Evaluation metrics and rollout helpers."""

from __future__ import annotations

from . import baselines, plots, prep
from .cell_eval_runner import run_cell_eval
from .metrics import (
    cosine_similarity,
    deg_overlap_top_k,
    delta_pearson,
    expression_r2,
    latent_l2_error,
    mae,
    mse,
    pearson_corr,
    r2_score,
    spearman_corr,
)
from .perturbation_eval import EvaluationResult, run_evaluation
from .report import write_report
from .rollouts import imagined_rollout

__all__ = [
    "EvaluationResult",
    "baselines",
    "cosine_similarity",
    "deg_overlap_top_k",
    "delta_pearson",
    "expression_r2",
    "imagined_rollout",
    "latent_l2_error",
    "mae",
    "mse",
    "pearson_corr",
    "plots",
    "prep",
    "r2_score",
    "run_cell_eval",
    "run_evaluation",
    "spearman_corr",
    "write_report",
]
