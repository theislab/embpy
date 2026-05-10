"""End-to-end evaluation pipeline for the perturbation world model.

The pipeline produces *one prediction per (cell, perturbation)* in the
test split, builds aligned real / pred AnnData objects, and runs the
cell-eval metrics suite. The same harness is reused by every baseline,
so model and baselines are compared on byte-identical inputs.

Public entry-point: :func:`run_evaluation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..configs import EvalConfig
from ..data.dataloader import DataArtifacts
from ..models.world_model import WorldModel
from .baselines import ALL_BASELINES, Baseline, BaselineTrainData
from .cell_eval_runner import run_cell_eval
from .prep import build_pred_anndata, build_real_anndata

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Bundle returned per evaluator (model or baseline)."""

    name: str
    per_perturbation: Any  # pandas.DataFrame
    aggregate: Any         # pandas.DataFrame
    real_adata: Any
    pred_adata: Any


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _control_template(artifacts: DataArtifacts) -> np.ndarray:
    """Mean expression of train-split control cells. Shape ``(n_genes,)``."""
    full = artifacts.full_dataset
    train_idx = artifacts.split.train_indices
    labels = full.perturbation_labels[train_idx]
    is_control = labels == full.control_label
    if not is_control.any():
        raise RuntimeError("No control cells in train split.")
    expression = full.expression_view(train_idx[is_control])
    return expression.mean(axis=0).astype(np.float32)


def _build_perturbation_to_action(
    indexer: Any,
    gene_table: np.ndarray,
    perturbations: list[str],
) -> dict[str, np.ndarray]:
    """Map each perturbation label to its row in the gene-embedding table.

    Multi-gene perturbations are mean-pooled to mirror the action encoder.
    """
    out: dict[str, np.ndarray] = {}
    for p in perturbations:
        encoded = indexer.encode(p, control_label="non-targeting")
        rows = [gene_table[i] for i in encoded if i != 0]
        if not rows:
            continue
        out[str(p)] = np.stack(rows, axis=0).mean(axis=0).astype(np.float32)
    return out


def _build_test_inputs(
    real_adata: Any,
    artifacts: DataArtifacts,
    *,
    perturbation_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(state, action_labels)`` aligned to ``real_adata.obs``.

    For every test cell we use the train-split control template as the
    basal state. ``action_labels`` is the ``(n_test_cells,)`` string
    array of perturbation identifiers from the real AnnData.
    """
    n_cells = real_adata.n_obs
    template = _control_template(artifacts)
    state = np.broadcast_to(template, (n_cells, template.size)).astype(np.float32, copy=True)
    action_labels = np.asarray(real_adata.obs[perturbation_key].values, dtype=str)
    return state, action_labels


# ----------------------------------------------------------------------
# Model prediction (per-cell, mirroring the baseline API)
# ----------------------------------------------------------------------


@torch.no_grad()
def _predict_with_world_model(
    model: WorldModel,
    artifacts: DataArtifacts,
    *,
    real_adata: Any,
    perturbation_key: str,
    n_samples_per_pert: int,
    device: torch.device,
) -> np.ndarray:
    """Predict ``(n_test_cells, n_genes)`` post-perturbation expression.

    Strategy: for each unique perturbation label appearing in the test
    AnnData, draw ``n_samples_per_pert`` control cells, run the world
    model rollout for one step, decode to gene space, and average. The
    same per-perturbation prediction is then broadcast to every test
    cell carrying that label (so the output is aligned with
    ``real_adata.obs``).
    """
    model.eval().to(device)
    full = artifacts.full_dataset
    train_idx = artifacts.split.train_indices
    train_labels = full.perturbation_labels[train_idx]
    control_mask = train_labels == full.control_label
    control_pool = train_idx[control_mask]
    if control_pool.size == 0:
        raise RuntimeError("No control cells in train split for prediction.")

    rng = np.random.default_rng(0)
    K = full.stack_size
    indexer = full.indexer
    n_pert = full.n_pert

    labels = np.asarray(real_adata.obs[perturbation_key].values, dtype=str)
    unique = np.unique(labels)
    n_genes = real_adata.n_vars

    per_pert_pred: dict[str, np.ndarray] = {}
    for pert in unique:
        if pert == full.control_label:
            continue
        sampled = rng.choice(control_pool, size=n_samples_per_pert,
                             replace=control_pool.size < n_samples_per_pert)
        per_sample: list[np.ndarray] = []
        for cell_idx in sampled:
            stack_idx = rng.choice(control_pool, size=K, replace=control_pool.size < K)
            stack_idx[0] = cell_idx
            init_obs = torch.from_numpy(full.expression[stack_idx]).unsqueeze(0).to(device)

            encoded = indexer.encode(pert, full.control_label)
            action_indices = np.zeros((1, 1, n_pert), dtype=np.int64)
            for j, gid in enumerate(encoded[:n_pert]):
                action_indices[0, 0, j] = gid
            action_t = torch.from_numpy(action_indices).to(device)

            out = model.rollout(init_obs, action_t)
            x_hat = out["x_hat"]
            if x_hat is None:
                raise RuntimeError("WorldModel without decoder cannot produce gene-space predictions.")
            per_sample.append(x_hat[0, 0].cpu().numpy())
        per_pert_pred[str(pert)] = np.mean(np.stack(per_sample, axis=0), axis=0).astype(np.float32)

    out = np.empty((labels.size, n_genes), dtype=np.float32)
    template = _control_template(artifacts)
    for i, lbl in enumerate(labels):
        if lbl == full.control_label:
            out[i] = template
        else:
            vec = per_pert_pred.get(str(lbl))
            if vec is None:
                out[i] = template
            else:
                out[i] = vec
    return out


def _predictions_to_dict(
    predictions: np.ndarray,
    labels: np.ndarray,
    control_label: str,
) -> dict[str, np.ndarray]:
    """Group per-cell predictions into ``{perturbation: mean_vector}``."""
    out: dict[str, np.ndarray] = {}
    for lbl in np.unique(labels):
        if lbl == control_label:
            continue
        mask = labels == lbl
        out[str(lbl)] = predictions[mask].mean(axis=0).astype(np.float32)
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_evaluation(
    *,
    artifacts: DataArtifacts,
    output_dir: str | Path,
    eval_cfg: EvalConfig,
    perturbation_key: str = "perturbation",
    control_label: str = "non-targeting",
    model: WorldModel | None = None,
    baselines: dict[str, Baseline] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, EvaluationResult]:
    """Run the full evaluation harness on the model and/or every baseline.

    Parameters
    ----------
    artifacts
        Output of :func:`world_model.data.build_dataloaders`. Provides
        the deterministic split + the gene/action lookups.
    output_dir
        Where to persist real/pred AnnData (when
        ``eval_cfg.save_predictions=True``) and metric tables.
    eval_cfg
        Knobs controlling sampling and cell-eval usage.
    model
        Optional :class:`WorldModel`. If provided, evaluated under the name
        ``"world_model"``.
    baselines
        Optional ``{name: Baseline}`` mapping. If ``None``, all
        baselines registered in :data:`ALL_BASELINES` are used.
    device
        Torch device for model inference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    full = artifacts.full_dataset
    test_perts = list(artifacts.split.test_perturbations or [])
    if not test_perts:
        labels = full.perturbation_labels[artifacts.split.test_indices]
        test_perts = sorted(set(labels.tolist()) - {full.control_label})
    if not test_perts:
        raise RuntimeError("No test perturbations -- nothing to evaluate.")

    real_adata = build_real_anndata(
        expression=full.expression,
        perturbation_labels=full.perturbation_labels,
        test_indices=artifacts.split.test_indices,
        test_perturbations=test_perts,
        gene_symbols=artifacts.gene_symbols,
        perturbation_key=perturbation_key,
    )
    if eval_cfg.save_predictions:
        try:
            real_adata.write(eval_dir / "real.h5ad")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not write real.h5ad (%s).", e)

    test_state, test_actions = _build_test_inputs(
        real_adata, artifacts, perturbation_key=perturbation_key,
    )

    train_perts = list(artifacts.split.train_perturbations or [])
    perturbation_to_action_train = _build_perturbation_to_action(
        artifacts.indexer, artifacts.gene_table, train_perts,
    )
    perturbation_to_action_test = _build_perturbation_to_action(
        artifacts.indexer, artifacts.gene_table, test_perts,
    )

    results: dict[str, EvaluationResult] = {}
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if model is not None:
        logger.info("Evaluating world model on %d test perturbations.", len(test_perts))
        wm_per_cell = _predict_with_world_model(
            model, artifacts,
            real_adata=real_adata,
            perturbation_key=perturbation_key,
            n_samples_per_pert=eval_cfg.n_control_samples_per_pert,
            device=device,
        )
        wm_preds = _predictions_to_dict(wm_per_cell, test_actions, full.control_label)
        wm_pred_adata = build_pred_anndata(wm_preds, real_adata, perturbation_key=perturbation_key)
        per_pert, agg = run_cell_eval(
            real_adata, wm_pred_adata,
            perturbation_key=perturbation_key,
            control_label=control_label,
            deg_top_k=eval_cfg.deg_top_k,
            use_cell_eval=eval_cfg.use_cell_eval,
        )
        results["world_model"] = EvaluationResult(
            name="world_model", per_perturbation=per_pert, aggregate=agg,
            real_adata=real_adata, pred_adata=wm_pred_adata,
        )
        if eval_cfg.save_predictions:
            try:
                wm_pred_adata.write(eval_dir / "pred_world_model.h5ad")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not write pred_world_model.h5ad (%s).", e)

    if baselines is None:
        baselines = {name: cls() for name, cls in ALL_BASELINES.items()}

    train_data = BaselineTrainData(
        expression=full.expression,
        perturbation_labels=full.perturbation_labels,
        train_indices=artifacts.split.train_indices,
        control_label=full.control_label,
        gene_symbols=list(artifacts.gene_symbols),
        perturbation_to_action=perturbation_to_action_train,
    )

    for name, baseline in baselines.items():
        logger.info("Evaluating baseline %s ...", name)
        baseline.fit(train_data)
        if hasattr(baseline, "set_action_embeddings"):
            baseline.set_action_embeddings(perturbation_to_action_test)
        per_cell = baseline.predict(test_state, test_actions)
        preds_dict = _predictions_to_dict(per_cell, test_actions, full.control_label)
        pred_adata = build_pred_anndata(preds_dict, real_adata, perturbation_key=perturbation_key)
        per_pert, agg = run_cell_eval(
            real_adata, pred_adata,
            perturbation_key=perturbation_key,
            control_label=control_label,
            deg_top_k=eval_cfg.deg_top_k,
            use_cell_eval=eval_cfg.use_cell_eval,
        )
        results[name] = EvaluationResult(
            name=name, per_perturbation=per_pert, aggregate=agg,
            real_adata=real_adata, pred_adata=pred_adata,
        )
        if eval_cfg.save_predictions:
            try:
                pred_adata.write(eval_dir / f"pred_{name}.h5ad")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not write pred_%s.h5ad (%s).", name, e)

    return results


__all__ = ["EvaluationResult", "run_evaluation"]
