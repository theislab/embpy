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
    aggregate: Any  # pandas.DataFrame
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
    query_indexer = getattr(full, "query_indexer", indexer)
    n_pert = full.n_pert

    labels = np.asarray(real_adata.obs[perturbation_key].values, dtype=str)
    unique = np.unique(labels)
    # Predictions live in the model's observation space (= embedding dim
    # under a foreign backbone, raw HVG dim otherwise). real_adata may
    # be in a different space when run_evaluation has already chosen to
    # build the truth side from raw_expression -- decoding happens at
    # the call site, not here.
    n_genes = int(full.expression.shape[1])

    per_pert_pred: dict[str, np.ndarray] = {}
    for pert in unique:
        if pert == full.control_label:
            continue
        sampled = rng.choice(control_pool, size=n_samples_per_pert, replace=control_pool.size < n_samples_per_pert)
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


@torch.no_grad()
def _predict_incontext(
    model: Any,
    artifacts: DataArtifacts,
    *,
    real_adata: Any,
    perturbation_key: str,
    n_samples_per_pert: int,
    device: torch.device,
) -> np.ndarray:
    """In-context analogue of :func:`_predict_with_world_model`.

    For every held-out (test) perturbation the model is given a SET of
    support triplets drawn from the TRAIN split -- each
    ``(control_stack, train_action_i, train_perturbed_stack_i)`` -- plus
    the query ``(control_stack, test_action, ?)``, and predicts the
    query's perturbed state. This is the in-context generalization the
    model was trained for: learn how train perturbations behave, then
    answer a perturbation never seen in training. Output matches the
    baseline / world-model contract: ``(n_test_cells, n_genes)``.
    """
    model.eval().to(device)
    full = artifacts.full_dataset
    train_idx = artifacts.split.train_indices
    train_labels = full.perturbation_labels[train_idx]
    control_mask = train_labels == full.control_label
    control_pool = train_idx[control_mask]
    if control_pool.size == 0:
        raise RuntimeError("No control cells in train split for in-context eval.")

    # Train (non-control) perturbations usable as support, with their
    # train-split cell pools.
    support_labels_all = sorted(set(train_labels[~control_mask].tolist()))
    if not support_labels_all:
        raise RuntimeError("No train perturbations to build a support set from.")
    pool_by_label = {p: train_idx[train_labels == p] for p in support_labels_all}

    rng = np.random.default_rng(0)
    K = full.stack_size
    indexer = full.indexer
    n_pert = full.n_pert
    M = min(int(getattr(model, "default_support_size", 16)), len(support_labels_all))
    n_genes = int(full.expression.shape[1])

    # Bucket-aware context selection ("most similar labels = same
    # cell-type / batch"), matching the within-bucket sampling the
    # dataset uses at train time. Falls back to the global train pool
    # when bucketing is off or a bucket lacks usable support.
    train_support_set = set(support_labels_all)
    cbl: dict = getattr(full, "_cells_by_bucket_label", {}) or {}
    buckets_on = getattr(full, "_cell_buckets", None) is not None and bool(cbl)
    bucket_control: dict = {}
    bucket_support: dict = {}
    if buckets_on:
        for b, per_label in cbl.items():
            ctrl = per_label.get(full.control_label)
            if ctrl is None or ctrl.size == 0:
                continue
            sup = [lbl for lbl in per_label if lbl in train_support_set and lbl != full.control_label]
            if sup:
                bucket_control[b] = ctrl
                bucket_support[b] = sup

    def _action(lbl: str, *, query: bool = False) -> np.ndarray:
        a = np.zeros((n_pert,), dtype=np.int64)
        active_indexer = query_indexer if query else indexer
        for j, gid in enumerate(active_indexer.encode(lbl, full.control_label)[:n_pert]):
            a[j] = gid
        return a

    def _stack(pool: np.ndarray) -> np.ndarray:
        idx = rng.choice(pool, size=K, replace=pool.size < K)
        return full.expression[idx]

    labels = np.asarray(real_adata.obs[perturbation_key].values, dtype=str)
    unique = np.unique(labels)

    per_pert_pred: dict[str, np.ndarray] = {}
    for pert in unique:
        if pert == full.control_label:
            continue
        # Buckets where this query perturbation actually occurs and that
        # carry both a control and >=1 train support perturbation.
        cand_buckets = [b for b in bucket_support if str(pert) in cbl[b]] if buckets_on else []
        samples: list[np.ndarray] = []
        for _ in range(n_samples_per_pert):
            if cand_buckets:  # within-bucket (same substrate as the query)
                b = cand_buckets[rng.integers(len(cand_buckets))]
                ctrl_pool = bucket_control[b]
                sup_pool = [s for s in bucket_support[b] if s != str(pert)] or bucket_support[b]
                m = min(M, len(sup_pool))
                sup = list(rng.choice(np.asarray(sup_pool, dtype=object), size=m, replace=len(sup_pool) < m))
                support_next = np.stack([_stack(cbl[b][s]) for s in sup])
            else:  # global fallback
                ctrl_pool = control_pool
                sup = list(rng.choice(support_labels_all, size=M, replace=len(support_labels_all) < M))
                support_next = np.stack([_stack(pool_by_label[s]) for s in sup])
            support_obs = np.stack([_stack(ctrl_pool) for _ in sup])
            support_act = np.stack([_action(str(s)) for s in sup])
            batch = {
                "support_obs": torch.from_numpy(support_obs).unsqueeze(0).to(device),
                "support_next": torch.from_numpy(support_next).unsqueeze(0).to(device),
                "support_act": torch.from_numpy(support_act).unsqueeze(0).to(device),
                "query_obs": torch.from_numpy(_stack(ctrl_pool)).unsqueeze(0).to(device),
                "query_act": torch.from_numpy(_action(str(pert), query=True)).unsqueeze(0).to(device),
            }
            out = model.predict(batch)
            if out["x_hat"] is None:
                raise RuntimeError("In-context model without decoder cannot predict genes.")
            samples.append(out["x_hat"][0].cpu().numpy())
        per_pert_pred[str(pert)] = np.mean(
            np.stack(samples, axis=0),
            axis=0,
        ).astype(np.float32)

    out = np.empty((labels.size, n_genes), dtype=np.float32)
    template = _control_template(artifacts)
    for i, lbl in enumerate(labels):
        out[i] = per_pert_pred.get(str(lbl), template) if lbl != full.control_label else template
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

    # Decide which space cell-eval will run in.
    #   - Local backbone: .expression is already gene-space; no decode needed.
    #   - State backbone (supports_decode=True): predictions get decoded
    #     via backbone.decode() before comparison; truth side comes from
    #     full.raw_expression (the original HVG matrix preserved by the
    #     dataloader's pre-encode step).
    #   - Stack backbone (no decode()): fall back to embedding-space
    #     metrics with synthetic dim_* var-names. DEG-based cell-eval
    #     metrics are not biologically interpretable in this regime.
    raw_exp = getattr(full, "raw_expression", full.expression)
    backbone = artifacts.state_backbone
    embedding_eq_genes = (raw_exp is full.expression) or (raw_exp.shape[1] == full.expression.shape[1])
    backbone_supports_decode = bool(backbone is not None and getattr(backbone, "supports_decode", False))
    gene_space_eval = embedding_eq_genes or backbone_supports_decode

    if gene_space_eval:
        # If the backbone exposes a fixed training gene set (STACK), align
        # to HVG ∩ STACK. STATE decodes to whatever gene_names we ask
        # for, so no filtering is needed there.
        final_gene_names = list(artifacts.gene_symbols)
        if not embedding_eq_genes and hasattr(backbone, "training_gene_names"):
            train_set = set(backbone.training_gene_names())
            kept = [g for g in artifacts.gene_symbols if g in train_set]
            if len(kept) < len(artifacts.gene_symbols):
                logger.warning(
                    "Backbone %r covers %d / %d HVGs; restricting eval to "
                    "the intersection. The dropped HVGs are absent from the "
                    "backbone's training gene list.",
                    getattr(backbone, "name", "?"),
                    len(kept),
                    len(artifacts.gene_symbols),
                )
            final_gene_names = kept

        if final_gene_names != list(artifacts.gene_symbols):
            name_to_idx = {g: i for i, g in enumerate(artifacts.gene_symbols)}
            cols = np.array([name_to_idx[g] for g in final_gene_names], dtype=np.int64)
            raw_exp_eval = raw_exp[:, cols]
        else:
            raw_exp_eval = raw_exp

        # Median library size of training control cells -- a stable scalar
        # to feed STACK's NB decoder. Falls back to all cells if no controls.
        if not embedding_eq_genes and getattr(backbone, "name", "") == "stack":
            ctrl_mask = full.perturbation_labels == full.control_label
            pool = raw_exp[ctrl_mask] if ctrl_mask.any() else raw_exp
            decode_lib_size: float | None = float(np.median(pool.sum(axis=1)))
        else:
            decode_lib_size = None

        real_adata = build_real_anndata(
            expression=raw_exp_eval,
            perturbation_labels=full.perturbation_labels,
            test_indices=artifacts.split.test_indices,
            test_perturbations=test_perts,
            gene_symbols=final_gene_names,
            perturbation_key=perturbation_key,
        )
    else:
        final_gene_names = None
        decode_lib_size = None
        logger.warning(
            "Gene-space eval skipped: state backbone %r has no decode(). "
            "Falling back to embedding-space metrics with synthetic dim_* "
            "var_names; DEG-based metrics will not be biologically "
            "interpretable. Consider adding an in-context-generation "
            "adapter for STACK if gene-space eval is required.",
            getattr(backbone, "name", "?"),
        )
        var_names = [f"dim_{i}" for i in range(int(full.expression.shape[1]))]
        real_adata = build_real_anndata(
            expression=full.expression,
            perturbation_labels=full.perturbation_labels,
            test_indices=artifacts.split.test_indices,
            test_perturbations=test_perts,
            gene_symbols=var_names,
            perturbation_key=perturbation_key,
        )

    def _maybe_decode_preds(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Decode embedding-space predictions to gene space when applicable."""
        if not gene_space_eval or embedding_eq_genes or not preds:
            return preds
        keys = list(preds.keys())
        stacked = np.stack([preds[k] for k in keys], axis=0).astype(np.float32)
        decoded = np.asarray(
            backbone.decode(
                stacked,
                gene_names=list(final_gene_names),
                lib_size=decode_lib_size,
            ),
            dtype=np.float32,
        )
        return {k: decoded[i] for i, k in enumerate(keys)}

    if eval_cfg.save_predictions:
        try:
            real_adata.write(eval_dir / "real.h5ad")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not write real.h5ad (%s).", e)

    test_state, test_actions = _build_test_inputs(
        real_adata,
        artifacts,
        perturbation_key=perturbation_key,
    )

    train_perts = list(artifacts.split.train_perturbations or [])
    perturbation_to_action_train = _build_perturbation_to_action(
        artifacts.indexer,
        artifacts.gene_table,
        train_perts,
    )
    perturbation_to_action_test = _build_perturbation_to_action(
        artifacts.indexer,
        artifacts.gene_table,
        test_perts,
    )

    results: dict[str, EvaluationResult] = {}
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if model is not None:
        logger.info("Evaluating world model on %d test perturbations.", len(test_perts))
        from ..models.incontext_world_model import InContextWorldModel

        predict_fn = _predict_incontext if isinstance(model, InContextWorldModel) else _predict_with_world_model
        wm_per_cell = predict_fn(
            model,
            artifacts,
            real_adata=real_adata,
            perturbation_key=perturbation_key,
            n_samples_per_pert=eval_cfg.n_control_samples_per_pert,
            device=device,
        )
        wm_preds = _predictions_to_dict(wm_per_cell, test_actions, full.control_label)
        wm_preds = _maybe_decode_preds(wm_preds)
        wm_pred_adata = build_pred_anndata(wm_preds, real_adata, perturbation_key=perturbation_key)
        per_pert, agg = run_cell_eval(
            real_adata,
            wm_pred_adata,
            perturbation_key=perturbation_key,
            control_label=control_label,
            deg_top_k=eval_cfg.deg_top_k,
            use_cell_eval=eval_cfg.use_cell_eval,
        )
        results["world_model"] = EvaluationResult(
            name="world_model",
            per_perturbation=per_pert,
            aggregate=agg,
            real_adata=real_adata,
            pred_adata=wm_pred_adata,
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
        preds_dict = _maybe_decode_preds(preds_dict)
        pred_adata = build_pred_anndata(preds_dict, real_adata, perturbation_key=perturbation_key)
        per_pert, agg = run_cell_eval(
            real_adata,
            pred_adata,
            perturbation_key=perturbation_key,
            control_label=control_label,
            deg_top_k=eval_cfg.deg_top_k,
            use_cell_eval=eval_cfg.use_cell_eval,
        )
        results[name] = EvaluationResult(
            name=name,
            per_perturbation=per_pert,
            aggregate=agg,
            real_adata=real_adata,
            pred_adata=pred_adata,
        )
        if eval_cfg.save_predictions:
            try:
                pred_adata.write(eval_dir / f"pred_{name}.h5ad")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not write pred_%s.h5ad (%s).", name, e)

    return results


__all__ = ["EvaluationResult", "run_evaluation"]
