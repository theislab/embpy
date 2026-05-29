"""Assemble one world-model-ready AnnData from cached action embeddings.

This script is the merge step for the SLURM embedding builder:

1. Start from an AnnData that already has state embeddings in ``.obsm``.
2. Load one or more action embedding NPZ files produced by
   ``world_model.scripts.embed_perturbations --output``.
3. Expand each unique-perturbation matrix to per-cell rows in ``.obsm``.
4. Write a single AnnData containing all requested model inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from embpy.resources.gene.control import ControlPolicy
from world_model.data.embeddings import (
    EmbeddingStatus,
    describe_status_counts,
    make_control_vector,
    make_unresolved_vector,
)
from world_model.utils import setup_logging

logger = logging.getLogger(__name__)
_VALID_STATUS_VALUES = {status.value for status in EmbeddingStatus}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach cached perturbation embeddings to a state-encoded AnnData.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--output-h5ad", required=True)
    parser.add_argument("--state-obsm-key", default="X_stack")
    parser.add_argument("--perturbation-key", default="perturbation")
    parser.add_argument(
        "--embedding",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Embedding NPZ to attach. May be repeated. The obsm key is X_pert_<NAME>.",
    )
    parser.add_argument(
        "--control-sentinel-seed",
        type=int,
        default=0,
        help="Seed for deterministic control sentinel vectors.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Exit non-zero if any embedding has unresolved perturbation rows.",
    )
    args = parser.parse_args(argv)
    if not args.embedding:
        parser.error("Pass at least one --embedding NAME=PATH.")
    return args


def _parse_embedding_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--embedding must be NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--embedding has an empty NAME: {value!r}")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding NPZ not found for {name!r}: {path}")
    return name, path


def _status_values(raw: np.ndarray, n: int) -> np.ndarray:
    if raw.size == 0:
        return np.asarray([EmbeddingStatus.RESOLVED.value] * n, dtype=object)
    values = []
    for item in raw.tolist():
        if isinstance(item, bytes):
            item = item.decode()
        text = str(item)
        if text not in _VALID_STATUS_VALUES and "." in text:
            tail = text.rsplit(".", 1)[-1]
            if tail in _VALID_STATUS_VALUES:
                text = tail
        values.append(text)
    return np.asarray(values, dtype=object)


def _load_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    if "symbols" not in archive or "embeddings" not in archive:
        raise KeyError(f"{path} must contain 'symbols' and 'embeddings' arrays.")
    labels = [str(x) for x in archive["symbols"].tolist()]
    embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"{path}: embeddings must be 2D, got {embeddings.shape!r}.")
    if embeddings.shape[0] != len(labels):
        raise ValueError(f"{path}: symbols length {len(labels)} does not match embeddings rows {embeddings.shape[0]}.")
    raw_statuses = archive["statuses"] if "statuses" in archive else np.asarray([], dtype=object)
    statuses = _status_values(np.asarray(raw_statuses, dtype=object), len(labels))
    if statuses.shape[0] != len(labels):
        raise ValueError(f"{path}: statuses length {statuses.shape[0]} does not match symbols length {len(labels)}.")
    return labels, embeddings, statuses


def _read_sidecar(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".status.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read status sidecar %s: %s", sidecar, exc)
        return {}


def _store_embpy_payload(
    adata,
    *,
    obsm_key: str,
    labels: list[str],
    embeddings: np.ndarray,
    model_name: str,
    npz_path: Path,
    sidecar: dict[str, object],
) -> None:
    try:
        from embpy.io import to_anndata
        from embpy.io.result import EmbeddingProvenance, EmbeddingResult

        result = EmbeddingResult(
            matrix=np.asarray(embeddings, dtype=np.float32),
            entity_ids=tuple(labels),
            entity_type="perturbation",
            id_scheme="perturbation_label",
            provenance=EmbeddingProvenance.create(
                model=model_name,
                extra={
                    "entity_type": "perturbation",
                    "source": sidecar.get("source", "cached_npz"),
                    "npz_path": str(npz_path),
                    "table": sidecar.get("table"),
                    "h5ad": sidecar.get("h5ad"),
                    "counts": sidecar.get("counts"),
                },
            ),
            aliases={label: {"perturbation_label": label} for label in labels},
        )
        to_anndata(result, target=adata, attach_to="uns", key=obsm_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not store embpy payload for %s: %s", obsm_key, exc)


def _attach_one(
    adata,
    *,
    name: str,
    npz_path: Path,
    perturbation_key: str,
    control_policy: ControlPolicy,
    control_sentinel_seed: int,
) -> dict[str, object]:
    labels, embeddings, statuses = _load_npz(npz_path)
    sidecar = _read_sidecar(npz_path)
    model_name = str(sidecar.get("model_name") or name)
    obsm_key = f"X_pert_{name}"

    if embeddings.shape[1] <= 0:
        raise ValueError(f"{npz_path}: cannot attach zero-dimensional embeddings.")
    dim = int(embeddings.shape[1])
    by_label = {label: embeddings[i].astype(np.float32) for i, label in enumerate(labels)}
    by_status = {label: str(statuses[i]) for i, label in enumerate(labels)}

    control_vec = make_control_vector(dim, seed=control_sentinel_seed)
    unresolved_vec = make_unresolved_vector(dim)
    obs_labels = adata.obs[perturbation_key].astype(str).to_numpy()
    cell_matrix = np.zeros((adata.n_obs, dim), dtype=np.float32)
    cell_status: list[str] = []
    n_control_cells = 0
    n_unresolved_cells = 0

    for i, label in enumerate(obs_labels):
        classification = control_policy.classify(str(label))
        if classification.kind == "control":
            cell_matrix[i] = control_vec
            cell_status.append(EmbeddingStatus.CONTROL.value)
            n_control_cells += 1
            continue
        vec = by_label.get(str(label))
        status = by_status.get(str(label), EmbeddingStatus.UNRESOLVED.value)
        if vec is None:
            cell_matrix[i] = unresolved_vec
            cell_status.append(EmbeddingStatus.UNRESOLVED.value)
            n_unresolved_cells += 1
        else:
            cell_matrix[i] = vec
            cell_status.append(status)
            if status == EmbeddingStatus.UNRESOLVED.value:
                n_unresolved_cells += 1

    adata.obsm[obsm_key] = cell_matrix
    status_col = f"{obsm_key}_status"
    adata.obs[status_col] = cell_status

    counts = describe_status_counts([EmbeddingStatus(str(s)) for s in statuses.tolist()])
    meta = {
        "model_name": model_name,
        "source": sidecar.get("source", "cached_npz"),
        "npz_path": str(npz_path),
        "obsm_key": obsm_key,
        "status_obs_column": status_col,
        "perturbation_key": perturbation_key,
        "embedding_dim": dim,
        "n_unique_non_control_perturbations": int(len(labels)),
        "n_cells": int(adata.n_obs),
        "n_control_cells": int(n_control_cells),
        "n_unresolved_cells": int(n_unresolved_cells),
        "counts": counts,
        "control_sentinel_seed": int(control_sentinel_seed),
        "sidecar": sidecar,
    }
    root = adata.uns.setdefault("world_model_action_embeddings", {})
    root[obsm_key] = meta
    _store_embpy_payload(
        adata,
        obsm_key=obsm_key,
        labels=labels,
        embeddings=embeddings,
        model_name=model_name,
        npz_path=npz_path,
        sidecar=sidecar,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    import anndata as ad

    in_path = Path(args.input_h5ad)
    if not in_path.exists():
        raise FileNotFoundError(f"Input AnnData not found: {in_path}")
    logger.info("Loading state AnnData from %s", in_path)
    adata = ad.read_h5ad(in_path)
    if args.state_obsm_key not in adata.obsm:
        raise KeyError(
            f"state obsm key {args.state_obsm_key!r} not found in {in_path}; available: {list(adata.obsm.keys())}"
        )
    if args.perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"perturbation key {args.perturbation_key!r} not found in obs; available: {list(adata.obs.columns)}"
        )

    policy = ControlPolicy.default()
    attached: list[dict[str, object]] = []
    for raw in args.embedding:
        name, path = _parse_embedding_arg(raw)
        logger.info("Attaching %s from %s", name, path)
        meta = _attach_one(
            adata,
            name=name,
            npz_path=path,
            perturbation_key=args.perturbation_key,
            control_policy=policy,
            control_sentinel_seed=int(args.control_sentinel_seed),
        )
        attached.append(meta)

    unresolved = {
        str(meta["obsm_key"]): int(meta["n_unresolved_cells"])
        for meta in attached
        if int(meta["n_unresolved_cells"]) > 0
    }
    if args.fail_on_unresolved and unresolved:
        raise RuntimeError(f"Unresolved perturbation rows found: {unresolved}")

    out_path = Path(args.output_h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    summary = {
        "output_h5ad": str(out_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "state_obsm_key": args.state_obsm_key,
        "action_obsm_keys": [str(meta["obsm_key"]) for meta in attached],
        "unresolved_cells": unresolved,
    }
    print(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote world-model-ready AnnData to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
