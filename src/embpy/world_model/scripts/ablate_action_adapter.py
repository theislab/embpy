"""Action-adapter ablation runner.

Sweeps over :class:`ActionAdapterSpec` rows, training one world model
per adapter variant. Re-uses everything else from the base config:

* Same train/test split (pinned via ``split.cache_path``).
* Same action-embedding cache (the foundation model is identical across
  the adapter sweep).
* Same dataset / encoder / dynamics / loss / optim.

For each spec we override only ``action_adapter.{kind, hidden_dim,
dropout, activation, lora_rank, lora_alpha}`` plus per-spec
``output_dir`` / ``run_name``. Wall-clock + peak GPU memory + status are
captured into ``<run_dir>/_ablation_run.json`` so the aggregator can
attribute failures to specific adapter kinds.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embpy.world_model.configs import (
    ActionAdapterConfig,
    WorldModelConfig,
    load_yaml_config,
)
from embpy.world_model.data import build_dataloaders
from embpy.world_model.evaluation.ablation import (
    ActionAdapterSpec,
    aggregate_adapter,
    resolve_adapter_grid,
)
from embpy.world_model.evaluation.ablation.aggregate import (
    persist_adapter_summary,
    render_adapter_plots,
    write_adapter_report,
)
from embpy.world_model.scripts import train as train_script

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the action-adapter ablation sweep.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--grid", default=None,
                        help="YAML grid file. Defaults to DEFAULT_ACTION_ADAPTER_GRID.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated grid keys to KEEP.")
    parser.add_argument("--skip", default=None, help="Comma-separated grid keys to DROP.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned per-spec configs and exit before training.")
    parser.add_argument(
        "--train-fn",
        default=None,
        help="(internal) dotted path overriding embpy.world_model.scripts.train.main.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------
# Per-spec config construction
# ---------------------------------------------------------------------


def build_spec_config(
    base_cfg: WorldModelConfig,
    spec: ActionAdapterSpec,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
) -> WorldModelConfig:
    """Deep-copy the base config and overlay the adapter spec."""
    cfg = deepcopy(base_cfg)
    cfg.action_adapter = ActionAdapterConfig(
        kind=spec.kind,
        hidden_dim=spec.hidden_dim,
        dropout=spec.dropout,
        activation=spec.activation,
        lora_rank=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
    )
    cfg.output_dir = str(output_root / spec.key)
    cfg.run_name = f"{base_cfg.run_name}_{spec.key}"
    if seed is not None:
        cfg.seed = int(seed)
    cfg.split = dataclasses.replace(cfg.split, cache_path=str(shared_split_path))
    return cfg


def _dump_yaml(cfg: WorldModelConfig, path: Path) -> None:
    import yaml  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------
# Shared split
# ---------------------------------------------------------------------


def _shared_split_path(output_root: Path, base_cfg: WorldModelConfig) -> Path:
    return output_root / "_shared_split" / f"{base_cfg.data.dataset}.npz"


def precompute_shared_split(
    base_cfg: WorldModelConfig,
    output_root: Path,
) -> tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    split_path = _shared_split_path(output_root, base_cfg)
    if not split_path.exists():
        split_cfg = dataclasses.replace(base_cfg.split, cache_path=str(split_path))
        build_dataloaders(
            base_cfg.data,
            split_cfg=split_cfg,
            action_cfg=base_cfg.action_embedding,
            seed=base_cfg.seed,
            output_dir=output_root,
        )
    if not split_path.exists():
        raise RuntimeError(
            f"Split pre-computation did not produce {split_path}. "
            f"Check data.h5ad_path and split configuration."
        )
    sha = _sha256(split_path)
    logger.info("Shared split NPZ: %s (sha256=%s)", split_path, sha)
    return split_path, sha


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Per-spec execution
# ---------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_train_fn(dotted: str | None):  # type: ignore[no-untyped-def]
    if dotted is None:
        return train_script.main
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name:
        raise ValueError(f"--train-fn must be a dotted path; got {dotted!r}")
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _peak_gpu_memory_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return None


def _reset_peak_gpu_memory() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def run_spec(
    spec: ActionAdapterSpec,
    base_cfg: WorldModelConfig,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
    shared_split_sha: str,
    train_fn,
) -> dict[str, Any]:
    cfg = build_spec_config(
        base_cfg, spec,
        output_root=output_root,
        seed=seed,
        shared_split_path=shared_split_path,
    )
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_yaml = run_dir / "config.yaml"
    _dump_yaml(cfg, spec_yaml)

    meta: dict[str, Any] = {
        "key": spec.key,
        "kind": spec.kind,
        "hidden_dim": spec.hidden_dim,
        "dropout": spec.dropout,
        "lora_rank": spec.lora_rank,
        "started_at": _now_iso(),
        "argv": ["--config", str(spec_yaml)],
        "shared_split_sha256": shared_split_sha,
    }

    _reset_peak_gpu_memory()
    t0 = time.perf_counter()
    try:
        train_fn(["--config", str(spec_yaml)])
        meta["status"] = "ok"
    except SystemExit as exc:
        meta["status"] = "failed" if (exc.code or 0) != 0 else "ok"
        if meta["status"] == "failed":
            meta["error"] = f"SystemExit({exc.code!r})"
            logger.error("Spec %s exited with code %r", spec.key, exc.code)
    except BaseException as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = tb
        logger.error("Spec %s failed: %s\n%s", spec.key, exc, tb)
    finally:
        meta["ended_at"] = _now_iso()
        meta["wall_clock_s"] = time.perf_counter() - t0
        meta["peak_gpu_mem_mb"] = _peak_gpu_memory_mb()
        (run_dir / "_ablation_run.json").write_text(json.dumps(meta, indent=2, default=str))

    if Path(shared_split_path).exists():
        sha_after = _sha256(Path(shared_split_path))
        if sha_after != shared_split_sha:
            logger.warning(
                "Shared split sha changed mid-sweep (%s -> %s); subsequent runs "
                "will not be comparable.",
                shared_split_sha, sha_after,
            )
    return meta


# ---------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    base_cfg = load_yaml_config(args.base_config)
    grid = resolve_adapter_grid(grid_path=args.grid, only=args.only, skip=args.skip)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grid_resolved_path = output_root / "grid_resolved.json"
    grid_resolved_path.write_text(json.dumps(
        {"specs": [s.as_dict() for s in grid], "base_config": str(args.base_config)},
        indent=2, default=str,
    ))

    if args.dry_run:
        return _dry_run(base_cfg, grid, output_root=output_root, seed=args.seed)

    shared_split_path, shared_split_sha = precompute_shared_split(base_cfg, output_root)
    train_fn = _resolve_train_fn(args.train_fn)

    metas: list[dict[str, Any]] = []
    for spec in grid:
        logger.info("=== Adapter ablation: %s (kind=%s) ===", spec.key, spec.kind)
        meta = run_spec(
            spec, base_cfg,
            output_root=output_root,
            seed=args.seed,
            shared_split_path=shared_split_path,
            shared_split_sha=shared_split_sha,
            train_fn=train_fn,
        )
        metas.append(meta)
        logger.info(
            "Spec %s: status=%s wall_clock=%.1fs",
            spec.key, meta.get("status"), meta.get("wall_clock_s", 0.0),
        )

    long_df, wide_df = aggregate_adapter(output_root, grid)
    persist_adapter_summary(output_root, grid, long_df, wide_df)
    plot_paths = render_adapter_plots(wide_df, output_root / "plots")
    write_adapter_report(output_root, grid, long_df, wide_df, plot_paths)

    n_failed = sum(1 for m in metas if m.get("status") == "failed")
    logger.info("Adapter sweep done. %d/%d specs failed.", n_failed, len(metas))
    return 0 if n_failed == 0 else 1


def _dry_run(
    base_cfg: WorldModelConfig,
    grid: list[ActionAdapterSpec],
    *,
    output_root: Path,
    seed: int | None,
) -> int:
    shared_split_path = _shared_split_path(output_root, base_cfg)
    print(f"Base config: {base_cfg.run_name} ({base_cfg.data.dataset})")
    print(f"Output root: {output_root}")
    print(f"Shared split path: {shared_split_path}")
    print(f"Adapter specs ({len(grid)}):")
    for spec in grid:
        cfg = build_spec_config(
            base_cfg, spec,
            output_root=output_root,
            seed=seed,
            shared_split_path=shared_split_path,
        )
        print(f"  - key={spec.key} -> output_dir={cfg.output_dir}")
        print(f"      action_adapter.kind        = {cfg.action_adapter.kind}")
        print(f"      action_adapter.hidden_dim  = {cfg.action_adapter.hidden_dim}")
        print(f"      action_adapter.dropout     = {cfg.action_adapter.dropout}")
        print(f"      action_adapter.lora_rank   = {cfg.action_adapter.lora_rank}")
        print(f"      action_adapter.lora_alpha  = {cfg.action_adapter.lora_alpha}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
