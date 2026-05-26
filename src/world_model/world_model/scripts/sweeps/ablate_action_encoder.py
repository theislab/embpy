"""Action-encoder ablation runner.

For each spec in the (filtered) grid:

1. Load the base ``WorldModelConfig`` from ``--base-config``.
2. Override only ``action_embedding.{source,model_name,id_type,region,pooling_strategy,extra_kwargs}``
   plus the per-spec ``output_dir`` / ``run_name``. Source is forced to ``bio_embedder``.
3. Pin ``split.cache_path`` to a single shared NPZ pre-computed once on
   the base config so every run sees byte-identical train/test indices.
4. Pin ``action_embedding.cache_dir`` to the shared default so any spec
   that has been embedded before is loaded from disk.
5. Dump the resolved per-spec config to ``<run_dir>/config.yaml`` and
   call :func:`world_model.scripts.train.main` in-process.
6. Capture wall-clock + peak GPU memory and write
   ``<run_dir>/_ablation_run.json`` regardless of success / failure.

After the loop, the aggregator (``evaluation.ablation.aggregate``) is
invoked to produce ``summary_long.csv``, ``summary_wide.csv``,
``summary.json``, ``plots/`` and ``report.md`` under
``--output-root``.
"""

# ruff: noqa: D103

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from world_model.configs import (
    ActionEmbeddingConfig,
    WorldModelConfig,
    load_yaml_config,
)
from world_model.data import build_dataloaders
from world_model.evaluation.ablation import (
    ActionEncoderSpec,
    aggregate_ablation,
    resolve_grid,
)
from world_model.evaluation.ablation.aggregate import (
    persist_summary,
    render_plots,
    write_report,
)
from world_model.scripts import train as train_script

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the action-encoder ablation sweep.")
    parser.add_argument("--base-config", required=True,
                        help="Path to the base WorldModelConfig YAML.")
    parser.add_argument("--grid", default=None,
                        help="YAML grid file. Defaults to DEFAULT_ACTION_ENCODER_GRID.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated grid keys to KEEP.")
    parser.add_argument("--skip", default=None, help="Comma-separated grid keys to DROP.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned per-spec configs and exit before training.")
    parser.add_argument(
        "--train-fn",
        default=None,
        help="(internal) dotted path overriding world_model.scripts.train.main "
             "for testing.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------
# Per-spec config construction
# ---------------------------------------------------------------------


def build_spec_config(
    base_cfg: WorldModelConfig,
    spec: ActionEncoderSpec,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
) -> WorldModelConfig:
    """Deep-copy the base config and overlay the spec's overrides.

    Every change is documented inline; nothing else is touched.
    """
    cfg = deepcopy(base_cfg)
    base_cache_dir = cfg.action_embedding.cache_dir or ActionEmbeddingConfig().cache_dir
    cfg.action_embedding = ActionEmbeddingConfig(
        source="bio_embedder",
        model_name=spec.model_name,
        organism=base_cfg.action_embedding.organism,
        id_type=spec.id_type,
        region=spec.region,
        pooling_strategy=spec.pooling,
        resolver_backend=base_cfg.action_embedding.resolver_backend,
        mart_file=base_cfg.action_embedding.mart_file,
        chromosome_folder=base_cfg.action_embedding.chromosome_folder,
        device=base_cfg.action_embedding.device,
        cache_dir=base_cache_dir,
        extra_kwargs=dict(spec.extra_kwargs),
    )
    cfg.output_dir = str(output_root / spec.key)
    cfg.run_name = f"{base_cfg.run_name}_{spec.key}"
    if seed is not None:
        cfg.seed = int(seed)
    # Pin every run to the shared split file so the test indices are
    # byte-identical across specs.
    cfg.split = dataclasses.replace(cfg.split, cache_path=str(shared_split_path))
    return cfg


def _dump_yaml(cfg: WorldModelConfig, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------
# Pre-computed shared split
# ---------------------------------------------------------------------


def _shared_split_path(output_root: Path, base_cfg: WorldModelConfig) -> Path:
    return output_root / "_shared_split" / f"{base_cfg.data.dataset}.npz"


def precompute_shared_split(
    base_cfg: WorldModelConfig,
    output_root: Path,
) -> tuple[Path, str]:
    """Build (or load) the shared split NPZ and return (path, sha256).

    The runner uses the base ``action_embedding`` block when materialising
    the dataset for the split-only computation -- the contents of the
    embedding table do not affect the split, so the cheapest path is to
    re-use whatever the base config points at.
    """
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
            f"Check data.h5ad_path and split configuration in the base YAML."
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
    return datetime.now(UTC).isoformat()


def _resolve_train_fn(dotted: str | None):  # type: ignore[no-untyped-def]
    if dotted is None:
        return train_script.main
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name:
        raise ValueError(f"--train-fn must be a dotted path; got {dotted!r}")
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _peak_gpu_memory_mb() -> float | None:
    try:
        import torch
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
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def run_spec(
    spec: ActionEncoderSpec,
    base_cfg: WorldModelConfig,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
    shared_split_sha: str,
    train_fn,
) -> dict[str, Any]:
    """Run a single spec in-process. Returns the ``_ablation_run.json`` dict."""
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
        "model_name": spec.model_name,
        "id_type": spec.id_type,
        "region": spec.region,
        "pooling": spec.pooling,
        "extra_kwargs": dict(spec.extra_kwargs),
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
        # Critical: persist the run metadata even on failure so the
        # aggregator can attribute the missing artifacts to a spec.
        (run_dir / "_ablation_run.json").write_text(json.dumps(meta, indent=2, default=str))

    # Verify the split file is unchanged after the run -- guards against
    # a silent re-seed if a future train.py refactor recomputes it.
    if Path(shared_split_path).exists():
        sha_after = _sha256(Path(shared_split_path))
        if sha_after != shared_split_sha:
            logger.warning(
                "Shared split sha changed mid-sweep (%s -> %s); subsequent runs "
                "will not be comparable. Did train.py overwrite the split?",
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
    grid = resolve_grid(grid_path=args.grid, only=args.only, skip=args.skip)
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
        logger.info("=== Ablation: %s (model=%s) ===", spec.key, spec.model_name)
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
            "Spec %s: status=%s wall_clock=%.1fs", spec.key, meta.get("status"), meta.get("wall_clock_s", 0.0),
        )

    long_df, wide_df = aggregate_ablation(output_root, grid)
    persist_summary(output_root, grid, long_df, wide_df)
    plot_paths = render_plots(wide_df, output_root / "plots")
    write_report(output_root, grid, long_df, wide_df, plot_paths)

    n_failed = sum(1 for m in metas if m.get("status") == "failed")
    logger.info("Sweep done. %d/%d specs failed.", n_failed, len(metas))
    return 0 if n_failed == 0 else 1


def _dry_run(
    base_cfg: WorldModelConfig,
    grid: list[ActionEncoderSpec],
    *,
    output_root: Path,
    seed: int | None,
) -> int:
    """Print every per-spec config without training. Useful as a sanity check."""
    shared_split_path = _shared_split_path(output_root, base_cfg)
    print(f"Base config: {base_cfg.run_name} ({base_cfg.data.dataset})")
    print(f"Output root: {output_root}")
    print(f"Shared split path: {shared_split_path}")
    print(f"Specs ({len(grid)}):")
    for spec in grid:
        cfg = build_spec_config(
            base_cfg, spec,
            output_root=output_root,
            seed=seed,
            shared_split_path=shared_split_path,
        )
        print(f"  - key={spec.key} -> output_dir={cfg.output_dir}")
        print(f"      action_embedding.source     = {cfg.action_embedding.source}")
        print(f"      action_embedding.model_name = {cfg.action_embedding.model_name}")
        print(f"      action_embedding.id_type    = {cfg.action_embedding.id_type}")
        print(f"      action_embedding.region     = {cfg.action_embedding.region}")
        print(f"      action_embedding.pooling    = {cfg.action_embedding.pooling_strategy}")
        if spec.extra_kwargs:
            print(f"      extra_kwargs = {dict(spec.extra_kwargs)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
