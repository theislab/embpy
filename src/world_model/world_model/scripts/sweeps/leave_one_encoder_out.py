"""Leave-one-encoder-out (LOEO) transfer sweep.

Two-stage protocol:

* Stage A -- pretrain reuse. For each unique encoder X in the grid,
  train one Nadig pretrain run via :func:`world_model.scripts.train.main`
  and save its final checkpoint at
  ``<output-root>/_pretrain/<X>/<run_name>_pretrain_final.pt``. This
  also produces the diagonal cell ``<X>__to__<X>``.

* Stage B -- fine-tune sweep. For every off-diagonal pair ``(X, Y)``,
  build a derived config that:
    - sets ``transfer.pretrain_action_encoder = X``,
    - sets ``transfer.finetune_action_encoder = Y``,
    - points ``transfer.pretrain_checkpoint`` at X's saved checkpoint
      (so the pretrain phase is *skipped*; the swap helper picks up
      from there),
    - sets ``transfer.swap_strategy`` to the selected strategy.

The CLI accepts ``--diagonal-only`` (just the diagonal -- a sanity
reproducibility check), ``--only`` to filter pairs (``X:Y,...``),
and ``--dry-run`` to print the planned cells without training.

Outputs::

    <output-root>/
      _pretrain/<X>/...          (Stage A diagonal runs + checkpoints)
      <strategy>/<X>__to__<Y>/   (Stage B per-cell run dirs)
      <strategy>/heatmaps/<metric>.png
      <strategy>/summary_long.csv
"""

# ruff: noqa: D103

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.configs import (
    ActionEmbeddingConfig,
    WorldModelConfig,
    load_yaml_config,
)
from world_model.evaluation.ablation import (
    ActionEncoderSpec,
    resolve_grid,
)
from world_model.scripts import train as train_script

logger = logging.getLogger(__name__)


VALID_STRATEGIES = ("none", "reset_adapter", "learn_alignment", "reset_all_action")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-encoder-out transfer sweep.")
    parser.add_argument("--base-config", required=True,
                        help="Transfer-mode WorldModelConfig YAML.")
    parser.add_argument("--grid", default=None,
                        help="Encoder grid YAML (defaults to DEFAULT_ACTION_ENCODER_GRID).")
    parser.add_argument("--strategy", default="reset_adapter", choices=VALID_STRATEGIES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only", default=None,
                        help="Comma-separated 'X:Y' pairs to KEEP.")
    parser.add_argument("--diagonal-only", action="store_true",
                        help="Only run the X==Y diagonal (sanity check).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-fn", default=None,
                        help="(internal) dotted path overriding train.main for tests.")
    return parser.parse_args(argv)


def _resolve_train_fn(dotted: str | None):  # type: ignore[no-untyped-def]
    if dotted is None:
        return train_script.main
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name:
        raise ValueError(f"--train-fn must be a dotted path; got {dotted!r}")
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


# ---------------------------------------------------------------------
# Per-cell config construction
# ---------------------------------------------------------------------


def _action_block_for(spec: ActionEncoderSpec, base_cfg: WorldModelConfig) -> ActionEmbeddingConfig:
    base_cache_dir = base_cfg.action_embedding.cache_dir or ActionEmbeddingConfig().cache_dir
    return ActionEmbeddingConfig(
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


def build_pretrain_config(
    base_cfg: WorldModelConfig,
    encoder: ActionEncoderSpec,
    *,
    output_root: Path,
    seed: int | None,
) -> WorldModelConfig:
    """Stage A config: standard transfer run with X on both sides, no swap."""
    cfg = deepcopy(base_cfg)
    cfg.mode = "transfer"
    cfg.transfer = dataclasses.replace(
        cfg.transfer,
        enabled=True,
        pretrain_action_encoder=_action_block_for(encoder, base_cfg),
        finetune_action_encoder=_action_block_for(encoder, base_cfg),
        swap_strategy="none",
        pretrain_checkpoint=None,
    )
    cfg.action_embedding = _action_block_for(encoder, base_cfg)
    cfg.output_dir = str(output_root / "_pretrain" / encoder.key)
    cfg.run_name = f"{base_cfg.run_name}_{encoder.key}"
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


def build_finetune_config(
    base_cfg: WorldModelConfig,
    pretrain_enc: ActionEncoderSpec,
    finetune_enc: ActionEncoderSpec,
    *,
    output_root: Path,
    pretrain_checkpoint: Path,
    strategy: str,
    seed: int | None,
) -> WorldModelConfig:
    """Stage B config: load X's checkpoint, set Y on the fine-tune side, apply swap."""
    cfg = deepcopy(base_cfg)
    cfg.mode = "transfer"
    cfg.transfer = dataclasses.replace(
        cfg.transfer,
        enabled=True,
        pretrain_action_encoder=_action_block_for(pretrain_enc, base_cfg),
        finetune_action_encoder=_action_block_for(finetune_enc, base_cfg),
        swap_strategy=strategy,
        pretrain_checkpoint=str(pretrain_checkpoint),
    )
    cfg.action_embedding = _action_block_for(finetune_enc, base_cfg)
    cell_dir = output_root / strategy / f"{pretrain_enc.key}__to__{finetune_enc.key}"
    cfg.output_dir = str(cell_dir)
    cfg.run_name = f"{base_cfg.run_name}_{pretrain_enc.key}_to_{finetune_enc.key}"
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


def _dump_yaml(cfg: WorldModelConfig, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------
# Pretrain checkpoint discovery
# ---------------------------------------------------------------------


def _expected_checkpoint_path(cfg: WorldModelConfig) -> Path:
    """Mirror of trainer._save: ``<output_dir>/<run_name>_pretrain/<run_name>_pretrain_final.pt``.

    The trainer used inside ``_run_transfer`` saves into its own output
    dir (``<run_dir>/pretrain``) with filename
    ``<run_name>_pretrain_final.pt``.
    """
    return Path(cfg.output_dir) / "pretrain" / f"{cfg.run_name}_pretrain_final.pt"


def _find_pretrain_checkpoint(run_dir: Path, fallback: Path) -> Path:
    """Return the trainer's saved checkpoint, falling back to a glob if naming changes."""
    if fallback.exists():
        return fallback
    candidates = sorted(run_dir.glob("**/pretrain/*_final.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No pretrain checkpoint under {run_dir}. Expected {fallback}."
    )


# ---------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_cell(
    cfg: WorldModelConfig,
    cfg_yaml: Path,
    *,
    train_fn,
    label: str,
) -> dict[str, Any]:
    cfg_yaml.parent.mkdir(parents=True, exist_ok=True)
    _dump_yaml(cfg, cfg_yaml)
    meta: dict[str, Any] = {"label": label, "started_at": _now_iso(), "config": str(cfg_yaml)}
    t0 = time.perf_counter()
    try:
        train_fn(["--config", str(cfg_yaml)])
        meta["status"] = "ok"
    except SystemExit as exc:
        meta["status"] = "failed" if (exc.code or 0) != 0 else "ok"
        if meta["status"] == "failed":
            meta["error"] = f"SystemExit({exc.code!r})"
    except BaseException as exc:  # noqa: BLE001
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
    finally:
        meta["ended_at"] = _now_iso()
        meta["wall_clock_s"] = time.perf_counter() - t0
    return meta


def _read_world_model_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "world_model_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def _summary_long_for_strategy(
    output_root: Path,
    encoder_grid: list[ActionEncoderSpec],
    strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for X in encoder_grid:
        for Y in encoder_grid:
            run_dir = output_root / strategy / f"{X.key}__to__{Y.key}"
            wm = _read_world_model_metrics(run_dir)
            cell_meta_path = run_dir / "_lone_cell.json"
            cell_meta = (
                json.loads(cell_meta_path.read_text())
                if cell_meta_path.exists() else {"status": "missing"}
            )
            if wm.empty:
                rows.append({
                    "pretrain": X.key,
                    "finetune": Y.key,
                    "strategy": strategy,
                    "metric": np.nan,
                    "value": np.nan,
                    "run_dir": str(run_dir),
                    "status": cell_meta.get("status", "missing"),
                })
                continue
            row0 = wm.iloc[0]
            for metric in wm.columns:
                rows.append({
                    "pretrain": X.key,
                    "finetune": Y.key,
                    "strategy": strategy,
                    "metric": metric,
                    "value": _to_float(row0[metric]),
                    "run_dir": str(run_dir),
                    "status": cell_meta.get("status", "ok"),
                })
    return pd.DataFrame(rows)


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _render_strategy_heatmaps(
    long_df: pd.DataFrame,
    encoder_grid: list[ActionEncoderSpec],
    out_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        return []
    metrics = sorted(m for m in long_df["metric"].dropna().unique().tolist())
    keys = [e.key for e in encoder_grid]
    out_paths: list[Path] = []
    for metric in metrics:
        sub = long_df[long_df["metric"] == metric]
        mat = np.full((len(keys), len(keys)), np.nan, dtype=np.float64)
        for _, row in sub.iterrows():
            try:
                i = keys.index(row["pretrain"])
                j = keys.index(row["finetune"])
            except ValueError:
                continue
            v = _to_float(row.get("value"))
            if v is not None:
                mat[i, j] = v
        fig, ax = plt.subplots(figsize=(0.6 * len(keys) + 2.5, 0.6 * len(keys) + 2.5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_yticklabels(keys)
        ax.set_title(f"{metric}: pretrain (rows) -> finetune (cols)")
        for i in range(len(keys)):
            for j in range(len(keys)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax)
        path = out_dir / f"{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)
    return out_paths


# ---------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------


def _parse_only_pairs(only: str | None, encoder_grid: list[ActionEncoderSpec]) -> set[tuple[str, str]] | None:
    if only is None:
        return None
    keys = {e.key for e in encoder_grid}
    pairs: set[tuple[str, str]] = set()
    for chunk in only.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--only pair must look like 'X:Y'; got {chunk!r}")
        x, y = chunk.split(":", 1)
        if x not in keys or y not in keys:
            raise KeyError(f"--only references unknown encoder key(s) in {chunk!r}")
        pairs.add((x, y))
    return pairs


def _enumerate_cells(
    encoder_grid: list[ActionEncoderSpec],
    *,
    diagonal_only: bool,
    only_pairs: set[tuple[str, str]] | None,
) -> list[tuple[ActionEncoderSpec, ActionEncoderSpec]]:
    cells: list[tuple[ActionEncoderSpec, ActionEncoderSpec]] = []
    for X in encoder_grid:
        for Y in encoder_grid:
            if diagonal_only and X.key != Y.key:
                continue
            if only_pairs is not None and (X.key, Y.key) not in only_pairs:
                continue
            cells.append((X, Y))
    return cells


# ---------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        force=True)

    base_cfg = load_yaml_config(args.base_config)
    encoder_grid = resolve_grid(grid_path=args.grid)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    only_pairs = _parse_only_pairs(args.only, encoder_grid)
    cells = _enumerate_cells(
        encoder_grid,
        diagonal_only=args.diagonal_only,
        only_pairs=only_pairs,
    )

    (output_root / "grid_resolved.json").write_text(json.dumps({
        "encoders": [e.as_dict() for e in encoder_grid],
        "strategy": args.strategy,
        "diagonal_only": bool(args.diagonal_only),
        "n_cells_planned": len(cells),
    }, indent=2, default=str))

    if args.dry_run:
        print(f"LOEO: strategy={args.strategy} grid_size={len(encoder_grid)} cells_planned={len(cells)}")
        for X, Y in cells:
            print(f"  - {X.key} -> {Y.key}")
        return 0

    train_fn = _resolve_train_fn(args.train_fn)

    # ----- Stage A: pretrain reuse, one per unique X actually needed.
    needed_pretrain_keys = sorted({X.key for X, _ in cells})
    enc_by_key = {e.key: e for e in encoder_grid}
    pretrain_meta: dict[str, dict[str, Any]] = {}
    pretrain_ckpt: dict[str, Path] = {}
    for x_key in needed_pretrain_keys:
        X = enc_by_key[x_key]
        pre_cfg = build_pretrain_config(
            base_cfg, X, output_root=output_root, seed=args.seed,
        )
        pre_yaml = Path(pre_cfg.output_dir) / "config.yaml"
        logger.info("=== LOEO Stage A: pretrain X=%s ===", x_key)
        meta = _run_cell(pre_cfg, pre_yaml, train_fn=train_fn, label=f"pretrain_{x_key}")
        meta["pretrain_key"] = x_key
        run_dir = Path(pre_cfg.output_dir)
        (run_dir / "_lone_pretrain.json").write_text(json.dumps(meta, indent=2, default=str))
        pretrain_meta[x_key] = meta
        if meta.get("status") == "ok":
            try:
                pretrain_ckpt[x_key] = _find_pretrain_checkpoint(
                    run_dir, _expected_checkpoint_path(pre_cfg),
                )
            except FileNotFoundError as exc:
                logger.error("Pretrain %s succeeded but no checkpoint found: %s", x_key, exc)
                meta["status"] = "failed"
                meta["error"] = str(exc)
        else:
            logger.error("Pretrain %s failed; skipping its row.", x_key)

    # ----- Stage B: fine-tune sweep, one cell per (X, Y).
    n_failed = 0
    for X, Y in cells:
        if X.key not in pretrain_ckpt:
            # The diagonal cell is also Stage A; skip duplicating it.
            if X.key == Y.key:
                continue
            logger.warning("Skipping cell %s -> %s (no pretrain ckpt).", X.key, Y.key)
            continue
        if X.key == Y.key:
            # Diagonal: Stage A already produced this cell's artifacts.
            continue
        ft_cfg = build_finetune_config(
            base_cfg, X, Y,
            output_root=output_root,
            pretrain_checkpoint=pretrain_ckpt[X.key],
            strategy=args.strategy,
            seed=args.seed,
        )
        ft_yaml = Path(ft_cfg.output_dir) / "config.yaml"
        logger.info("=== LOEO Stage B: %s -> %s (strategy=%s) ===",
                    X.key, Y.key, args.strategy)
        meta = _run_cell(
            ft_cfg, ft_yaml, train_fn=train_fn,
            label=f"{X.key}_to_{Y.key}",
        )
        meta["pretrain"] = X.key
        meta["finetune"] = Y.key
        meta["strategy"] = args.strategy
        run_dir = Path(ft_cfg.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_lone_cell.json").write_text(json.dumps(meta, indent=2, default=str))
        if meta.get("status") == "failed":
            n_failed += 1

    # Mirror diagonal cells under <strategy>/<X>__to__<X>/ so the long
    # CSV finds them. We symlink the world-model metrics rather than
    # copying to keep the protocol cheap.
    diag_root = output_root / args.strategy
    diag_root.mkdir(parents=True, exist_ok=True)
    for x_key, ckpt in pretrain_ckpt.items():
        diag_cell = diag_root / f"{x_key}__to__{x_key}"
        diag_cell.mkdir(parents=True, exist_ok=True)
        src_metrics = ckpt.parent.parent / "world_model_metrics.csv"
        dst_metrics = diag_cell / "world_model_metrics.csv"
        if src_metrics.exists() and not dst_metrics.exists():
            try:
                dst_metrics.symlink_to(src_metrics.resolve())
            except OSError:
                # Fallback to a small copy on platforms without symlinks.
                dst_metrics.write_bytes(src_metrics.read_bytes())
        (diag_cell / "_lone_cell.json").write_text(json.dumps({
            "pretrain": x_key,
            "finetune": x_key,
            "strategy": args.strategy,
            "status": pretrain_meta.get(x_key, {}).get("status", "missing"),
            "linked_from": str(src_metrics),
        }, indent=2, default=str))

    long_df = _summary_long_for_strategy(output_root, encoder_grid, args.strategy)
    long_df.to_csv(output_root / args.strategy / "summary_long.csv", index=False)
    _render_strategy_heatmaps(long_df, encoder_grid, output_root / args.strategy / "heatmaps")

    logger.info("LOEO done. Stage-B failures=%d.", n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
