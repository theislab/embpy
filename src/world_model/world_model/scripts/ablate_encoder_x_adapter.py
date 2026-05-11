"""Cross-product sweep over the encoder grid (Phase 2) x adapter grid (Phase 3).

Default: 5 encoders x 6 adapters = 30 runs. Skipped by default; this
script must be invoked explicitly. Wall-clock budget on a single A100
with the default smoke-friendly knobs is roughly:

    * 30 cells * <per-cell wall-clock> + cache pre-warm time
    * For full Replogle + esm2_650M / borzoi_v0, expect 6-12 hours total.

Each cell reuses the shared train/test split and the action-embedding
disk cache, so re-running with one extra adapter or encoder only pays
for that handful of new cells.

Output:
    <output-root>/<encoder>__x__<adapter>/   per-cell run dir
    <output-root>/summary_long.csv           one row per (cell, metric)
    <output-root>/summary_wide.csv           one row per cell
    <output-root>/heatmaps/<metric>.png      5x6 heatmap per metric
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.configs import (
    ActionAdapterConfig,
    ActionEmbeddingConfig,
    WorldModelConfig,
    load_yaml_config,
)
from world_model.evaluation.ablation import (
    ActionAdapterSpec,
    ActionEncoderSpec,
    resolve_adapter_grid,
    resolve_grid,
)
from world_model.scripts import train as train_script
from world_model.scripts.ablate_action_encoder import (
    _resolve_train_fn,
    precompute_shared_split,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the encoder x adapter cross sweep.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--encoder-grid", default=None)
    parser.add_argument("--adapter-grid", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only-encoders", default=None)
    parser.add_argument("--only-adapters", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-fn", default=None)
    return parser.parse_args(argv)


def _cell_key(enc: ActionEncoderSpec, adp: ActionAdapterSpec) -> str:
    return f"{enc.key}__x__{adp.key}"


def build_cell_config(
    base_cfg: WorldModelConfig,
    enc: ActionEncoderSpec,
    adp: ActionAdapterSpec,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
) -> WorldModelConfig:
    cfg = deepcopy(base_cfg)
    base_cache_dir = cfg.action_embedding.cache_dir or ActionEmbeddingConfig().cache_dir
    cfg.action_embedding = ActionEmbeddingConfig(
        source="bio_embedder",
        model_name=enc.model_name,
        organism=base_cfg.action_embedding.organism,
        id_type=enc.id_type,
        region=enc.region,
        pooling_strategy=enc.pooling,
        resolver_backend=base_cfg.action_embedding.resolver_backend,
        mart_file=base_cfg.action_embedding.mart_file,
        chromosome_folder=base_cfg.action_embedding.chromosome_folder,
        device=base_cfg.action_embedding.device,
        cache_dir=base_cache_dir,
        extra_kwargs=dict(enc.extra_kwargs),
    )
    cfg.action_adapter = ActionAdapterConfig(
        kind=adp.kind,
        hidden_dim=adp.hidden_dim,
        dropout=adp.dropout,
        activation=adp.activation,
        lora_rank=adp.lora_rank,
        lora_alpha=adp.lora_alpha,
    )
    key = _cell_key(enc, adp)
    cfg.output_dir = str(output_root / key)
    cfg.run_name = f"{base_cfg.run_name}_{key}"
    if seed is not None:
        cfg.seed = int(seed)
    cfg.split = dataclasses.replace(cfg.split, cache_path=str(shared_split_path))
    return cfg


def _dump_yaml(cfg: WorldModelConfig, path: Path) -> None:
    import yaml  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cell(
    enc: ActionEncoderSpec,
    adp: ActionAdapterSpec,
    base_cfg: WorldModelConfig,
    *,
    output_root: Path,
    seed: int | None,
    shared_split_path: Path,
    train_fn,
) -> dict[str, Any]:
    cfg = build_cell_config(
        base_cfg, enc, adp,
        output_root=output_root,
        seed=seed,
        shared_split_path=shared_split_path,
    )
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = run_dir / "config.yaml"
    _dump_yaml(cfg, cfg_yaml)

    meta: dict[str, Any] = {
        "encoder": enc.key,
        "adapter": adp.key,
        "model_name": enc.model_name,
        "kind": adp.kind,
        "started_at": _now_iso(),
    }
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
        (run_dir / "_cell_run.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def _read_world_model_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "world_model_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df


def aggregate_cross(
    output_root: Path,
    encoder_grid: list[ActionEncoderSpec],
    adapter_grid: list[ActionAdapterSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for enc in encoder_grid:
        for adp in adapter_grid:
            key = _cell_key(enc, adp)
            run_dir = output_root / key
            wm = _read_world_model_metrics(run_dir)
            cell_meta_path = run_dir / "_cell_run.json"
            cell_meta = (
                json.loads(cell_meta_path.read_text())
                if cell_meta_path.exists() else {"status": "missing"}
            )
            wide: dict[str, Any] = {
                "cell_key": key,
                "encoder": enc.key,
                "adapter": adp.key,
                "model_name": enc.model_name,
                "kind": adp.kind,
                "status": cell_meta.get("status", "missing"),
                "wall_clock_s": cell_meta.get("wall_clock_s"),
                "run_dir": str(run_dir),
            }
            if not wm.empty:
                row0 = wm.iloc[0]
                for metric in wm.columns:
                    val = _to_float(row0[metric])
                    long_rows.append({
                        "encoder": enc.key,
                        "adapter": adp.key,
                        "metric": metric,
                        "value": val,
                        "run_dir": str(run_dir),
                    })
                    wide[str(metric)] = val
            else:
                long_rows.append({
                    "encoder": enc.key,
                    "adapter": adp.key,
                    "metric": np.nan,
                    "value": np.nan,
                    "run_dir": str(run_dir),
                })
            wide_rows.append(wide)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def render_heatmaps(
    long_df: pd.DataFrame,
    encoder_grid: list[ActionEncoderSpec],
    adapter_grid: list[ActionAdapterSpec],
    output_root: Path,
) -> list[Path]:
    """One heatmap per metric (rows = encoders, columns = adapters)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out_dir = output_root / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        return []
    out_paths: list[Path] = []
    metrics = sorted(m for m in long_df["metric"].dropna().unique().tolist())
    enc_keys = [e.key for e in encoder_grid]
    adp_keys = [a.key for a in adapter_grid]

    for metric in metrics:
        sub = long_df[long_df["metric"] == metric]
        mat = np.full((len(enc_keys), len(adp_keys)), np.nan, dtype=np.float64)
        for _, row in sub.iterrows():
            try:
                i = enc_keys.index(row["encoder"])
                j = adp_keys.index(row["adapter"])
            except ValueError:
                continue
            v = _to_float(row.get("value"))
            if v is not None:
                mat[i, j] = v
        fig, ax = plt.subplots(figsize=(0.7 * len(adp_keys) + 2.5, 0.5 * len(enc_keys) + 2.5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(adp_keys)))
        ax.set_yticks(range(len(enc_keys)))
        ax.set_xticklabels(adp_keys, rotation=30, ha="right")
        ax.set_yticklabels(enc_keys)
        ax.set_title(f"{metric}: encoder x adapter")
        for i in range(len(enc_keys)):
            for j in range(len(adp_keys)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax)
        path = out_dir / f"{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)
    return out_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        force=True)
    base_cfg = load_yaml_config(args.base_config)
    encoder_grid = resolve_grid(grid_path=args.encoder_grid, only=args.only_encoders)
    adapter_grid = resolve_adapter_grid(grid_path=args.adapter_grid, only=args.only_adapters)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "grid_resolved.json").write_text(json.dumps({
        "encoders": [s.as_dict() for s in encoder_grid],
        "adapters": [s.as_dict() for s in adapter_grid],
        "base_config": str(args.base_config),
    }, indent=2, default=str))

    if args.dry_run:
        print(f"Cross sweep: {len(encoder_grid)} encoders x {len(adapter_grid)} adapters = "
              f"{len(encoder_grid) * len(adapter_grid)} cells")
        for enc in encoder_grid:
            for adp in adapter_grid:
                print(f"  - {_cell_key(enc, adp)}")
        return 0

    shared_split_path, _ = precompute_shared_split(base_cfg, output_root)
    train_fn = _resolve_train_fn(args.train_fn)

    n_failed = 0
    for enc in encoder_grid:
        for adp in adapter_grid:
            logger.info("=== Cross cell: %s ===", _cell_key(enc, adp))
            meta = run_cell(
                enc, adp, base_cfg,
                output_root=output_root,
                seed=args.seed,
                shared_split_path=shared_split_path,
                train_fn=train_fn,
            )
            if meta.get("status") == "failed":
                n_failed += 1

    long_df, wide_df = aggregate_cross(output_root, encoder_grid, adapter_grid)
    long_df.to_csv(output_root / "summary_long.csv", index=False)
    wide_df.to_csv(output_root / "summary_wide.csv", index=False)
    render_heatmaps(long_df, encoder_grid, adapter_grid, output_root)

    logger.info("Cross sweep done. %d cells failed.", n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
