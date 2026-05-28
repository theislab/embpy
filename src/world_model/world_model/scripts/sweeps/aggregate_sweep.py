r"""Cross-embedding aggregator for the gene-embedding sweep.

Each embedding's run writes ``runs/world_model/single_<ds>_<emb>/comparison.csv``
(world_model row + every baseline row + cell_eval metrics) via
``compare.py``. This script stitches all of those into one table so you can
see, in a single view, how each embedding does against every other one and
against the baselines.

Outputs (under ``--out``, default ``runs/world_model/_sweep``):

* ``comparison_all.csv``        -- long: one row per (dataset, embedding,
                                   evaluator), all metric columns.
* ``world_model_by_embedding.csv`` -- wide: one row per (dataset, embedding)
                                   keeping only the ``world_model`` evaluator,
                                   sorted by the primary metric. This is the
                                   "which embedding wins" table.
* ``report.md``                 -- the same, rendered as Markdown.
* ``plots/sweep_<metric>.png``  -- boxplot per embedding (if matplotlib).

Re-runnable any time; it only reads CSVs already on disk, so missing /
still-running embeddings are simply skipped.

    python -m world_model.scripts.sweeps.aggregate_sweep \\
        --datasets replogle nadig --out runs/world_model/_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from world_model.utils import setup_logging

logger = logging.getLogger(__name__)

# Lower-is-better for these; everything else is higher-is-better.
_LOWER_IS_BETTER = {"mse", "mae"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Aggregate the embedding sweep.")
    p.add_argument("--runs-root", default="runs/world_model")
    p.add_argument("--datasets", nargs="+", default=["replogle", "nadig"])
    p.add_argument("--out", default="runs/world_model/_sweep")
    p.add_argument(
        "--primary-metric",
        default="deg_overlap@50",
        help="Metric used to rank embeddings in the wide table / report.",
    )
    return p.parse_args(argv)


def _emb_name(run_dir: Path, dataset: str) -> str:
    """``single_<ds>_<emb>[__job..__ts]`` -> ``<emb>``."""
    stem = run_dir.name
    for prefix in (
        f"single_{dataset}_",
        f"crossmod_incontext_{dataset}_",
        f"{dataset}_",
    ):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.split("__", 1)[0]


def _comparison_csvs(runs_root: Path, dataset: str) -> list[Path]:
    return sorted(
        [
            *runs_root.glob(f"single_{dataset}_*/comparison.csv"),
            *runs_root.glob(f"crossmod_incontext_{dataset}_*/comparison.csv"),
            *runs_root.glob(f"cross_modality_incontext/{dataset}_*/comparison.csv"),
        ]
    )


def _run_dirs(runs_root: Path, dataset: str) -> list[Path]:
    return sorted(
        [
            *[p for p in runs_root.glob(f"single_{dataset}_*") if p.is_dir()],
            *[p for p in runs_root.glob(f"crossmod_incontext_{dataset}_*") if p.is_dir()],
            *[p for p in runs_root.glob(f"cross_modality_incontext/{dataset}_*") if p.is_dir()],
        ]
    )


def _collect(runs_root: Path, datasets: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ds in datasets:
        for csv in _comparison_csvs(runs_root, ds):
            emb = _emb_name(csv.parent, ds)
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                logger.warning("Skipping unreadable %s (%s)", csv, e)
                continue
            df.insert(0, "run_dir", str(csv.parent))
            df.insert(0, "seed", _run_seed(csv.parent))
            df.insert(0, "embedding", emb)
            df.insert(0, "dataset", ds)
            rows.append(df)
            logger.info("Loaded %s (%d evaluators)", csv, len(df))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _run_seed(run_dir: Path) -> str:
    for rel in ("run_info.json", "config.yaml"):
        path = run_dir / rel
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text())
            else:
                import yaml

                payload = yaml.safe_load(path.read_text()) or {}
            seed = payload.get("seed")
            if seed is not None:
                return str(seed)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read seed from %s: %s", path, exc)
    return "unknown"


def _collect_per_perturbation(runs_root: Path, datasets: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ds in datasets:
        for run_dir in _run_dirs(runs_root, ds):
            emb = _emb_name(run_dir, ds)
            seed = _run_seed(run_dir)
            long_path = run_dir / "eval" / "per_perturbation_long.csv"
            if long_path.exists():
                try:
                    df = pd.read_csv(long_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping unreadable %s (%s)", long_path, exc)
                    continue
                df["dataset"] = df.get("dataset", ds)
                df["embedding"] = emb
                df["seed"] = df.get("seed", seed)
                df["run_dir"] = str(run_dir)
                rows.append(df)
                continue
            for csv in sorted((run_dir / "eval").glob("per_pert_*.csv")):
                try:
                    per = pd.read_csv(csv)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping unreadable %s (%s)", csv, exc)
                    continue
                evaluator = csv.stem.removeprefix("per_pert_")
                metric_cols = [c for c in per.columns if c != "perturbation" and pd.api.types.is_numeric_dtype(per[c])]
                if not metric_cols or "perturbation" not in per.columns:
                    continue
                melted = per.melt(
                    id_vars=["perturbation"],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )
                melted.insert(0, "seed", seed)
                melted.insert(0, "baseline", evaluator)
                melted.insert(0, "model", evaluator)
                melted.insert(0, "embedding", emb)
                melted.insert(0, "dataset", ds)
                melted["run_dir"] = str(run_dir)
                rows.append(melted)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Aggregate sweep artifacts from CLI arguments."""
    args = parse_args(argv)
    setup_logging(level=logging.INFO)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    long_df = _collect(Path(args.runs_root), list(args.datasets))
    if long_df.empty:
        logger.error(
            "No comparison.csv found under %s for datasets %s. Have the sweep runs finished?",
            args.runs_root,
            args.datasets,
        )
        return 1

    long_path = out / "comparison_all.csv"
    long_df.to_csv(long_path, index=False)
    logger.info("Wrote %s (%d rows)", long_path, len(long_df))

    per_pert_long = _collect_per_perturbation(Path(args.runs_root), list(args.datasets))
    per_pert_path = out / "per_perturbation_long.csv"
    if not per_pert_long.empty:
        per_pert_long.to_csv(per_pert_path, index=False)
        logger.info("Wrote %s (%d rows)", per_pert_path, len(per_pert_long))

    name_col = "name" if "name" in long_df.columns else long_df.columns[2]
    wm = long_df[long_df[name_col] == "world_model"].copy()
    metric = args.primary_metric
    numeric_metrics = [
        c
        for c in wm.columns
        if c not in {"dataset", "embedding", "seed", "run_dir", name_col} and pd.api.types.is_numeric_dtype(wm[c])
    ]
    if not wm.empty and numeric_metrics:
        wm = wm.groupby(["dataset", "embedding"], as_index=False)[numeric_metrics].median(numeric_only=True)
    if metric in wm.columns and not wm.empty:
        wm = wm.sort_values(["dataset", metric], ascending=[True, metric in _LOWER_IS_BETTER])
    wide_path = out / "world_model_by_embedding.csv"
    wm.to_csv(wide_path, index=False)
    logger.info("Wrote %s (%d embeddings)", wide_path, len(wm))

    # Markdown report.
    lines = [
        "# Gene-embedding sweep",
        "",
        f"Datasets: {', '.join(args.datasets)}  ",
        f"Primary metric: `{metric}` ({'lower' if metric in _LOWER_IS_BETTER else 'higher'} is better)",
        "",
        "## World-model performance by embedding",
        "",
        wm.to_markdown(index=False) if not wm.empty else "_no world_model rows_",
        "",
        "## Full comparison (world model + baselines, all embeddings)",
        "",
        long_df.to_markdown(index=False),
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines))
    logger.info("Wrote %s", report)

    # Optional plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = out / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_source = per_pert_long
        if plot_source.empty and metric in long_df.columns:
            rows = []
            for row in long_df.to_dict(orient="records"):
                value = row.get(metric)
                if value is None or pd.isna(value):
                    continue
                rows.append(
                    {
                        "dataset": row.get("dataset"),
                        "embedding": row.get("embedding"),
                        "model": row.get(name_col, "unknown"),
                        "seed": row.get("seed", "unknown"),
                        "perturbation": "aggregate",
                        "metric": metric,
                        "value": float(value),
                    }
                )
            plot_source = pd.DataFrame(rows)
        if plot_source.empty or "dataset" not in plot_source.columns:
            return 0
        for ds, g in plot_source.groupby("dataset"):
            sub = g[g["metric"] == metric].dropna(subset=["value"]).copy()
            if sub.empty:
                continue
            labels = sorted(sub["embedding"].astype(str).unique())
            data = [sub.loc[sub["embedding"].astype(str) == label, "value"].to_numpy(dtype=float) for label in labels]
            med = [float(np.nanmedian(v)) if len(v) else np.nan for v in data]
            order = np.argsort(med)
            if metric not in _LOWER_IS_BETTER:
                order = order[::-1]
            labels = [labels[i] for i in order]
            data = [data[i] for i in order]
            fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4))
            ax.boxplot(
                data,
                tick_labels=[f"{label}\nn={len(vals)}" for label, vals in zip(labels, data, strict=False)],
                showfliers=False,
            )
            rng = np.random.default_rng(0)
            for i, vals in enumerate(data, start=1):
                ax.scatter(i + rng.normal(0, 0.035, size=len(vals)), vals, s=14, alpha=0.6, color="#333333")
            ax.set_title(f"{ds}: {metric} by embedding")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fp = plots_dir / f"sweep_{ds}_{metric.replace('@', '')}.png"
            fig.savefig(fp, dpi=120)
            fig.savefig(fp.with_suffix(".svg"))
            plt.close(fig)
            logger.info("Wrote %s", fp)
    except Exception as e:  # noqa: BLE001
        logger.warning("Plot skipped (%s); CSV/MD are still written.", e)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
