"""Cross-embedding aggregator for the gene-embedding sweep.

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
* ``plots/sweep_<metric>.png``  -- grouped bar per embedding (if matplotlib).

Re-runnable any time; it only reads CSVs already on disk, so missing /
still-running embeddings are simply skipped.

    python -m world_model.scripts.aggregate_sweep \\
        --datasets replogle nadig --out runs/world_model/_sweep
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from world_model.utils import setup_logging

logger = logging.getLogger(__name__)

# Lower-is-better for these; everything else is higher-is-better.
_LOWER_IS_BETTER = {"mse", "mae"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate the embedding sweep.")
    p.add_argument("--runs-root", default="runs/world_model")
    p.add_argument("--datasets", nargs="+", default=["replogle", "nadig"])
    p.add_argument("--out", default="runs/world_model/_sweep")
    p.add_argument(
        "--primary-metric", default="deg_overlap@50",
        help="Metric used to rank embeddings in the wide table / report.",
    )
    return p.parse_args(argv)


def _emb_name(run_dir: Path, dataset: str) -> str:
    """``single_<ds>_<emb>[__job..__ts]`` -> ``<emb>``."""
    stem = run_dir.name
    prefix = f"single_{dataset}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem.split("__", 1)[0]


def _collect(runs_root: Path, datasets: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ds in datasets:
        # Exact run dir first, then any job/timestamp-stamped variants.
        seen: dict[str, tuple[float, Path]] = {}
        for csv in sorted(runs_root.glob(f"single_{ds}_*/comparison.csv")):
            emb = _emb_name(csv.parent, ds)
            mtime = csv.stat().st_mtime
            # Keep the most recently written comparison.csv per embedding.
            if emb not in seen or mtime > seen[emb][0]:
                seen[emb] = (mtime, csv)
        for emb, (_, csv) in sorted(seen.items()):
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                logger.warning("Skipping unreadable %s (%s)", csv, e)
                continue
            df.insert(0, "embedding", emb)
            df.insert(0, "dataset", ds)
            rows.append(df)
            logger.info("Loaded %s (%d evaluators)", csv, len(df))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.INFO)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    long_df = _collect(Path(args.runs_root), list(args.datasets))
    if long_df.empty:
        logger.error(
            "No comparison.csv found under %s for datasets %s. "
            "Have the sweep runs finished?",
            args.runs_root, args.datasets,
        )
        return 1

    long_path = out / "comparison_all.csv"
    long_df.to_csv(long_path, index=False)
    logger.info("Wrote %s (%d rows)", long_path, len(long_df))

    name_col = "name" if "name" in long_df.columns else long_df.columns[2]
    wm = long_df[long_df[name_col] == "world_model"].copy()
    metric = args.primary_metric
    if metric in wm.columns and not wm.empty:
        wm = wm.sort_values(
            ["dataset", metric],
            ascending=[True, metric in _LOWER_IS_BETTER],
        )
    wide_path = out / "world_model_by_embedding.csv"
    wm.to_csv(wide_path, index=False)
    logger.info("Wrote %s (%d embeddings)", wide_path, len(wm))

    # Markdown report.
    lines = [
        "# Gene-embedding sweep",
        "",
        f"Datasets: {', '.join(args.datasets)}  ",
        f"Primary metric: `{metric}` "
        f"({'lower' if metric in _LOWER_IS_BETTER else 'higher'} is better)",
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

    # Optional plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if metric in wm.columns and not wm.empty:
            (out / "plots").mkdir(exist_ok=True)
            for ds, g in wm.groupby("dataset"):
                fig, ax = plt.subplots(
                    figsize=(max(6, 0.5 * len(g)), 4),
                )
                ax.bar(g["embedding"].astype(str), g[metric].astype(float))
                ax.set_title(f"{ds}: world_model {metric} by embedding")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=75)
                fig.tight_layout()
                fp = out / "plots" / f"sweep_{ds}_{metric.replace('@', '')}.png"
                fig.savefig(fp, dpi=120)
                plt.close(fig)
                logger.info("Wrote %s", fp)
    except Exception as e:  # noqa: BLE001
        logger.warning("Plot skipped (%s); CSV/MD are still written.", e)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
