"""End-to-end smoke test: tiny config, CPU, all components.

Runs the same code path as a real training run -- no special branches.
Asserts that all expected output files exist before exiting. Designed
to finish in well under five minutes on a laptop.

Usage:

    python -m world_model.scripts.smoke_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from world_model.scripts.train import main as train_main
from world_model.utils.run_identity import resolve_latest_suffixed_run_dir

logger = logging.getLogger(__name__)


REQUIRED_FILES = (
    "train_log.csv",
    "report.md",
    "comparison.csv",
    "plots/loss_curves.png",
    "plots/comparison.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the world-model smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/world_model/world_model/configs/experiments/smoke.yaml",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the smoke test."""
    args = parse_args(argv)
    overrides = [*_smoke_asset_overrides(args.overrides), *args.overrides]
    train_argv = ["--config", args.config, *overrides]
    train_main(train_argv)

    output_dir = resolve_latest_suffixed_run_dir("runs/world_model/smoke")
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        path = output_dir / rel
        if not path.exists():
            missing.append(str(path))
    if missing:
        logger.error("Smoke test FAILED. Missing: %s", missing)
        sys.exit(1)
    logger.info("Smoke test PASSED. All expected outputs present in %s", output_dir)


def _smoke_asset_overrides(user_overrides: list[str]) -> list[str]:
    """Generate tiny h5ad + .emstore assets when the default files are absent."""
    explicit = {item.split("=", 1)[0] for item in user_overrides if "=" in item}
    if {"data.h5ad_path", "action_embedding.store_path"} & explicit:
        return []

    h5ad_path = Path("data/datasets/nadig/NadigOConner2024_jurkat.h5ad")
    store_path = Path("data/embeddings/gene_embeddings/genept/genept.emstore")
    if h5ad_path.exists() and store_path.exists():
        return []

    asset_dir = Path("runs/world_model/_smoke_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    tiny_h5ad = asset_dir / "tiny_smoke.h5ad"
    tiny_store = asset_dir / "tiny_genept.emstore"
    if not tiny_h5ad.exists():
        _write_tiny_h5ad(tiny_h5ad)
    if not tiny_store.exists():
        _write_tiny_store(tiny_store)
    return [
        f"data.h5ad_path={tiny_h5ad}",
        f"action_embedding.store_path={tiny_store}",
        "action_embedding.store_key=gene:smoke",
        "data.num_workers=0",
        "data.pin_memory=false",
        "eval.use_cell_eval=false",
    ]


def _write_tiny_h5ad(path: Path) -> None:
    import anndata as ad
    import numpy as np

    rng = np.random.default_rng(0)
    n_cells, n_genes = 96, 32
    var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    labels = ["non-targeting"] * 36 + ["GENE_000"] * 20 + ["GENE_001"] * 20 + ["GENE_002"] * 20
    x = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    # Add a tiny perturbation signal so eval metrics are finite.
    for offset, label in enumerate(("GENE_000", "GENE_001", "GENE_002"), start=0):
        rows = [i for i, value in enumerate(labels) if value == label]
        x[rows, offset] += 2.0
    adata = ad.AnnData(X=x, obs={"perturbation": labels})
    adata.var_names = var_names
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _write_tiny_store(path: Path) -> None:
    import numpy as np

    from embpy.io.result import EmbeddingProvenance, EmbeddingResult
    from embpy.store import EmbeddingStore

    ids = [f"GENE_{i:03d}" for i in range(3)]
    matrix = np.eye(3, 8, dtype=np.float32)
    result = EmbeddingResult(
        matrix=matrix,
        entity_ids=tuple(ids),
        entity_type="gene",
        id_scheme="symbol",
        provenance=EmbeddingProvenance(model="smoke"),
    )
    EmbeddingStore.from_results(result).write(path)


if __name__ == "__main__":  # pragma: no cover
    main()
