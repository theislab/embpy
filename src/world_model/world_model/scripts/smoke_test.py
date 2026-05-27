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
    """Generate a tiny h5ad with state/action obsm assets when defaults are absent."""
    explicit = {item.split("=", 1)[0] for item in user_overrides if "=" in item}
    if {"data.h5ad_path", "data.state_obsm_key", "action_embedding.obsm_key"} & explicit:
        return []

    h5ad_path = Path("data/datasets/nadig/NadigOConner2024_jurkat.h5ad")
    if _has_required_obsm(h5ad_path, state_key="X_state", action_key="X_pert_tiny"):
        return []

    asset_dir = Path("runs/world_model/_smoke_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    tiny_h5ad = asset_dir / "tiny_smoke.h5ad"
    if not tiny_h5ad.exists():
        _write_tiny_h5ad(tiny_h5ad)
    return [
        f"data.h5ad_path={tiny_h5ad}",
        "data.state_obsm_key=X_state",
        "action_embedding.source=anndata_obsm",
        "action_embedding.obsm_key=X_pert_tiny",
        "data.num_workers=0",
        "data.pin_memory=false",
        "eval.use_cell_eval=false",
    ]


def _has_required_obsm(path: Path, *, state_key: str, action_key: str) -> bool:
    if not path.exists():
        return False
    try:
        import anndata as ad

        adata = ad.read_h5ad(path, backed="r")
        try:
            return state_key in adata.obsm and action_key in adata.obsm
        finally:
            adata.file.close()
    except Exception:
        return False


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
    adata.obsm["X_state"] = x[:, :16].astype(np.float32)
    action_rows = {
        "non-targeting": np.zeros(8, dtype=np.float32),
        "GENE_000": np.eye(3, 8, dtype=np.float32)[0],
        "GENE_001": np.eye(3, 8, dtype=np.float32)[1],
        "GENE_002": np.eye(3, 8, dtype=np.float32)[2],
    }
    adata.obsm["X_pert_tiny"] = np.stack([action_rows[label] for label in labels], axis=0)
    adata.uns["world_model_state_embeddings"] = {
        "X_state": {
            "model_name": "tiny_smoke_state",
            "feature_names": [f"state_{i}" for i in range(16)],
        }
    }
    adata.uns["world_model_action_embeddings"] = {
        "X_pert_tiny": {
            "model_name": "tiny",
            "source": "synthetic",
            "obsm_key": "X_pert_tiny",
            "embedding_dim": 8,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


if __name__ == "__main__":  # pragma: no cover
    main()
