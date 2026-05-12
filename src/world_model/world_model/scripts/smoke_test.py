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

logger = logging.getLogger(__name__)


REQUIRED_FILES = (
    "train_log.csv",
    "report.md",
    "comparison.csv",
    "plots/loss_curves.png",
    "plots/comparison.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the world-model smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/world_model/world_model/configs/experiments/smoke.yaml",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_argv = ["--config", args.config, *args.overrides]
    train_main(train_argv)

    output_dir = Path("outputs/world_model/smoke")
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        path = output_dir / rel
        if not path.exists():
            missing.append(str(path))
    if missing:
        logger.error("Smoke test FAILED. Missing: %s", missing)
        sys.exit(1)
    logger.info("Smoke test PASSED. All expected outputs present in %s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
