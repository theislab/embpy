"""Pre-compute action embeddings for the perturbations of a dataset.

This is the recommended smoke test before launching a long training
run with a brand-new ``model_name``: it materialises and caches the
embedding for every unique perturbation in the AnnData so the first
training epoch starts immediately instead of waiting on resolver +
forward passes.

Usage:

    python -m embpy.world_model.scripts.embed_perturbations \\
        --dataset replogle \\
        --h5ad data/datasets/replogle/replogle_2022_k562_essential.h5ad \\
        --model esm2_650M \\
        --output outputs/_cache/action_embeddings/esm2_650M/full_mean_human.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from embpy.world_model.data.embeddings import (
    BioEmbedderProvider,
    EmbeddingCacheKey,
    PrecomputedProvider,
    save_cached,
)
from embpy.world_model.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute action embeddings.")
    parser.add_argument("--dataset", choices=["replogle", "nadig"], required=True)
    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="MODEL_REGISTRY key, e.g. esm2_650M, borzoi_v0, minilm_l6_v2.")
    parser.add_argument("--organism", type=str, default="human")
    parser.add_argument("--region", choices=["full", "exons", "introns"], default="full")
    parser.add_argument("--pooling-strategy", type=str, default="mean")
    parser.add_argument("--id-type", choices=["symbol", "ensembl_id"], default="symbol")
    parser.add_argument("--resolver-backend", choices=["api", "local"], default="api")
    parser.add_argument("--cache-dir", type=str, default="outputs/_cache/action_embeddings")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional explicit NPZ path; defaults to cache_dir/<model>/<region>_<pool>_<organism>.npz.")
    parser.add_argument("--perturbation-key", type=str, default="perturbation")
    parser.add_argument("--control-label", type=str, default="non-targeting")
    return parser.parse_args(argv)


def _load_unique_perturbations(h5ad_path: str | Path, *, perturbation_key: str, control_label: str) -> list[str]:
    import anndata as ad  # noqa: PLC0415

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData not found: {path}")
    adata = ad.read_h5ad(path, backed="r")
    if perturbation_key not in adata.obs.columns:
        raise KeyError(f"{perturbation_key!r} not in adata.obs (got {list(adata.obs.columns)})")
    raw = adata.obs[perturbation_key].astype(str).values
    unique: list[str] = sorted(set(str(x) for x in raw) - {control_label})
    # Tolerate multi-gene encodings -- precompute every individual gene
    # so the GeneIndexer can encode any subset.
    individual: set[str] = set()
    for label in unique:
        for piece in label.replace(",", "+").split("+"):
            piece = piece.strip()
            if piece and piece != control_label:
                individual.add(piece)
    return sorted(individual)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(level=logging.INFO)

    symbols = _load_unique_perturbations(
        args.h5ad,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
    )
    logger.info("Found %d unique perturbation symbols in %s", len(symbols), args.h5ad)

    provider = BioEmbedderProvider(
        model_name=args.model,
        organism=args.organism,
        resolver_backend=args.resolver_backend,
        id_type=args.id_type,
        region=args.region,
        pooling_strategy=args.pooling_strategy,
        cache_dir=args.cache_dir,
    )

    embeddings = provider.embed(symbols)
    logger.info("Embedded %d / %d symbols at dim=%d (%d unresolved)",
                len(symbols) - len(provider._last_unresolved), len(symbols),
                embeddings.shape[1] if embeddings.size else 0,
                len(provider._last_unresolved))

    output_path = (
        Path(args.output) if args.output is not None
        else Path(args.cache_dir) / provider.cache_key.relative_path()
    )
    save_cached(
        Path(args.cache_dir),
        provider.cache_key,
        symbols,
        embeddings,
    )

    if args.output is not None:
        # Mirror the cache write to the user-specified output path so
        # downstream PrecomputedProvider can pick it up directly.
        np.savez(
            output_path,
            symbols=np.asarray(symbols, dtype=object),
            embeddings=embeddings.astype(np.float32),
        )
    logger.info("Wrote embeddings to %s", output_path)

    # Sanity probe: round-trip via PrecomputedProvider so callers know
    # the cache is parsable by the legacy code path.
    try:
        prov = PrecomputedProvider(output_path)
        table, indexer = prov.build_table(symbols[: min(5, len(symbols))])
        logger.info(
            "PrecomputedProvider round-trip OK: table shape=%s, indexer rows=%d",
            tuple(table.shape), len(indexer),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Round-trip sanity probe failed (%s); cache file is still valid.", e)


if __name__ == "__main__":  # pragma: no cover
    main()
