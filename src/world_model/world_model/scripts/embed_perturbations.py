"""Pre-compute action embeddings for the perturbations of a dataset.

This is the recommended smoke test before launching a long training
run with a brand-new ``model_name``: it materialises and caches the
embedding for every unique perturbation in the AnnData so the first
training epoch starts immediately instead of waiting on resolver +
forward passes.

Part A.3 changes (control / status handling):

* The classic ``--control-label non-targeting`` (exact literal) is
  replaced by :class:`ControlPolicy`, which matches the curated regex
  set in :data:`embpy.resources.gene.control.DEFAULT_CONTROL_PATTERNS`
  case-insensitively. So ``NTC``, ``NT5``, ``non-targeting_1``,
  ``AAVS1``, ``control``, and similar variants are now ALL classified
  as control and never sent to BioEmbedder. Real genes that happen to
  start with NT (``NT5C2``, ``NTRK1``) are NOT controls.
* The script writes a sidecar ``<output>.status.json`` enumerating the
  count and members of each ``EmbeddingStatus`` bucket so a quick
  ``cat`` confirms the embedder saw what you expected.
* ``--fail-on-unresolved`` exits with code 2 (non-zero) if any
  embedded row would be UNRESOLVED. Default off; turn it on in cluster
  jobs so silent data-quality regressions cannot reach training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from embpy.resources.gene.control import ControlPolicy
from world_model.data.embeddings import (
    BioEmbedderProvider,
    EmbeddingStatus,
    PrecomputedProvider,
    describe_status_counts,
    save_cached,
)
from world_model.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute action embeddings.")
    parser.add_argument("--dataset", choices=["replogle", "nadig"], required=True)
    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=True,
        help="MODEL_REGISTRY key, e.g. esm2_650M, borzoi_v0, minilm_l6_v2.",
    )
    parser.add_argument("--organism", type=str, default="human")
    parser.add_argument(
        "--region", choices=["full", "exons", "introns"], default="full",
    )
    parser.add_argument("--pooling-strategy", type=str, default="mean")
    parser.add_argument(
        "--id-type", choices=["symbol", "ensembl_id"], default="symbol",
    )
    parser.add_argument(
        "--resolver-backend", choices=["api", "local"], default="api",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="runs/_cache/action_embeddings",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional explicit NPZ path; defaults to "
             "cache_dir/<model>/<region>_<pool>_<organism>.npz.",
    )
    parser.add_argument("--perturbation-key", type=str, default="perturbation")
    parser.add_argument(
        "--control-extra-labels", nargs="*", default=[],
        help="Extra dataset-specific labels to treat as control. The "
             "curated default regex set (non-targeting*, NTC*, AAVS1*, "
             "control*, ctrl*, safe-harbor*, scramble*, empty-vector*) "
             "applies regardless of this flag.",
    )
    parser.add_argument(
        "--control-sentinel-seed", type=int, default=0,
        help="Seed for the deterministic CONTROL sentinel vector.",
    )
    parser.add_argument(
        "--fail-on-unresolved", action="store_true",
        help="Exit non-zero if any row is UNRESOLVED. Recommended for "
             "cluster pre-warm jobs.",
    )
    parser.add_argument(
        "--control-strict", action="store_true",
        help="If a label mixes control + gene components, raise. Off by "
             "default to accept the rare mixed sgRNA library labels.",
    )
    return parser.parse_args(argv)


def _load_unique_perturbations(
    h5ad_path: str | Path,
    *,
    perturbation_key: str,
    control_policy: ControlPolicy,
) -> tuple[list[str], int]:
    """Return the deduplicated list of gene-side labels and the control count.

    The returned list contains *gene* (and "mixed" gene-component)
    labels only; pure-control labels are excluded so they never reach
    the embedder. Combos are kept in their raw form because the
    provider's ``embed_with_status`` will re-split them.
    """
    import anndata as ad  # noqa: PLC0415

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData not found: {path}")
    adata = ad.read_h5ad(path, backed="r")
    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"{perturbation_key!r} not in adata.obs "
            f"(got {list(adata.obs.columns)})"
        )
    raw = adata.obs[perturbation_key].astype(str).values
    seen: set[str] = set()
    ordered: list[str] = []
    n_control_cells = 0
    for label in raw:
        s = str(label)
        c = control_policy.classify(s)
        if c.kind == "control":
            n_control_cells += 1
            continue
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered, n_control_cells


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.INFO)

    policy = ControlPolicy.from_iterable(
        args.control_extra_labels,
        strict=bool(args.control_strict),
    )

    perturbation_labels, n_control_cells = _load_unique_perturbations(
        args.h5ad,
        perturbation_key=args.perturbation_key,
        control_policy=policy,
    )
    logger.info(
        "Found %d unique non-control perturbation labels in %s "
        "(filtered out %d control cells).",
        len(perturbation_labels), args.h5ad, n_control_cells,
    )

    # Sanity check: assert no control variant slipped through the
    # filter. Easier to detect a logic bug here than after a 3-hour
    # Borzoi forward pass.
    leaked = [
        s for s in perturbation_labels
        if policy.classify(s).kind == "control"
    ]
    if leaked:
        raise RuntimeError(
            f"ControlPolicy filtering is inconsistent: {len(leaked)} "
            f"labels were kept as non-control but classify back to control. "
            f"First 10: {leaked[:10]}."
        )

    provider = BioEmbedderProvider(
        model_name=args.model,
        organism=args.organism,
        resolver_backend=args.resolver_backend,
        id_type=args.id_type,
        region=args.region,
        pooling_strategy=args.pooling_strategy,
        cache_dir=args.cache_dir,
        control_policy=policy,
        control_sentinel_seed=int(args.control_sentinel_seed),
    )

    embeddings, statuses = provider.embed_with_status(perturbation_labels)
    counts = describe_status_counts(
        [EmbeddingStatus(s) for s in statuses.tolist()]
    )
    logger.info(
        "Embedded %d / %d rows at dim=%d (RESOLVED=%d, CONTROL=%d, "
        "UNRESOLVED=%d).",
        counts.get(EmbeddingStatus.RESOLVED.value, 0)
        + counts.get(EmbeddingStatus.CONTROL.value, 0),
        len(perturbation_labels),
        embeddings.shape[1] if embeddings.size else 0,
        counts.get(EmbeddingStatus.RESOLVED.value, 0),
        counts.get(EmbeddingStatus.CONTROL.value, 0),
        counts.get(EmbeddingStatus.UNRESOLVED.value, 0),
    )

    output_path = (
        Path(args.output) if args.output is not None
        else Path(args.cache_dir) / provider.cache_key.relative_path()
    )
    save_cached(
        Path(args.cache_dir),
        provider.cache_key,
        perturbation_labels,
        embeddings,
    )
    if args.output is not None:
        np.savez(
            output_path,
            symbols=np.asarray(perturbation_labels, dtype=object),
            embeddings=embeddings.astype(np.float32),
            statuses=statuses,
        )
    logger.info("Wrote embeddings to %s", output_path)

    # Write a structured sidecar JSON next to the NPZ so the user can
    # ``cat`` it without booting Python.
    status_meta = {
        "model_name": args.model,
        "h5ad": str(args.h5ad),
        "n_rows": int(len(perturbation_labels)),
        "n_control_cells_filtered_upstream": int(n_control_cells),
        "counts": counts,
        "unresolved_symbols_first_50": list(provider._last_unresolved)[:50],
        "control_symbols_first_20": list(provider._last_controls)[:20],
        "control_sentinel_seed": int(args.control_sentinel_seed),
        "control_policy_patterns": list(policy.patterns),
        "control_extra_labels": list(policy.extra_labels),
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".status.json")
    sidecar.write_text(json.dumps(status_meta, indent=2, default=str))
    logger.info("Status sidecar -> %s", sidecar)

    # Round-trip sanity probe (precomputed loader).
    try:
        prov = PrecomputedProvider(
            output_path, control_policy=policy,
        )
        head = perturbation_labels[: min(5, len(perturbation_labels))]
        table, indexer = prov.build_table(head)
        logger.info(
            "PrecomputedProvider round-trip OK: table shape=%s, indexer rows=%d",
            tuple(table.shape), len(indexer),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Round-trip sanity probe failed (%s); cache file is still valid.", e,
        )

    if args.fail_on_unresolved and counts.get(EmbeddingStatus.UNRESOLVED.value, 0):
        logger.error(
            "Exiting non-zero because --fail-on-unresolved is set and "
            "%d rows are UNRESOLVED.",
            counts[EmbeddingStatus.UNRESOLVED.value],
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
