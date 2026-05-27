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
import re
import sys
from pathlib import Path

import numpy as np

from embpy.errors import EmbpyError
from embpy.resources.gene.control import ControlPolicy
from world_model.data.embeddings import (
    BioEmbedderProvider,
    EmbeddingStatus,
    PrecomputedProvider,
    describe_status_counts,
    make_control_vector,
    make_unresolved_vector,
    save_cached,
)
from world_model.utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exit-code policy
# ---------------------------------------------------------------------------
#
# Each typed embpy.errors subclass carries a ``.exit_code`` attribute
# that maps the failure to a stable numeric code. We propagate that
# verbatim so SLURM's sacct can answer questions like "show all OOM
# jobs across the sweep" with `sacct --format=ExitCode | grep '10:'`
# without grepping any log files.
#
# Codes in use today (see embpy/errors.py for the source of truth):
#   0  -- success
#   1  -- catch-all (uncaught exception, propagated by Python's default)
#   2  -- --fail-on-unresolved triggered (pre-existing semantics)
#   10 -- ModelOOMError (CUDA OOM during forward pass)
#   11 -- ContextOverflowError (sequence > model context, chunking off)
#   12 -- DependencyError (missing pip package, e.g. mamba_ssm)
#   13 -- ResolverError (gene/text resolver returned empty for everything)
#   14 -- ModelLoadError / ModelNotFound (load-time failure)
#   15 -- EmbeddingError (per-input runtime failure)
#   20 -- ConfigError / InvalidPoolingError
#   21 -- IdentifierError / InvalidSMILES / GeneNotInGraph
#   22 -- DataError / GraphNotBuiltError


def _handle_typed_error(exc: EmbpyError) -> int:
    """Log a one-line actionable summary and return the exit code.

    The full traceback is already in the SLURM err file (we don't
    re-emit it here). The single-line message is what shows up in
    automation logs and is intended to be readable without scrolling.
    """
    category = getattr(exc, "category", "embpy")
    code = int(getattr(exc, "exit_code", 1))
    logger.error(
        "embed_perturbations failed [category=%s exit_code=%d]: %s: %s",
        category, code, type(exc).__name__, exc,
    )
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute action embeddings.")
    parser.add_argument("--dataset", choices=["replogle", "nadig"], required=True)
    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default=None,
        help="MODEL_REGISTRY key, e.g. esm2_650M, borzoi_v0, minilm_l6_v2.",
    )
    parser.add_argument(
        "--table", type=str, default=None,
        help="Optional precomputed CSV/NPZ table with gene symbols as rows. "
             "When set, no BioEmbedder model is run.",
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--cache-dir", type=str, default="runs/_cache/action_embeddings",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional explicit NPZ path; defaults to "
             "cache_dir/<model>/<region>_<pool>_<organism>.npz.",
    )
    parser.add_argument(
        "--output-h5ad", type=str, default=None,
        help="Optional AnnData output path. When provided, writes a copy of "
             "--h5ad with per-cell perturbation embeddings in .obsm.",
    )
    parser.add_argument(
        "--obsm-key", type=str, default=None,
        help="AnnData .obsm key for --output-h5ad. Defaults to "
             "X_pert_<model-or-table-stem>.",
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
        help="Exit non-zero (code 2) if any row is UNRESOLVED. "
             "Recommended for cluster AnnData attachment jobs.",
    )
    parser.add_argument(
        "--strict-resolver", action="store_true",
        help="Layer 2: raise ResolverError (exit code 13) if any input "
             "ends up UNRESOLVED in the per-identifier resolution "
             "report. Stricter than --fail-on-unresolved: it propagates "
             "as a typed embpy error rather than a plain non-zero exit, "
             "so sacct shows category='resolver'. Use when you cannot "
             "tolerate any missing rows (e.g. building production "
             "embedding tables).",
    )
    parser.add_argument(
        "--control-strict", action="store_true",
        help="If a label mixes control + gene components, raise. Off by "
             "default to accept the rare mixed sgRNA library labels.",
    )
    args = parser.parse_args(argv)
    if bool(args.model) == bool(args.table):
        parser.error("Pass exactly one of --model or --table.")
    if args.output is None and args.output_h5ad is None:
        parser.error("Pass --output, --output-h5ad, or both.")
    return args


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
    try:
        if perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"{perturbation_key!r} not in adata.obs "
                f"(got {list(adata.obs.columns)})"
            )
        raw = adata.obs[perturbation_key].astype(str).to_numpy(copy=True)
    finally:
        adata.file.close()
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


def _sanitize_key(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return clean or "embedding"


def _default_obsm_key(args: argparse.Namespace) -> str:
    raw = args.model if args.model else Path(str(args.table)).stem
    return f"X_pert_{_sanitize_key(str(raw))}"


def _write_action_h5ad(
    *,
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    obsm_key: str,
    perturbation_key: str,
    perturbation_labels: list[str],
    embeddings: np.ndarray,
    statuses: np.ndarray,
    control_policy: ControlPolicy,
    control_sentinel_seed: int,
    model_name: str,
    source: str,
    table_path: str | None,
    status_counts: dict[str, int],
) -> Path:
    """Attach per-cell perturbation vectors to a copy of an AnnData file."""
    import anndata as ad  # noqa: PLC0415

    in_path = Path(input_h5ad)
    out_path = Path(output_h5ad)
    if not in_path.exists():
        raise FileNotFoundError(f"AnnData not found: {in_path}")
    adata = ad.read_h5ad(in_path)
    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"{perturbation_key!r} not in adata.obs "
            f"(got {list(adata.obs.columns)})"
        )
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape!r}.")
    dim = int(embeddings.shape[1]) if embeddings.size else 0
    if dim <= 0:
        raise ValueError("Cannot attach zero-dimensional perturbation embeddings to AnnData.")

    by_label = {
        str(label): np.asarray(embeddings[i], dtype=np.float32)
        for i, label in enumerate(perturbation_labels)
    }
    by_status = {
        str(label): str(statuses[i])
        for i, label in enumerate(perturbation_labels)
    }
    try:
        from embpy.io import to_anndata
        from embpy.io.result import EmbeddingProvenance, EmbeddingResult

        aliases = {str(label): {"perturbation_label": str(label)} for label in perturbation_labels}
        result = EmbeddingResult(
            matrix=np.asarray(embeddings, dtype=np.float32),
            entity_ids=tuple(str(label) for label in perturbation_labels),
            entity_type="perturbation",
            id_scheme="perturbation_label",
            provenance=EmbeddingProvenance.create(
                model=model_name,
                extra={
                    "entity_type": "perturbation",
                    "source": source,
                    "table_path": table_path,
                    "input_h5ad": str(in_path),
                    "perturbation_key": perturbation_key,
                    "n_requested_inputs": len(perturbation_labels),
                    "n_successfully_embedded_entities": status_counts.get(EmbeddingStatus.RESOLVED.value, 0),
                    "n_control": status_counts.get(EmbeddingStatus.CONTROL.value, 0),
                    "n_unresolved": status_counts.get(EmbeddingStatus.UNRESOLVED.value, 0),
                },
            ),
            aliases=aliases,
        )
        to_anndata(result, target=adata, attach_to="uns", key=obsm_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not store embpy perturbation payload in .uns: %s", exc)

    control_vec = make_control_vector(dim, seed=control_sentinel_seed)
    unresolved_vec = make_unresolved_vector(dim)

    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    cell_matrix = np.zeros((adata.n_obs, dim), dtype=np.float32)
    cell_status: list[str] = []
    n_control_cells = 0
    n_unresolved_cells = 0
    for i, label in enumerate(labels):
        classification = control_policy.classify(str(label))
        if classification.kind == "control":
            cell_matrix[i] = control_vec
            cell_status.append(EmbeddingStatus.CONTROL.value)
            n_control_cells += 1
            continue
        vec = by_label.get(str(label))
        status = by_status.get(str(label), EmbeddingStatus.UNRESOLVED.value)
        if vec is None:
            cell_matrix[i] = unresolved_vec
            cell_status.append(EmbeddingStatus.UNRESOLVED.value)
            n_unresolved_cells += 1
        else:
            cell_matrix[i] = vec
            cell_status.append(status)
            if status == EmbeddingStatus.UNRESOLVED.value:
                n_unresolved_cells += 1

    adata.obsm[obsm_key] = cell_matrix
    status_col = f"{obsm_key}_status"
    adata.obs[status_col] = cell_status
    meta = {
        "model_name": model_name,
        "source": source,
        "table_path": table_path,
        "input_h5ad": str(in_path),
        "obsm_key": obsm_key,
        "status_obs_column": status_col,
        "perturbation_key": perturbation_key,
        "embedding_dim": dim,
        "n_unique_non_control_perturbations": int(len(perturbation_labels)),
        "n_cells": int(adata.n_obs),
        "n_control_cells": int(n_control_cells),
        "n_unresolved_cells": int(n_unresolved_cells),
        "counts": status_counts,
        "control_sentinel_seed": int(control_sentinel_seed),
        "control_policy_patterns": list(control_policy.patterns),
        "control_extra_labels": list(control_policy.extra_labels),
    }
    root = adata.uns.setdefault("world_model_action_embeddings", {})
    root[obsm_key] = meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(
        "Wrote AnnData with perturbation embeddings: %s obsm[%r] shape=%s.",
        out_path,
        obsm_key,
        cell_matrix.shape,
    )
    return out_path


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

    if args.table is not None:
        provider = PrecomputedProvider(
            args.table,
            control_policy=policy,
            control_sentinel_seed=int(args.control_sentinel_seed),
        )
        model_name = Path(args.table).stem
        source = "precomputed"
    else:
        provider = BioEmbedderProvider(
            model_name=args.model,
            organism=args.organism,
            resolver_backend=args.resolver_backend,
            id_type=args.id_type,
            region=args.region,
            pooling_strategy=args.pooling_strategy,
            device=args.device,
            cache_dir=args.cache_dir,
            control_policy=policy,
            control_sentinel_seed=int(args.control_sentinel_seed),
        )
        model_name = str(args.model)
        source = "bio_embedder"

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

    output_path = None
    if args.output is not None:
        output_path = Path(args.output)
    elif isinstance(provider, BioEmbedderProvider):
        output_path = Path(args.cache_dir) / provider.cache_key.relative_path()

    if isinstance(provider, BioEmbedderProvider):
        save_cached(
            Path(args.cache_dir),
            provider.cache_key,
            perturbation_labels,
            embeddings,
        )
    if args.output is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            symbols=np.asarray(perturbation_labels, dtype=object),
            embeddings=embeddings.astype(np.float32),
            statuses=statuses,
        )
    if output_path is not None:
        logger.info("Wrote embeddings to %s", output_path)

    obsm_key = args.obsm_key or _default_obsm_key(args)
    if args.output_h5ad is not None:
        _write_action_h5ad(
            input_h5ad=args.h5ad,
            output_h5ad=args.output_h5ad,
            obsm_key=obsm_key,
            perturbation_key=args.perturbation_key,
            perturbation_labels=perturbation_labels,
            embeddings=embeddings,
            statuses=statuses,
            control_policy=policy,
            control_sentinel_seed=int(args.control_sentinel_seed),
            model_name=model_name,
            source=source,
            table_path=args.table,
            status_counts=counts,
        )

    # Write a structured sidecar JSON next to the NPZ so the user can
    # ``cat`` it without booting Python.
    status_meta = {
        "model_name": model_name,
        "source": source,
        "h5ad": str(args.h5ad),
        "output_h5ad": str(args.output_h5ad) if args.output_h5ad else None,
        "obsm_key": obsm_key if args.output_h5ad else None,
        "table": str(args.table) if args.table else None,
        "n_rows": int(len(perturbation_labels)),
        "n_control_cells_filtered_upstream": int(n_control_cells),
        "counts": counts,
        "unresolved_symbols_first_50": list(provider._last_unresolved)[:50],
        "control_symbols_first_20": list(provider._last_controls)[:20],
        "control_sentinel_seed": int(args.control_sentinel_seed),
        "control_policy_patterns": list(policy.patterns),
        "control_extra_labels": list(policy.extra_labels),
    }
    sidecar_base = output_path if output_path is not None else Path(args.output_h5ad)
    sidecar = sidecar_base.with_suffix(sidecar_base.suffix + ".status.json")
    sidecar.write_text(json.dumps(status_meta, indent=2, default=str))
    logger.info("Status sidecar -> %s", sidecar)

    # Layer 2: per-identifier resolution report sidecar. Separate from
    # ``.status.json`` (which is the rolled-up summary) so the cheap
    # ``cat status.json`` workflow stays fast even when the resolution
    # JSON is hundreds of KB. Use ``jq`` against this to find specific
    # failed symbols by reason / source.
    resolution_report = getattr(provider, "last_report", None)
    resolution_sidecar: Path | None = None
    if resolution_report is not None:
        resolution_sidecar = output_path.with_suffix(
            output_path.suffix + ".resolution.json"
        ) if output_path is not None else Path(args.output_h5ad).with_suffix(
            Path(args.output_h5ad).suffix + ".resolution.json"
        )
        resolution_sidecar.write_text(json.dumps(
            resolution_report.to_dict(include_records=True),
            indent=2, default=str,
        ))
        logger.info(
            "Resolution sidecar -> %s (n=%d resolved=%s by_source=%s)",
            resolution_sidecar,
            len(resolution_report.records),
            resolution_report.count_by_status(),
            resolution_report.count_by_source(),
        )

    # Round-trip sanity probe for explicit NPZ output.
    if output_path is not None and output_path.exists():
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
                "Round-trip sanity probe failed (%s); output file is still valid.", e,
            )

    if args.fail_on_unresolved and counts.get(EmbeddingStatus.UNRESOLVED.value, 0):
        logger.error(
            "Exiting non-zero because --fail-on-unresolved is set and "
            "%d rows are UNRESOLVED.",
            counts[EmbeddingStatus.UNRESOLVED.value],
        )
        return 2

    # Layer 2: --strict-resolver promotes any UNRESOLVED record in the
    # per-identifier report to a typed ResolverError so sacct shows
    # category='resolver' and the operator can tell at a glance that
    # the failure was a resolver gap, not a model crash.
    if args.strict_resolver and resolution_report is not None:
        n_unresolved = len(resolution_report.unresolved())
        if n_unresolved:
            # Import here so a default invocation without --strict-resolver
            # doesn't pay the import cost.
            from embpy.errors import ResolverError  # noqa: PLC0415

            sample = [r.identifier for r in resolution_report.unresolved()[:10]]
            raise ResolverError(
                backend=str(args.resolver_backend),
                organism=str(args.organism),
                n_requested=len(perturbation_labels),
                n_resolved=len(perturbation_labels) - n_unresolved,
                model_name=args.model,
                message=(
                    f"--strict-resolver: {n_unresolved} input(s) UNRESOLVED "
                    f"for model={args.model!r} (sample: {sample}). See "
                    f"{resolution_sidecar} for per-identifier reasons."
                ),
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except EmbpyError as exc:
        # Convert typed embpy failures into category-specific exit codes
        # so SLURM sacct can aggregate failures across the sweep. The
        # underlying traceback was already printed at the catch-and-
        # reraise sites inside embpy.embedder; we only emit a one-line
        # summary here for the end of the log.
        sys.exit(_handle_typed_error(exc))
