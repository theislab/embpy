"""Diagnostics: dump a few sample sequences to inspect context bucketing.

When ``DataConfig.sequence_bucket_key`` is set, the dataset emits each
sequence anchored to a single biological/technical bucket. This module
provides a one-shot writer that materialises the first few sequences
from the training set and writes them in human-readable form to a text
file, so you can verify that the bucketing is producing the contexts
you expect (same bucket across timesteps, varied perturbations).

Used by :mod:`world_model.scripts.train` immediately after dataloaders
are built.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def dump_sample_contexts(
    dataset: Any,
    output_path: str | Path,
    *,
    n_sequences: int = 5,
    n_steps_to_show: int | None = None,
) -> None:
    """Write the first ``n_sequences`` sample sequences to ``output_path``.

    Each sequence/task is printed in three sections:

    1. Header: sequence index + bucket summary (id, value, n_cells,
       n_labels, n_control_cells).
    2. Either trajectory timesteps or in-context support/query labels
       and action-index summaries.
    3. Footer: blank line.

    Parameters
    ----------
    dataset
        A :class:`PerturbationSequenceDataset` (or subclass). Must
        expose ``__getitem__``, ``available_perturbations`` and
        optionally ``bucket_summary``.
    output_path
        Where to write the report.
    n_sequences
        Number of sequences to materialise.
    n_steps_to_show
        Per-sequence cap on timesteps printed. ``None`` prints all T.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = max(1, int(n_sequences))
    bucket_summary_fn = getattr(dataset, "bucket_summary", None)

    bucketing_on = (
        getattr(dataset, "_cell_buckets", None) is not None
        and getattr(dataset, "_sampleable_buckets", None)
    )

    lines: list[str] = []
    lines.append("# Sample sequences from the training dataset")
    lines.append(f"# bucketing = {'ON' if bucketing_on else 'OFF'}")
    if bucketing_on:
        lines.append(
            f"# sampleable buckets = {len(dataset._sampleable_buckets)}  "  # noqa: SLF001
            f"(first 10 ids: {list(dataset._sampleable_buckets)[:10]})"  # noqa: SLF001
        )
    context_mode = getattr(dataset, "context_mode", "trajectory")
    lines.append(
        f"# context_mode = {context_mode}, T = {dataset.sequence_length}, K = {dataset.stack_size}, "
        f"n_pert = {dataset.n_pert}, control = {dataset.control_label!r}"
    )
    lines.append("")

    for i in range(n):
        sample = dataset[i]
        bucket_id = int(sample.get("bucket_id", -1))
        perts = sample.get("perturbations", [])

        lines.append(f"=== sequence #{i} ===")
        if bucket_summary_fn is not None and bucket_id >= 0:
            info = bucket_summary_fn(bucket_id)
            if info:
                lines.append(
                    f"  bucket_id={info.get('bucket_id')}  "
                    f"bucket_value={info.get('bucket_value')!r}  "
                    f"n_cells={info.get('n_cells')}  "
                    f"n_labels={info.get('n_labels')}  "
                    f"n_control_cells={info.get('n_control_cells')}"
                )
        elif bucket_id < 0:
            lines.append("  bucket_id=-1 (bucketing disabled)")

        if "support_act" in sample and "query_act" in sample:
            support_actions = (
                sample["support_act"].cpu().numpy() if hasattr(sample["support_act"], "cpu") else sample["support_act"]
            )
            query_action = (
                sample["query_act"].cpu().numpy() if hasattr(sample["query_act"], "cpu") else sample["query_act"]
            )
            support_labels = list(perts[:-1])
            query_label = perts[-1] if perts else "<missing>"
            s_cap = len(support_labels) if n_steps_to_show is None else min(n_steps_to_show, len(support_labels))
            lines.append("  support:")
            for s in range(s_cap):
                action_ids = (
                    list(support_actions[s]) if hasattr(support_actions[s], "__iter__") else [support_actions[s]]
                )
                lines.append(
                    f"    m={s:>2}  action_label={support_labels[s]!r:<30}  action_indices={action_ids}"
                )
            if s_cap < len(support_labels):
                lines.append(f"    ... ({len(support_labels) - s_cap} more support triplets elided) ...")
            q_ids = list(query_action) if hasattr(query_action, "__iter__") else [query_action]
            leak = query_label in support_labels
            lines.append(f"  query: action_label={query_label!r:<30}  action_indices={q_ids}  in_support={leak}")
        else:
            actions = sample["action_indices"].cpu().numpy() if hasattr(
                sample["action_indices"], "cpu"
            ) else sample["action_indices"]
            t_cap = len(perts) if n_steps_to_show is None else min(n_steps_to_show, len(perts))
            for t in range(t_cap):
                action_ids = list(actions[t]) if hasattr(actions[t], "__iter__") else [actions[t]]
                lines.append(
                    f"  t={t:>2}  action_label={perts[t]!r:<30}  action_indices={action_ids}"
                )
            if t_cap < len(perts):
                lines.append(f"  ... ({len(perts) - t_cap} more timesteps elided) ...")
        lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info(
        "Sample-context inspector: wrote %d sequences to %s (bucketing=%s).",
        n, output_path, "ON" if bucketing_on else "OFF",
    )


__all__ = ["dump_sample_contexts"]
