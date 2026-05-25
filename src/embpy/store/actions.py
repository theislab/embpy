"""Headless perturbation/action-table compilation from target embeddings.

This module turns a set of perturbation *condition labels* into a table of
pooled action vectors using a gene (or other target-entity) embedding matrix,
*without* requiring an AnnData object. :meth:`embpy.store.EmbpyAccessor.compile_actions`
is a thin AnnData-aware wrapper over :func:`compile_action_table` defined here,
so the same pooling / control / unresolved semantics are shared by both the
AnnData workflow and any headless caller (e.g. a world-model dataloader that
already holds the condition labels and a linked store).

Status contract
---------------
Each compiled condition is tagged with an :class:`ActionStatus`:

* ``RESOLVED``   -- at least one target component resolved; the row is the
                   mean / sum of the resolved component vectors. Partially
                   unresolved combos stay ``RESOLVED`` but record their
                   missing components in :attr:`ActionTable.missing_targets`.
* ``CONTROL``    -- the whole label is a control / non-targeting guide; the
                   row is a deterministic, *non-zero* sentinel vector. A zero
                   row is deliberately avoided because it is ambiguous with
                   ``UNRESOLVED``.
* ``UNRESOLVED`` -- a non-control label whose components could not be resolved
                   at all. Only produced when ``on_unresolved != "raise"``,
                   and always paired with a structured WARNING. The row is a
                   zero vector -- a signal, never a silent default.

The deterministic control sentinel (:func:`control_sentinel_vector`) is
intentionally bit-compatible with
``world_model.data.embeddings.sentinel.make_control_vector`` so a store-backed
action table and the world model's provider stack agree on the "no
perturbation" token.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from embpy.resources.gene.control import ControlPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "ActionStatus",
    "ActionTable",
    "compile_action_table",
    "control_sentinel_vector",
    "split_targets",
]

# Default combo splitter. Kept byte-identical to the historical
# ``accessor._split_targets`` regex so existing ``compile_actions`` callers
# that do not pass a ``control_policy`` see unchanged splitting behaviour.
_COMBO_SPLIT_RE = re.compile(r"[+,;/|]")


class ActionStatus(StrEnum):
    """Per-condition status of a compiled action row.

    A :class:`~enum.StrEnum` so values are JSON-serialisable directly and
    compare equal to their plain-string form (matching the string values of
    ``world_model.data.embeddings.sentinel.EmbeddingStatus``).
    """

    RESOLVED = "RESOLVED"
    CONTROL = "CONTROL"
    UNRESOLVED = "UNRESOLVED"


def split_targets(value: str) -> list[str]:
    """Split a combo label like ``"g1+g2"`` into component target ids.

    Splits on ``+ , ; / |`` and strips whitespace. A label with no
    separators returns ``[value]`` so single-target conditions resolve
    against the target lookup unchanged.
    """
    parts = [p.strip() for p in _COMBO_SPLIT_RE.split(value) if p.strip()]
    return parts or [value]


def control_sentinel_vector(dim: int, *, seed: int = 0) -> np.ndarray:
    """Return a deterministic, non-zero, seeded control vector of shape ``(dim,)``.

    The vector is drawn from a unit-variance Gaussian and rescaled to an L2
    norm of ``sqrt(dim)``. Determinism comes from a ``SeedSequence`` mixed
    with the literal salt ``"control"`` -- the same construction used by the
    world-model provider stack -- so the two layers produce identical control
    tokens for a given ``(dim, seed)``.
    """
    if dim <= 0:
        raise ValueError(f"control_sentinel_vector: dim must be > 0, got {dim}.")
    salt = int.from_bytes(b"control", "big", signed=False) % (2**31 - 1)
    sequence = np.random.SeedSequence([int(seed), salt])
    rng = np.random.default_rng(sequence)
    raw = rng.standard_normal(int(dim)).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        # Unreachable for dim > 0 from standard_normal; defensive so we never
        # emit a true zero row (which would be ambiguous with UNRESOLVED).
        return np.ones((int(dim),), dtype=np.float32) / float(np.sqrt(dim))
    return (raw * (float(np.sqrt(dim)) / norm)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ActionTable:
    """Result of :func:`compile_action_table`.

    Attributes
    ----------
    table
        ``(n_conditions, n_dims)`` DataFrame indexed by condition label with
        ``dim_0 ... dim_{n_dims-1}`` columns.
    statuses
        ``{condition: ActionStatus.value}`` for every row in ``table``.
    missing_targets
        ``{condition: [unresolved component ids]}`` for conditions with at
        least one component the target lookup could not resolve (both fully
        ``UNRESOLVED`` rows and partially-missing ``RESOLVED`` combos).
    aggregation
        ``"mean"`` or ``"sum"`` -- how component vectors were pooled.
    control_sentinel_seed
        Seed used for the CONTROL sentinel vector.
    n_dims
        Embedding dimensionality of every row.
    """

    table: pd.DataFrame
    statuses: dict[str, str]
    missing_targets: dict[str, list[str]]
    aggregation: str
    control_sentinel_seed: int
    n_dims: int

    def _count(self, status: ActionStatus) -> int:
        return sum(1 for v in self.statuses.values() if v == status.value)

    @property
    def n_resolved(self) -> int:
        """Number of conditions with a resolved (pooled) action vector."""
        return self._count(ActionStatus.RESOLVED)

    @property
    def n_control(self) -> int:
        """Number of conditions mapped to the CONTROL sentinel."""
        return self._count(ActionStatus.CONTROL)

    @property
    def n_unresolved(self) -> int:
        """Number of conditions emitted as UNRESOLVED zero rows."""
        return self._count(ActionStatus.UNRESOLVED)

    @property
    def control_conditions(self) -> list[str]:
        """Condition labels mapped to the CONTROL sentinel."""
        return [c for c, s in self.statuses.items() if s == ActionStatus.CONTROL.value]

    @property
    def unresolved_conditions(self) -> list[str]:
        """Condition labels emitted as UNRESOLVED zero rows."""
        return [c for c, s in self.statuses.items() if s == ActionStatus.UNRESOLVED.value]

    def counts(self) -> dict[str, int]:
        """Per-status counts, suitable for a JSON sidecar."""
        return {
            ActionStatus.RESOLVED.value: self.n_resolved,
            ActionStatus.CONTROL.value: self.n_control,
            ActionStatus.UNRESOLVED.value: self.n_unresolved,
        }


def compile_action_table(
    conditions: Sequence[str],
    target_matrix: np.ndarray,
    target_ids: Sequence[str],
    *,
    relation: Mapping[str, Sequence[str]] | None = None,
    aggregation: Literal["mean", "sum"] = "mean",
    control_policy: ControlPolicy | None = None,
    control_values: Sequence[str] | None = None,
    on_unresolved: Literal["raise", "zero", "control"] = "raise",
    control_sentinel_seed: int = 0,
    stage: str = "compile_action_table",
) -> ActionTable:
    """Pool target embeddings into one action vector per perturbation condition.

    Parameters
    ----------
    conditions
        Perturbation condition labels to compile (e.g. ``"g1"``, ``"g1+g2"``,
        ``"non-targeting"``). Duplicates are collapsed, first occurrence wins
        for row order.
    target_matrix
        ``(n_targets, n_dims)`` embedding matrix for the target entities
        (typically genes).
    target_ids
        Canonical id for each row of ``target_matrix`` (e.g. gene symbol).
    relation
        Optional ``{condition: [target_id, ...]}`` mapping. When a condition
        is present here, its targets come from the mapping instead of from
        splitting the label. Conditions absent from the mapping fall back to
        :func:`split_targets` (or ``control_policy.split_combo`` when a policy
        is supplied).
    aggregation
        ``"mean"`` (default) or ``"sum"`` pooling over resolved components.
    control_policy
        Optional :class:`~embpy.resources.gene.control.ControlPolicy`. When
        given, each condition is classified; pure-control labels get a CONTROL
        sentinel and combo splitting / control-component dropping use the
        policy. Passed in (never imported here) to keep ``import embpy`` cheap.
    control_values
        Extra exact labels to treat as control regardless of ``control_policy``.
    on_unresolved
        Behaviour when a non-control condition resolves to *no* target vector:

        * ``"raise"`` (default) -- raise :class:`ValueError`, preserving the
          historical ``compile_actions`` contract.
        * ``"zero"`` -- emit a zero row tagged ``UNRESOLVED`` and log a
          WARNING (the world-model "loud, never silent" contract).
        * ``"control"`` -- route to the CONTROL sentinel and log a WARNING.
    control_sentinel_seed
        Seed for :func:`control_sentinel_vector`.
    stage
        Label used in error / log messages so callers (e.g.
        ``adata.embpy.compile_actions``) get attributable diagnostics.
    """
    matrix = np.asarray(target_matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{stage}: target_matrix must be 2D, got shape {matrix.shape!r}.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{stage}: target_matrix must be numeric, got dtype {matrix.dtype}.")
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32, copy=False)
    ids = [str(t) for t in target_ids]
    if len(ids) != matrix.shape[0]:
        raise ValueError(f"{stage}: target_ids has length {len(ids)} but target_matrix has {matrix.shape[0]} rows.")
    if aggregation not in ("mean", "sum"):
        raise ValueError(f"{stage}: aggregation must be 'mean' or 'sum', got {aggregation!r}.")
    if on_unresolved not in ("raise", "zero", "control"):
        raise ValueError(f"{stage}: on_unresolved must be 'raise', 'zero', or 'control', got {on_unresolved!r}.")

    n_dims = int(matrix.shape[1])
    target_lookup = {eid: i for i, eid in enumerate(ids)}
    relation_lookup = {str(k): [str(t) for t in v] for k, v in (relation or {}).items()}
    control_exact = {str(x) for x in (control_values or [])}
    sentinel = control_sentinel_vector(n_dims, seed=control_sentinel_seed)

    # Collapse duplicates, first occurrence wins for deterministic row order.
    ordered_conditions: list[str] = list(dict.fromkeys(str(c) for c in conditions))

    vectors: list[np.ndarray] = []
    statuses: dict[str, str] = {}
    missing_targets: dict[str, list[str]] = {}

    for condition in ordered_conditions:
        if _is_control(condition, control_policy, control_exact):
            vectors.append(sentinel)
            statuses[condition] = ActionStatus.CONTROL.value
            continue

        targets = _resolve_targets(condition, relation_lookup, control_policy)
        found = [matrix[target_lookup[t]] for t in targets if t in target_lookup]
        miss = [t for t in targets if t not in target_lookup]
        if miss:
            missing_targets[condition] = miss

        if found:
            stacked = np.vstack(found)
            pooled = stacked.mean(axis=0) if aggregation == "mean" else stacked.sum(axis=0)
            vectors.append(pooled.astype(np.float32))
            statuses[condition] = ActionStatus.RESOLVED.value
            continue

        # Nothing resolved for a non-control condition.
        if on_unresolved == "raise":
            raise ValueError(
                f"{stage}: no target embeddings found for perturbation {condition!r} (missing targets: {miss[:5]})."
            )
        if on_unresolved == "control":
            vectors.append(sentinel)
            statuses[condition] = ActionStatus.CONTROL.value
            logger.warning(
                "%s: perturbation %r has no resolvable target (missing: %s); routing to CONTROL sentinel.",
                stage,
                condition,
                miss[:10],
            )
        else:  # "zero"
            vectors.append(np.zeros((n_dims,), dtype=np.float32))
            statuses[condition] = ActionStatus.UNRESOLVED.value
            logger.warning(
                "%s: perturbation %r is UNRESOLVED (missing: %s); emitting a zero row. "
                "Treat this as a data-quality bug, not a default.",
                stage,
                condition,
                miss[:10],
            )

    if vectors:
        data = np.vstack(vectors).astype(np.float32)
    else:
        data = np.zeros((0, n_dims), dtype=np.float32)
    table = pd.DataFrame(data, index=pd.Index(ordered_conditions, name="condition"))
    table.columns = [f"dim_{i}" for i in range(n_dims)]

    result = ActionTable(
        table=table,
        statuses=statuses,
        missing_targets=missing_targets,
        aggregation=aggregation,
        control_sentinel_seed=int(control_sentinel_seed),
        n_dims=n_dims,
    )
    logger.info(
        "%s: compiled %d conditions (resolved=%d control=%d unresolved=%d, dim=%d, agg=%s).",
        stage,
        len(ordered_conditions),
        result.n_resolved,
        result.n_control,
        result.n_unresolved,
        n_dims,
        aggregation,
    )
    if result.unresolved_conditions:
        logger.warning(
            "%s: %d/%d conditions UNRESOLVED (first 10: %s).",
            stage,
            result.n_unresolved,
            len(ordered_conditions),
            result.unresolved_conditions[:10],
        )
    return result


def _is_control(
    condition: str,
    control_policy: ControlPolicy | None,
    control_exact: set[str],
) -> bool:
    if condition in control_exact:
        return True
    if control_policy is not None and control_policy.classify(condition).kind == "control":
        return True
    return False


def _resolve_targets(
    condition: str,
    relation_lookup: Mapping[str, Sequence[str]],
    control_policy: ControlPolicy | None,
) -> list[str]:
    if condition in relation_lookup:
        return [str(t) for t in relation_lookup[condition]]
    if control_policy is not None:
        # Drop control components from mixed labels; keep gene components only.
        return [str(g) for g in control_policy.classify(condition).gene_components]
    return split_targets(condition)
