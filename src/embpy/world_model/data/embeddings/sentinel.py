"""Status-aware embedding contract.

Three kinds of "rows" appear in the action-embedding table:

* RESOLVED   -- a real gene whose embedding was successfully computed.
* CONTROL    -- a non-targeting / safe-harbor guide. Conceptually "no
                perturbation", but the dynamics module needs a distinct
                representation to learn that distinction (a zero row is
                ambiguous: see UNRESOLVED).
* UNRESOLVED -- a real gene symbol the embedder could not resolve. Zero
                rows are emitted only here, and ONLY paired with a
                structured warning. Downstream code should treat
                UNRESOLVED as a data-quality bug, not as silent fact.

The sentinel vector for CONTROL is a deterministic, non-zero, seeded
sample. Two reasons:

1. The world model's dynamics needs a stable, recognisable "no
   perturbation" token. A zero vector is indistinguishable from an
   UNRESOLVED row, breaking debugging and any downstream loss
   accounting.
2. A deterministic seed (numpy ``default_rng(seed)``) makes the vector
   reproducible across runs and machines, so cache files, regression
   tests, and audit metadata all agree. The vector is the same for
   every CONTROL row in a single run -- no per-symbol jitter -- so the
   dynamics module sees one "control token" the way every Atari /
   D4RL world model sees one "no-op action".

This module is intentionally dependency-light: it only imports
``numpy`` and the standard library, so it can be loaded from environments
without pyensembl, boltz, arc-state, etc.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

__all__ = [
    "CONTROL_SENTINEL_SEED",
    "EmbeddingStatus",
    "describe_status_counts",
    "make_control_vector",
    "make_unresolved_vector",
    "status_array",
]

logger = logging.getLogger(__name__)


CONTROL_SENTINEL_SEED: int = 0
"""Seed for the deterministic control sentinel vector.

Mixed with the string ``"control"`` via ``SeedSequence`` so the resulting
PRNG state is stable across numpy versions and across runs. Override at
the dataset level (``ActionEmbeddingConfig.control_sentinel_seed``) if a
user really needs an orthogonal control token per dataset; the default
is intentionally fixed so cache files compare cleanly across machines.
"""


class EmbeddingStatus(str, Enum):
    """Per-row status of an action embedding.

    Inherits from ``str`` so the enum is JSON-serialisable directly and
    so downstream array casts (``np.asarray([EmbeddingStatus.CONTROL, ...])``
    on numpy >= 1.21) produce a stable string dtype.
    """

    RESOLVED = "RESOLVED"
    CONTROL = "CONTROL"
    UNRESOLVED = "UNRESOLVED"


def _seeded_rng(seed: int, *salt: str) -> np.random.Generator:
    # SeedSequence with mixed-in string salt: the resulting state is
    # uniquely keyed by (seed, salt) and is portable across numpy
    # versions. We avoid raw `default_rng(seed)` so that future calls
    # for unrelated salts cannot collide.
    salt_ints = tuple(
        int.from_bytes(s.encode("utf-8"), "big", signed=False) % (2**31 - 1)
        for s in salt
    )
    sequence = np.random.SeedSequence([seed, *salt_ints])
    return np.random.default_rng(sequence)


def make_control_vector(
    dim: int,
    *,
    seed: int = CONTROL_SENTINEL_SEED,
) -> np.ndarray:
    """Return a deterministic, non-zero, seeded vector of shape ``(dim,)``.

    The vector is sampled from a unit-variance Gaussian and rescaled so
    its L2 norm is ``sqrt(dim)``. Rescaling matters because foundation
    embeddings vary wildly in norm across modalities (ESM-2 logits are
    O(10), Borzoi pooled activations are O(0.1)), and the dynamics
    module typically applies LayerNorm so an absolute non-zero norm is
    enough to keep the control token from collapsing into the padding /
    UNRESOLVED row.

    Parameters
    ----------
    dim
        Target dimensionality. Must be a positive integer.
    seed
        Seed integer mixed with the literal salt ``"control"``. Default
        is :data:`CONTROL_SENTINEL_SEED` (zero).

    Returns
    -------
    np.ndarray
        A ``(dim,)`` float32 array with unit-per-coordinate variance and
        ``||v|| == sqrt(dim)`` (up to floating point).

    Notes
    -----
    Two calls with the same ``(dim, seed)`` return *identical* arrays
    bit-for-bit on the same machine, and equivalent arrays across
    machines that share an x86_64 / aarch64 / numpy >= 1.17 baseline.
    """
    if dim <= 0:
        raise ValueError(f"make_control_vector: dim must be > 0, got {dim}")
    rng = _seeded_rng(int(seed), "control")
    raw = rng.standard_normal(int(dim)).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        # Fundamentally cannot happen for dim > 0 from standard_normal,
        # but defensively swap to ones so we never emit a true zero row.
        return np.ones((int(dim),), dtype=np.float32) / float(np.sqrt(dim))
    target = float(np.sqrt(dim))
    return (raw * (target / norm)).astype(np.float32)


def make_unresolved_vector(dim: int) -> np.ndarray:
    """Return a zero row of shape ``(dim,)``.

    Always emitted alongside a structured WARNING by the caller; the
    zero is a signal, not a default. Splitting this out as a one-line
    helper makes the contract grep-able and lets future revisions swap
    in a different sentinel (e.g. NaN) without touching providers.
    """
    if dim <= 0:
        raise ValueError(f"make_unresolved_vector: dim must be > 0, got {dim}")
    return np.zeros((int(dim),), dtype=np.float32)


def status_array(statuses: list[EmbeddingStatus]) -> np.ndarray:
    """Return a numpy array of string status codes for JSON / metadata use."""
    return np.asarray([s.value for s in statuses], dtype=object)


def describe_status_counts(statuses: list[EmbeddingStatus]) -> dict[str, int]:
    """Return per-bucket counts for action_embedding_meta.json."""
    counts: dict[str, int] = {s.value: 0 for s in EmbeddingStatus}
    for s in statuses:
        counts[s.value] = counts.get(s.value, 0) + 1
    return counts
