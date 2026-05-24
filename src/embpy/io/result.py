"""The canonical in-memory embedding result and its provenance record.

This module is the single source of truth for *what an embedding is*
once embpy has produced it: a matrix of ``(n_entities, n_dims)`` with
canonical row identifiers and a fixed ``(entity_type, id_scheme)``.

Design rule (do not break): :class:`EmbeddingResult` is **format
agnostic**. It knows nothing about AnnData, pandas, parquet or csv.
Anything that reads or writes a file lives in :mod:`embpy.io.exporters`
or :mod:`embpy.io.legacy`, never here. If this class grows a method that
mentions a file format, the layering is wrong -- stop and move it out.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingProvenance:
    """Everything needed to reproduce / audit one embedding artifact.

    Kept deliberately flat and JSON-serialisable so it can be written to
    a parquet sidecar or an AnnData ``uns`` block without custom codecs.
    Use :meth:`create` to stamp version / git-sha / timestamp
    automatically; construct directly only in tests where determinism
    matters.
    """

    model: str
    pooling: str | None = None
    layer: str | int | None = None
    embpy_version: str = ""
    git_sha: str | None = None
    timestamp: str = ""
    # Set by :func:`embpy.io.harmonize.harmonize`; ``None`` until then.
    harmonized_n_components: int | None = None
    explained_variance_ratio: tuple[float, ...] | None = None
    random_state: int | None = None
    # Free-form extras (e.g. resolver source per alias scheme). Plain
    # dict for JSON-friendliness; the frozen dataclass stops the *field*
    # being rebound, which is the property we actually rely on.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, model: str, **kwargs: Any) -> EmbeddingProvenance:
        """Build a provenance record, auto-filling version / git / time.

        ``embpy_version`` comes from the installed package metadata,
        ``git_sha`` from ``git rev-parse`` (best-effort; ``None`` outside
        a checkout), and ``timestamp`` is the current UTC time. Any of
        these can be overridden via ``kwargs``.
        """
        defaults: dict[str, Any] = {
            "embpy_version": _embpy_version(),
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        defaults.update(kwargs)
        return cls(model=model, **defaults)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON sidecars / ``uns`` blocks."""
        return {
            "model": self.model,
            "pooling": self.pooling,
            "layer": self.layer,
            "embpy_version": self.embpy_version,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "harmonized_n_components": self.harmonized_n_components,
            "explained_variance_ratio": (
                list(self.explained_variance_ratio)
                if self.explained_variance_ratio is not None
                else None
            ),
            "random_state": self.random_state,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """One canonical embedding: rows are entities, columns are dimensions.

    Parameters
    ----------
    matrix
        ``(n_entities, n_dims)`` float32 array. Coerced to ``float32`` and
        C-contiguous on construction.
    entity_ids
        Canonical identifiers, one per row, in ``id_scheme``. Stored as a
        tuple so the result stays hashable / immutable.
    entity_type
        What the rows are: ``"gene"``, ``"molecule"``, ``"protein"``,
        ``"sequence"``, ``"cytokine"``, ``"cell_line"``, ...
    id_scheme
        The identifier convention, e.g. ``"ensembl_gene_id"``,
        ``"canonical_smiles"``, ``"uniprot"``, ``"sequence"``.
    provenance
        :class:`EmbeddingProvenance`.
    aliases
        Optional cross-reference labels for *display only*, never the key:
        ``{entity_id: {alias_scheme: value}}`` (e.g. an Ensembl id mapped
        to its ``gene_symbol`` and ``uniprot``). Exporters surface these
        as extra columns / obs fields so users get human-readable access
        while the index stays canonical.

    Notes
    -----
    ``__post_init__`` rejects (with a value-naming error): non-2D
    matrices, a length mismatch between ``entity_ids`` and rows, duplicate
    ids, and any NaN/Inf entry. Failing loudly here means downstream
    exporters never have to re-validate.
    """

    matrix: np.ndarray
    entity_ids: tuple[str, ...]
    entity_type: str
    id_scheme: str
    provenance: EmbeddingProvenance
    aliases: Mapping[str, Mapping[str, str]] | None = None

    def __post_init__(self) -> None:
        mat = np.ascontiguousarray(np.asarray(self.matrix), dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(
                f"matrix must be 2D (n_entities, n_dims), got shape {mat.shape!r} "
                f"with ndim={mat.ndim}."
            )
        object.__setattr__(self, "matrix", mat)

        ids = tuple(str(x) for x in self.entity_ids)
        object.__setattr__(self, "entity_ids", ids)

        if len(ids) != mat.shape[0]:
            raise ValueError(
                f"entity_ids has length {len(ids)} but matrix has "
                f"{mat.shape[0]} rows; they must match."
            )

        dup = _first_duplicate(ids)
        if dup is not None:
            raise ValueError(
                f"entity_ids must be unique; first duplicate is {dup!r} "
                f"(entity_type={self.entity_type!r}, id_scheme={self.id_scheme!r})."
            )

        if not np.isfinite(mat).all():
            bad_rows = np.where(~np.isfinite(mat).all(axis=1))[0]
            i = int(bad_rows[0])
            raise ValueError(
                f"matrix contains NaN/Inf; first offending row is index {i} "
                f"(entity_id={ids[i]!r})."
            )

        if self.aliases is not None:
            id_set = set(ids)
            for key in self.aliases:
                if key not in id_set:
                    raise ValueError(
                        f"aliases key {key!r} is not one of entity_ids "
                        f"(id_scheme={self.id_scheme!r})."
                    )

    @property
    def n_entities(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_dims(self) -> int:
        return self.matrix.shape[1]

    @property
    def dim_names(self) -> list[str]:
        """Column labels ``dim_0 ... dim_{n_dims-1}`` used by exporters."""
        return [f"dim_{i}" for i in range(self.n_dims)]


def _first_duplicate(ids: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for x in ids:
        if x in seen:
            return x
        seen.add(x)
    return None


def _embpy_version() -> str:
    try:
        return version("embpy")
    except PackageNotFoundError:
        return ""


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


__all__ = ["EmbeddingProvenance", "EmbeddingResult"]
