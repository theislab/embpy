"""Identifier-aware adapters for multiple views of the same entities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from embpy.io.result import EmbeddingProvenance, EmbeddingResult

Views = Mapping[str, EmbeddingResult]


def subset_view(view: EmbeddingResult, ids: Sequence[str]) -> EmbeddingResult:
    """Select and order entities by canonical identifier, never by row position."""
    lookup = {key: i for i, key in enumerate(view.entity_ids)}
    wanted = tuple(str(key) for key in ids)
    rows = [lookup[key] for key in wanted]
    aliases = None if view.aliases is None else {key: view.aliases[key] for key in wanted if key in view.aliases}
    return EmbeddingResult(
        view.matrix[rows],
        wanted,
        view.entity_type,
        view.id_scheme,
        view.provenance,
        aliases,
    )


def subset_views(views: Views, entity_ids: Sequence[str]) -> dict[str, EmbeddingResult]:
    """Select the same requested entity IDs from each view in the given order."""
    return {name: subset_view(view, entity_ids) for name, view in views.items()}


def align_views(views: Views, *, join: str = "strict") -> tuple[dict[str, EmbeddingResult], dict[str, Any]]:
    """Match canonical IDs, preserving the first view's entity order.

    ``strict`` requires identical ID sets (row order may differ).
    ``inner`` explicitly takes their intersection and reports dropped counts.
    Entity types and identifier schemes must agree; this function does not
    infer gene/protein mappings or join unrelated cell and drug identifiers.
    """
    if join not in {"strict", "inner"}:
        raise ValueError("join must be 'strict' or 'inner'.")
    if not isinstance(views, Mapping) or not views:
        raise ValueError("views must be a nonempty mapping of names to EmbeddingResult objects.")
    for name, view in views.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Each view name must be a nonempty string.")
        if not isinstance(view, EmbeddingResult):
            raise TypeError(f"View {name!r} must be an EmbeddingResult.")
        if not view.n_entities or not view.n_dims or not np.isfinite(view.matrix).all():
            raise ValueError(f"View {name!r} must have nonempty, finite features.")
    first = next(iter(views.values()))
    common = set(first.entity_ids)
    for name, view in views.items():
        if (view.entity_type, view.id_scheme) != (first.entity_type, first.id_scheme):
            raise ValueError(f"View {name!r} has an incompatible entity_type or id_scheme.")
        ids = set(view.entity_ids)
        if join == "strict" and ids != set(first.entity_ids):
            raise ValueError(f"View {name!r} has different entity IDs; use join='inner' explicitly to intersect.")
        common.intersection_update(ids)
    if not common:
        raise ValueError("The views have no common entity IDs.")
    wanted = tuple(key for key in first.entity_ids if key in common)
    aligned = {name: subset_view(view, wanted) for name, view in views.items()}
    coverage = {
        "join": join,
        "n_entities": len(wanted),
        "input_counts": {name: view.n_entities for name, view in views.items()},
        "dropped_counts": {name: view.n_entities - len(wanted) for name, view in views.items()},
    }
    return aligned, coverage


def views_from_anndata(
    adata: Any,
    keys: Sequence[str],
    *,
    entity_type: str = "sample",
    id_scheme: str = "obs_names",
) -> dict[str, EmbeddingResult]:
    """Read named ``obsm`` matrices using unique observation IDs as row keys.

    Repeated compounds may have distinct observation IDs. Pass their compound
    labels separately as benchmark groups to prevent train/test overlap.
    """
    from scipy.sparse import issparse

    if isinstance(keys, str) or not keys or len(set(keys)) != len(keys):
        raise ValueError("keys must be a nonempty sequence of distinct obsm keys.")
    ids = tuple(str(x) for x in adata.obs_names)
    result = {}
    for key in keys:
        matrix = adata.obsm[key]
        matrix = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
        result[key] = EmbeddingResult(
            matrix,
            ids,
            entity_type,
            id_scheme,
            EmbeddingProvenance.create(key, extra={"source": "AnnData.obsm", "obsm_key": key}),
        )
    return result


def view_from_store(store: Any, identifiers: Sequence[str]) -> EmbeddingResult:
    """Read canonical IDs from a StaticEmbeddingStore, failing on missing IDs."""
    ids = tuple(str(x) for x in identifiers)
    return EmbeddingResult(
        store.get(ids, missing="raise", id_type=store.id_type),
        ids,
        store.entity_type,
        store.id_type,
        EmbeddingProvenance.create(store.key, extra={"source": str(store.path)}),
    )
