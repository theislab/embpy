"""Shared validation and provenance for fitted fusion operators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from embpy.io.result import EmbeddingProvenance, EmbeddingResult


def validate_weights(names: Sequence[str], weights: Mapping[str, float] | None) -> np.ndarray:
    """Validate fixed nonnegative view weights; require at least one positive weight."""
    if weights is not None and set(weights) != set(names):
        raise ValueError("weights must name exactly the supplied views.")
    values = np.ones(len(names)) if weights is None else np.asarray([weights[k] for k in names], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("weights must be finite, nonnegative, and not all zero.")
    return values


def id_digest(ids: Sequence[str]) -> str:
    """Fingerprint an ordered training cohort without storing all its identifiers."""
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def fused_result(operator, aligned, matrix, coverage) -> EmbeddingResult:
    """Build a canonical result with both input provenance and fitted-cohort metadata."""
    first = next(iter(aligned.values()))
    parameters = json.loads(json.dumps(operator.get_params(deep=False), default=lambda value: value.item()))
    details = {
        "operator": type(operator).__name__,
        "parameters": parameters,
        "view_order": list(operator.view_names_),
        "input_dimensions": operator.input_dimensions_,
        "training_ids_sha256": operator.training_ids_sha256_,
        "training_n_entities": operator.n_fit_entities_,
        "fit_coverage": operator.coverage_,
        "transform_coverage": coverage,
        "sources": {name: view.provenance.to_dict() for name, view in aligned.items()},
        "block_scales": operator.block_scales_,
    }
    if hasattr(operator, "pca_"):
        details["explained_variance_ratio"] = operator.pca_.explained_variance_ratio_.tolist()
    return EmbeddingResult(
        matrix,
        first.entity_ids,
        first.entity_type,
        first.id_scheme,
        EmbeddingProvenance.create("fusion:" + type(operator).__name__, extra={"fusion": details}),
        first.aliases,
    )
