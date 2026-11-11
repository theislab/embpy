"""Shared validation and provenance for fitted fusion operators."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from embpy.io.result import EmbeddingProvenance, EmbeddingResult

from ._views import align_views


class BaseFusion(TransformerMixin, BaseEstimator, ABC):
    """Cloneable, identifier-aware interface for fitted feature fusion.

    Implementations fit on named EmbeddingResult views and transform new
    entities using the same named views, widths and identifier schema.
    """

    def _fit_views(self, X, *, min_views=1, exact_views=None):
        for key in list(vars(self)):
            if key.endswith("_"):
                delattr(self, key)
        aligned, coverage = align_views(X, join=self.join)
        if len(aligned) < min_views or (exact_views is not None and len(aligned) != exact_views):
            requirement = f"exactly {exact_views}" if exact_views is not None else f"at least {min_views}"
            raise ValueError(f"{type(self).__name__} requires {requirement} views.")
        first = next(iter(aligned.values()))
        self.view_names_ = tuple(aligned)
        self.input_dimensions_ = {name: view.n_dims for name, view in aligned.items()}
        self.entity_schema_ = (first.entity_type, first.id_scheme)
        self.training_ids_sha256_ = id_digest(first.entity_ids)
        self.n_fit_entities_, self.coverage_ = first.n_entities, coverage
        return aligned

    def _align_transform(self, X):
        check_is_fitted(self, "is_fitted_")
        if set(X) != set(self.view_names_):
            raise ValueError("transform requires exactly the view names used during fit.")
        aligned, coverage = align_views({name: X[name] for name in self.view_names_}, join=self.join)
        for name, view in aligned.items():
            if (view.entity_type, view.id_scheme) != self.entity_schema_:
                raise ValueError("Entity schema differs from the fitted fusion schema.")
            if view.n_dims != self.input_dimensions_[name]:
                raise ValueError(f"View {name!r} has a different dimension than during fit.")
        return aligned, coverage

    def _metadata(self):
        return {}

    @abstractmethod
    def fit(self, X, y=None):
        """Fit using training views only."""

    @abstractmethod
    def transform(self, X):
        """Return an EmbeddingResult for new entities."""


@dataclass(frozen=True)
class FusionConfig:
    """A benchmark operator, optional view subset and inner-fold parameter grid.

    ``views=None`` uses all supplied views. ``param_grid`` names constructor
    parameters of ``operator`` (for example ``regularization``). It requires
    benchmark ``tune=True``; selection never uses outer test entities.
    Passing an operator directly remains equivalent to FusionConfig(operator).
    """

    operator: BaseFusion
    views: Sequence[str] | None = None
    param_grid: Mapping[str, Sequence] | None = None


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
        "projection_coefficients": getattr(operator, "n_projection_coefficients_", 0),
        **operator._metadata(),
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
