"""Dimensionality harmonization for cross-model comparison.

A user comparing embeddings from models of different widths can project
each to a common ``n_components`` with :func:`harmonize`. This reuses the
package's single PCA implementation (:func:`embpy._pca.pca_project`,
which :func:`embpy.tl.dimred.compute_pca` also delegates to) -- there is
exactly one PCA in the codebase, and this is not a second one. Using the
light leaf directly keeps harmonization free of the heavy analysis /
model stack that importing ``embpy.tl`` would drag in.

Important: PCA is fit **per embedding** (each model gets its own basis).
Harmonized results are therefore comparable in *dimensionality and
intrinsic geometry* (pairwise distances within a model are preserved up
to the truncation) but **not in coordinate frame** -- component ``k`` of
model A is unrelated to component ``k`` of model B. Callers must not
assume a shared basis across harmonized results.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from .result import EmbeddingResult

logger = logging.getLogger(__name__)


def harmonize(
    result: EmbeddingResult,
    n_components: int,
    *,
    random_state: int = 0,
) -> EmbeddingResult:
    """PCA-project ``result`` to ``n_components`` and record the variance.

    Parameters
    ----------
    result
        The embedding to project.
    n_components
        Target width. Clamped to ``min(n_entities, n_dims, n_components)``
        by the underlying PCA.
    random_state
        Threaded into the PCA SVD solver for determinism.

    Returns
    -------
    EmbeddingResult
        A new frozen result with the projected matrix and provenance
        updated with ``harmonized_n_components`` + ``explained_variance_ratio``.
        Entity ids / type / id_scheme / aliases are carried over unchanged.
    """
    from embpy._pca import pca_project

    coords_f64, variance = pca_project(
        result.matrix,
        n_components,
        random_state=random_state,
    )
    coords = np.ascontiguousarray(coords_f64, dtype=np.float32)
    variance_ratio = tuple(float(x) for x in variance)
    extra = dict(result.provenance.extra)
    extra["harmonization"] = {
        "method": "pca",
        "fit_scope": "per_embedding_result",
        "requested_n_components": int(n_components),
        "actual_n_components": int(coords.shape[1]),
        "random_state": int(random_state),
        "coordinate_basis": (
            "PCA was fit separately for this embedding result; components "
            "are not a shared cross-model coordinate basis."
        ),
    }

    new_prov = dataclasses.replace(
        result.provenance,
        harmonized_n_components=int(coords.shape[1]),
        explained_variance_ratio=variance_ratio,
        random_state=int(random_state),
        extra=extra,
    )
    logger.info(
        "Harmonized %s embedding %d -> %d dims (cumulative variance %.3f).",
        result.entity_type,
        result.n_dims,
        coords.shape[1],
        sum(variance_ratio),
    )
    return EmbeddingResult(
        matrix=coords,
        entity_ids=result.entity_ids,
        entity_type=result.entity_type,
        id_scheme=result.id_scheme,
        provenance=new_prov,
        aliases=result.aliases,
    )


__all__ = ["harmonize"]
