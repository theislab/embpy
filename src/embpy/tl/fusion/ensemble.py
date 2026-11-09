"""Late fusion of labelled query-by-candidate scores and ranks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._base import validate_weights


def _combine(scores, weights, higher_is_better, ranked):
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a nonempty mapping of view names to DataFrames.")
    first = next(iter(scores.values()))
    if not isinstance(first, pd.DataFrame) or first.empty:
        raise ValueError("Scores must be nonempty query-by-candidate DataFrames.")
    result = np.zeros(first.shape, dtype=np.float64)
    values = validate_weights(tuple(scores), weights)
    values = values / values.max()
    values /= values.sum()
    for (name, frame), weight in zip(scores.items(), values, strict=True):
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"Scores for {name!r} must have unique query and candidate IDs.")
        if set(frame.index) != set(first.index) or set(frame.columns) != set(first.columns):
            raise ValueError("All score matrices must have identical query and candidate ID sets.")
        matrix = frame.reindex(index=first.index, columns=first.columns).to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Scores must be finite; missing candidates cannot be silently imputed.")
        utility = matrix if higher_is_better else -matrix
        if ranked:
            utility = -rankdata(-utility, axis=1, method="average")
        result += weight * utility
    return pd.DataFrame(result, index=first.index.copy(), columns=first.columns.copy())


def fuse_scores(scores, *, weights=None, higher_is_better=True) -> pd.DataFrame:
    """Average scores after matching query/candidate IDs; larger output is better.

    Weights are fixed, nonnegative, and normalized to sum to one. Scores should
    have comparable scales. For distances, set higher_is_better=False; the
    returned utility is then the negative weighted mean distance.
    """
    return _combine(scores, weights, higher_is_better, False)


def fuse_ranks(scores, *, weights=None, higher_is_better=True) -> pd.DataFrame:
    """Return negative mean ranks, with average ranks for ties (larger is better).

    Ranking is performed independently per query and view before averaging.
    Each view must rank the same candidate universe; truncated lists cannot be
    mixed by silently assigning missing candidates a score or rank.
    """
    return _combine(scores, weights, higher_is_better, True)
