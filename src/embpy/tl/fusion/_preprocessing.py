"""Training-fitted view preprocessing shared by feature fusion operators."""

from collections.abc import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize


def fit_preprocessing(operator, aligned, *, balance_blocks=False):
    names = tuple(aligned)
    if isinstance(operator.normalization, Mapping):
        if set(operator.normalization) != set(names):
            raise ValueError("normalization must name exactly the supplied views.")
        norms = dict(operator.normalization)
    else:
        norms = dict.fromkeys(names, operator.normalization)
    if any(value not in {"none", "zscore", "l2"} for value in norms.values()):
        raise ValueError("normalization must be 'none', 'zscore', or 'l2'.")
    scalers, scales = {}, {}
    for name, view in aligned.items():
        values = np.asarray(view.matrix, dtype=np.float64)
        scaler = StandardScaler().fit(values) if norms[name] == "zscore" else None
        values = _normalize(values, norms[name], scaler)
        rms = float(np.sqrt(np.mean(np.sum(values * values, axis=1)))) if balance_blocks else 1.0
        scalers[name], scales[name] = scaler, rms if rms > 0 else 1.0
    operator.normalizations_ = norms
    operator.scalers_, operator.block_scales_ = scalers, scales
    return transform_preprocessing(operator, aligned)


def _normalize(values, method, scaler):
    if method == "zscore":
        return scaler.transform(values)
    if method == "l2":
        return normalize(values, axis=1)
    return values


def transform_preprocessing(operator, aligned):
    return {
        name: _normalize(
            np.asarray(view.matrix, dtype=np.float64), operator.normalizations_[name], operator.scalers_[name]
        )
        / operator.block_scales_[name]
        for name, view in aligned.items()
    }
