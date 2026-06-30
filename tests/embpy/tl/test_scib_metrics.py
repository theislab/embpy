"""Tests for embpy.tl.compute_scib_metrics.

The heavy path (the actual scIB battery) needs the optional ``scib`` + scanpy
stack and is exercised in the ``full`` CI job. Here we cover the lightweight
contract: input validation fails fast (before importing scib) and a missing
scib install raises a clear DependencyError.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from anndata import AnnData

from embpy.errors import DependencyError
from embpy.tl import compute_scib_metrics

_HAS_SCIB = importlib.util.find_spec("scib") is not None


def _toy_adata() -> AnnData:
    ad = AnnData(X=np.zeros((6, 4), dtype=np.float32))
    ad.obsm["X_emb"] = np.random.RandomState(0).rand(6, 3).astype(np.float32)
    ad.obs["cell_type"] = ["a", "a", "b", "b", "c", "c"]
    ad.obs["batch"] = ["x", "y", "x", "y", "x", "y"]
    return ad


def test_unknown_label_key_raises_keyerror():
    ad = _toy_adata()
    with pytest.raises(KeyError, match="label_key"):
        compute_scib_metrics(ad, "X_emb", label_key="nope")


def test_unknown_embedding_key_raises_keyerror():
    ad = _toy_adata()
    with pytest.raises(KeyError, match="not in adata.obsm"):
        compute_scib_metrics(ad, "X_missing", label_key="cell_type")


def test_unknown_batch_key_raises_keyerror():
    ad = _toy_adata()
    with pytest.raises(KeyError, match="batch_key"):
        compute_scib_metrics(ad, "X_emb", label_key="cell_type", batch_key="nope")


@pytest.mark.skipif(_HAS_SCIB, reason="scib is installed; this checks the missing-dep path")
def test_missing_scib_raises_dependency_error():
    ad = _toy_adata()
    with pytest.raises(DependencyError, match="scib"):
        compute_scib_metrics(ad, "X_emb", label_key="cell_type")
