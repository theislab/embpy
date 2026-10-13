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


# --- The returned column set ------------------------------------------------
# The heavy path is skipped in the default CI job, so the shape of the report
# was never asserted anywhere. That is how `isolated_label_asw` shipped as a
# guaranteed-NaN column: scIB identifies an isolated label by how few batches
# it appears in, so calling it with batch_key=None raises inside scib and
# _safe_metric turned that into NaN plus one warning per embedding. These two
# tests pin the contract with a stubbed scib, so no optional dependency is
# needed to catch a regression.


class _FakeMetrics:
    """Every scib metric the scorer calls, returning fixed values."""

    def cluster_optimal_resolution(self, ad, *, label_key, cluster_key, resolutions, verbose):
        ad.obs[cluster_key] = ad.obs[label_key].astype(str)

    def nmi(self, ad, cluster_key, label_key):
        return 0.8

    def ari(self, ad, cluster_key, label_key):
        return 0.6

    def silhouette(self, ad, label_key, embed_key):
        return 0.4

    def isolated_labels(self, ad, label_key, batch_key, embed_key, cluster, verbose):
        # Mirror scib: the batch covariate is not optional here.
        if batch_key is None:
            raise KeyError("[nan] not in index")
        return 0.2

    def clisi_graph(self, ad, label_key, type_, use_rep):
        return 1.0

    def silhouette_batch(self, ad, batch_key, label_key, embed_key, verbose):
        return 0.9

    def graph_connectivity(self, ad, label_key):
        return 0.7

    def ilisi_graph(self, ad, batch_key, type_, use_rep):
        return 0.5

    def kBET(self, ad, batch_key, label_key, type_, embed):
        return 0.3


class _FakeScib:
    def __init__(self):
        self.metrics = _FakeMetrics()


class _FakeScanpyPP:
    def neighbors(self, ad, use_rep, n_neighbors):
        pass


class _FakeScanpy:
    def __init__(self):
        self.pp = _FakeScanpyPP()


@pytest.fixture
def stub_scib(monkeypatch):
    from embpy.tl import scib_metrics as mod

    monkeypatch.setattr(mod, "_load_scib", lambda: _FakeScib())
    monkeypatch.setattr(mod, "_load_scanpy", lambda: _FakeScanpy())


def test_no_batch_key_omits_isolated_label_asw(stub_scib):
    report = compute_scib_metrics(_toy_adata(), "X_emb", label_key="cell_type")

    # Absent, not NaN: a column that cannot be computed should not be shown.
    assert "isolated_label_asw" not in report.columns
    assert not any(c in report.columns for c in ("asw_batch", "graph_conn", "ilisi", "kbet"))

    # bio_conservation averages the four metrics that were computable.
    assert report.loc["X_emb", "bio_conservation"] == pytest.approx(
        np.mean([0.8, 0.6, 0.4, 1.0])
    )
    assert np.isnan(report.loc["X_emb", "batch_correction"])
    assert report.loc["X_emb", "total"] == pytest.approx(report.loc["X_emb", "bio_conservation"])


def test_batch_key_adds_both_blocks(stub_scib):
    report = compute_scib_metrics(
        _toy_adata(), "X_emb", label_key="cell_type", batch_key="batch"
    )

    for col in ("nmi", "ari", "asw_label", "isolated_label_asw", "clisi",
                "asw_batch", "graph_conn", "ilisi", "kbet"):
        assert col in report.columns, col
        assert np.isfinite(report.loc["X_emb", col]), col

    bio = np.mean([0.8, 0.6, 0.4, 0.2, 1.0])
    batch = np.mean([0.9, 0.7, 0.5, 0.3])
    assert report.loc["X_emb", "bio_conservation"] == pytest.approx(bio)
    assert report.loc["X_emb", "batch_correction"] == pytest.approx(batch)
    assert report.loc["X_emb", "total"] == pytest.approx(0.6 * bio + 0.4 * batch)
