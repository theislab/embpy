"""Tests for BioEmbedderProvider.embed_with_status.

The real BioEmbedder is too heavy to import in CI; we monkey-patch
``BioEmbedderProvider._get_embedder`` with a stub that returns a fake
embedder object whose ``embed_genes_batch`` is whatever the test
specifies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from world_model.data.embeddings import (
    BioEmbedderProvider,
    EmbeddingStatus,
    make_control_vector,
)


class _StubEmbedder:
    """Returns a fixed-dim vector per resolvable symbol and None otherwise."""

    def __init__(
        self,
        resolved: dict[str, np.ndarray],
    ) -> None:
        self._resolved = resolved

    def embed_genes_batch(
        self,
        *,
        model: str,
        identifiers: Sequence[str],
        **_: object,
    ) -> list[np.ndarray | None]:
        del model
        return [self._resolved.get(i) for i in identifiers]


class _StandardPayloadEmbedder:
    """Minimal stub for the public BioEmbedder.embed(output='payload') path."""

    def __init__(self, resolved: dict[str, np.ndarray]) -> None:
        self._resolved = resolved
        self.embed_calls: list[list[str]] = []

    def embed(
        self,
        identifiers: Sequence[str],
        *,
        model: str,
        output: str,
        **_: object,
    ) -> dict[str, object]:
        del model
        assert output == "payload"
        ids = list(identifiers)
        self.embed_calls.append(ids)
        kept = [symbol for symbol in ids if symbol in self._resolved]
        return {
            "matrix": np.stack([self._resolved[symbol] for symbol in kept], axis=0),
            "entity_ids": [f"ENSG_{symbol}" for symbol in kept],
            "aliases": {f"ENSG_{symbol}": {"gene_symbol": symbol} for symbol in kept},
        }


def _make_provider(
    tmp_path,
    *,
    resolved: dict[str, np.ndarray],
) -> BioEmbedderProvider:
    """Build a provider whose embedder is a deterministic stub."""
    provider = BioEmbedderProvider(
        model_name="stub_v0",
        cache_dir=str(tmp_path / "cache"),
        organism="human",
    )
    provider._embedder = _StubEmbedder(resolved)  # bypass lazy import
    return provider


def test_provider_uses_public_bioembedder_embed_payload(tmp_path) -> None:
    provider = BioEmbedderProvider(
        model_name="stub_v0",
        cache_dir=str(tmp_path / "cache"),
        organism="human",
    )
    embedder = _StandardPayloadEmbedder({"TP53": np.asarray([1.0, 2.0], dtype=np.float32)})
    provider._embedder = embedder

    rows, statuses = provider.embed_with_status(["TP53", "UNKNOWN"])

    assert embedder.embed_calls == [["TP53", "UNKNOWN"]]
    assert statuses.tolist() == [
        EmbeddingStatus.RESOLVED.value,
        EmbeddingStatus.UNRESOLVED.value,
    ]
    np.testing.assert_allclose(rows[0], [1.0, 2.0])
    np.testing.assert_allclose(rows[1], [0.0, 0.0])


def test_three_status_buckets(tmp_path) -> None:
    dim = 8
    vec_tp53 = np.linspace(0.1, 0.8, dim).astype(np.float32)
    vec_myc = np.linspace(-0.4, 0.3, dim).astype(np.float32)
    provider = _make_provider(
        tmp_path,
        resolved={"TP53": vec_tp53, "MYC": vec_myc},
    )

    symbols = [
        "TP53",  # RESOLVED
        "non-targeting",  # CONTROL
        "NTC",  # CONTROL (regex variant)
        "AAVS1_2",  # CONTROL (regex variant)
        "BRCA_does_not_exist",  # UNRESOLVED
        "TP53+MYC",  # RESOLVED (combo)
        "TP53+non-targeting",  # mixed -> RESOLVED row (TP53 only)
        "NT5C2",  # REAL gene that starts with NT; UNRESOLVED here
    ]
    rows, statuses = provider.embed_with_status(symbols)
    assert rows.shape == (len(symbols), dim)
    assert statuses.shape == (len(symbols),)
    assert statuses[0] == EmbeddingStatus.RESOLVED.value
    assert statuses[1] == EmbeddingStatus.CONTROL.value
    assert statuses[2] == EmbeddingStatus.CONTROL.value
    assert statuses[3] == EmbeddingStatus.CONTROL.value
    assert statuses[4] == EmbeddingStatus.UNRESOLVED.value
    assert statuses[5] == EmbeddingStatus.RESOLVED.value
    assert statuses[6] == EmbeddingStatus.RESOLVED.value
    assert statuses[7] == EmbeddingStatus.UNRESOLVED.value

    # CONTROL rows are deterministic and non-zero.
    ctrl_ref = make_control_vector(dim, seed=provider.control_sentinel_seed)
    np.testing.assert_allclose(rows[1], ctrl_ref)
    np.testing.assert_allclose(rows[2], ctrl_ref)
    np.testing.assert_allclose(rows[3], ctrl_ref)
    assert np.linalg.norm(rows[1]) > 0.0

    # UNRESOLVED rows are zero.
    np.testing.assert_array_equal(rows[4], np.zeros(dim, dtype=np.float32))
    np.testing.assert_array_equal(rows[7], np.zeros(dim, dtype=np.float32))

    # Combo row is the mean of TP53 + MYC.
    expected_combo = ((vec_tp53 + vec_myc) / 2.0).astype(np.float32)
    np.testing.assert_allclose(rows[5], expected_combo, rtol=1e-6)

    # Mixed label keeps only TP53.
    np.testing.assert_allclose(rows[6], vec_tp53)


def test_control_vector_deterministic_across_calls(tmp_path) -> None:
    provider = _make_provider(tmp_path, resolved={"TP53": np.ones(4, dtype=np.float32)})
    rows_a, _ = provider.embed_with_status(["TP53", "non-targeting"])
    rows_b, _ = provider.embed_with_status(["non-targeting", "TP53"])
    np.testing.assert_array_equal(rows_a[1], rows_b[0])


def test_control_vector_seed_changes_value(tmp_path) -> None:
    provider_a = _make_provider(
        tmp_path / "a",
        resolved={"TP53": np.ones(4, dtype=np.float32)},
    )
    provider_b = BioEmbedderProvider(
        model_name="stub_v0",
        cache_dir=str(tmp_path / "b"),
        organism="human",
        control_sentinel_seed=7,
    )
    provider_b._embedder = _StubEmbedder({"TP53": np.ones(4, dtype=np.float32)})
    rows_a, _ = provider_a.embed_with_status(["non-targeting"])
    rows_b, _ = provider_b.embed_with_status(["non-targeting"])
    assert not np.allclose(rows_a[0], rows_b[0])


def test_combo_with_one_missing_piece_reports_partial(tmp_path) -> None:
    dim = 4
    provider = _make_provider(
        tmp_path,
        resolved={"TP53": np.ones(dim, dtype=np.float32)},
    )
    rows, statuses = provider.embed_with_status(["TP53+UNKNOWN"])
    # Row remains RESOLVED but the unresolved component is logged.
    assert statuses[0] == EmbeddingStatus.RESOLVED.value
    np.testing.assert_allclose(rows[0], np.ones(dim, dtype=np.float32))
    assert "UNKNOWN" in provider._last_unresolved


def test_empty_input_is_handled(tmp_path) -> None:
    provider = _make_provider(tmp_path, resolved={"TP53": np.ones(2, dtype=np.float32)})
    rows, statuses = provider.embed_with_status([])
    assert rows.shape == (0, 0)
    assert statuses.shape == (0,)


def test_legacy_embed_emits_deprecation_warning(tmp_path) -> None:
    provider = _make_provider(
        tmp_path,
        resolved={"TP53": np.ones(3, dtype=np.float32)},
    )
    with pytest.warns(DeprecationWarning):
        rows = provider.embed(["TP53", "non-targeting"])
    assert rows.shape == (2, 3)
    # Control row is non-zero deterministic sentinel.
    assert np.linalg.norm(rows[1]) > 0.0


def test_metadata_reports_per_bucket_counts(tmp_path) -> None:
    provider = _make_provider(
        tmp_path,
        resolved={"TP53": np.ones(3, dtype=np.float32)},
    )
    provider.embed_with_status(["TP53", "non-targeting", "UNKNOWN_GENE"])
    meta = provider.metadata(n_symbols=3, n_unresolved=1)
    assert meta.n_resolved == 1
    assert meta.n_control == 1
    assert meta.n_unresolved == 1
    assert "UNKNOWN_GENE" in meta.unresolved_symbols
    assert "non-targeting" in meta.control_symbols
