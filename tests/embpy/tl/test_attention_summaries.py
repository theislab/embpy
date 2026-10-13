"""Tests for 2-D attention summaries.

Every summary must return ``(n_entities, n_dims)`` so it satisfies the
``EmbeddingResult`` contract and inherits the existing exporters -- a rank-4
attention tensor cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from embpy.tl import (
    attention_entropy,
    attention_to_gene_set,
    head_uniformity,
    received_attention,
    summarize_attention,
)


@pytest.fixture
def attn():
    """(batch=3, heads=4, seq=10, seq=10) with each query row summing to 1."""
    rng = np.random.default_rng(0)
    return rng.dirichlet(np.ones(10), size=(3, 4, 10))


class TestOutputContract:
    """The whole point: 2-D (n_entities, n_dims) float32."""

    def test_entropy_is_batch_by_heads(self, attn):
        assert attention_entropy(attn).shape == (3, 4)

    def test_received_is_batch_by_tokens(self, attn):
        assert received_attention(attn).shape == (3, 10)

    def test_uniformity_is_batch_by_heads(self, attn):
        assert head_uniformity(attn).shape == (3, 4)

    def test_gene_set_is_batch_by_heads(self, attn):
        assert attention_to_gene_set(attn, [0, 2, 4]).shape == (3, 4)

    @pytest.mark.parametrize("fn", [attention_entropy, received_attention, head_uniformity])
    def test_all_are_2d_float32_and_finite(self, fn, attn):
        out = fn(attn)
        assert out.ndim == 2
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_rank3_input_treated_as_single_head(self, attn):
        single = attn[:, 0]  # (batch, seq, seq)
        assert attention_entropy(single).shape == (3, 1)


class TestEntropy:
    def test_uniform_attention_has_maximum_entropy(self):
        seq = 8
        uniform = np.full((1, 1, seq, seq), 1.0 / seq)
        assert attention_entropy(uniform)[0, 0] == pytest.approx(np.log(seq), abs=1e-5)

    def test_one_hot_attention_has_zero_entropy(self):
        seq = 8
        onehot = np.zeros((1, 1, seq, seq))
        onehot[..., 0] = 1.0
        assert attention_entropy(onehot)[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_focused_scores_below_diffuse(self, attn):
        seq = 10
        uniform = np.full((1, 1, seq, seq), 1.0 / seq)
        focused = np.zeros((1, 1, seq, seq))
        focused[..., 0] = 1.0
        assert attention_entropy(focused)[0, 0] < attention_entropy(uniform)[0, 0]

    def test_zero_entries_do_not_produce_nan(self):
        """0 * log(0) must be 0, not NaN -- np.log(where=) needs an explicit out=."""
        seq = 6
        sparse = np.zeros((1, 1, seq, seq))
        sparse[..., 0] = 0.5
        sparse[..., 1] = 0.5
        out = attention_entropy(sparse)
        assert np.isfinite(out).all()
        assert out[0, 0] == pytest.approx(np.log(2), abs=1e-6)


class TestReceivedAttention:
    def test_mass_sums_to_one_when_normalised(self, attn):
        got = received_attention(attn, normalize=True)
        assert np.allclose(got.sum(axis=1), 1.0, atol=1e-5)

    def test_unnormalised_sums_to_seq_len(self, attn):
        got = received_attention(attn, normalize=False)
        assert np.allclose(got.sum(axis=1), attn.shape[-1], atol=1e-4)

    def test_identifies_the_attended_token(self):
        seq = 7
        a = np.zeros((1, 1, seq, seq))
        a[..., 3] = 1.0  # every query attends to token 3
        assert int(np.argmax(received_attention(a)[0])) == 3


class TestHeadUniformity:
    def test_uniform_head_scores_zero(self):
        seq = 9
        uniform = np.full((1, 1, seq, seq), 1.0 / seq)
        assert head_uniformity(uniform)[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_one_hot_head_scores_one(self):
        seq = 9
        onehot = np.zeros((1, 1, seq, seq))
        onehot[..., 2] = 1.0
        assert head_uniformity(onehot)[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_in_unit_interval(self, attn):
        out = head_uniformity(attn)
        assert (out >= 0).all() and (out <= 1).all()


class TestGeneSet:
    def test_full_set_captures_all_mass(self, attn):
        got = attention_to_gene_set(attn, list(range(attn.shape[-1])))
        assert np.allclose(got, 1.0, atol=1e-5)

    def test_targeted_set_captures_its_share(self):
        seq = 6
        a = np.zeros((1, 1, seq, seq))
        a[..., 1] = 1.0
        assert attention_to_gene_set(a, [1])[0, 0] == pytest.approx(1.0)
        assert attention_to_gene_set(a, [0, 2])[0, 0] == pytest.approx(0.0)

    def test_empty_indices_raise(self, attn):
        with pytest.raises(ValueError, match="indices is empty"):
            attention_to_gene_set(attn, [])

    def test_out_of_range_indices_raise(self, attn):
        with pytest.raises(IndexError, match="must lie in"):
            attention_to_gene_set(attn, [0, 999])


class TestDispatch:
    @pytest.mark.parametrize("kind", ["entropy", "received", "uniformity"])
    def test_dispatch_matches_direct_call(self, kind, attn):
        direct = {"entropy": attention_entropy, "received": received_attention, "uniformity": head_uniformity}[kind]
        assert np.allclose(summarize_attention(attn, kind), direct(attn))

    def test_unknown_kind_raises(self, attn):
        with pytest.raises(ValueError, match="kind must be one of"):
            summarize_attention(attn, "importance")


class TestValidation:
    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            attention_entropy(np.zeros((1, 2, 4, 5)))

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match="must be"):
            attention_entropy(np.zeros((4, 4)))

    def test_accepts_torch_tensors(self, attn):
        torch = pytest.importorskip("torch")
        got = attention_entropy(torch.tensor(attn))
        assert got.shape == (3, 4)
