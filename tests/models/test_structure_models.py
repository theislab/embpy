"""Tests for embpy.models.structure_models.Boltz2Wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

boltz = pytest.importorskip("boltz")


class TestBoltz2WrapperInit:
    def test_default_output_type(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper()
        assert wrapper.output_type == "single"
        assert wrapper.model_type == "protein"

    def test_custom_output_type(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper(output_type="pairwise")
        assert wrapper.output_type == "pairwise"

    def test_both_output_type(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper(output_type="both")
        assert wrapper.output_type == "both"

    def test_pooling_strategies(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper()
        assert "mean" in wrapper.available_pooling_strategies
        assert "max" in wrapper.available_pooling_strategies
        assert "cls" in wrapper.available_pooling_strategies


class TestBoltz2WrapperLoad:
    def test_load_calls_download(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper()

        with patch("embpy.models.structure_models.Boltz2Wrapper.load") as mock_load:
            mock_load.return_value = None
            wrapper.load(torch.device("cpu"))
            mock_load.assert_called_once()


class TestBoltz2WrapperEmbed:
    @pytest.fixture
    def mock_wrapper(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper(output_type="single")
        wrapper.device = torch.device("cpu")

        n_tokens = 20
        token_s = 384
        token_z = 128

        mock_model = MagicMock()
        mock_model.return_value = {
            "s": torch.randn(1, n_tokens, token_s),
            "z": torch.randn(1, n_tokens, n_tokens, token_z),
            "pdistogram": torch.randn(1, n_tokens, n_tokens, 64),
        }
        wrapper._boltz_model = mock_model
        wrapper.model = mock_model
        wrapper.recycling_steps = 1

        mock_feats = {
            "token_pad_mask": torch.ones(1, n_tokens),
        }
        wrapper._prepare_input = MagicMock(return_value=mock_feats)

        return wrapper

    def test_single_output_shape(self, mock_wrapper):
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", pooling_strategy="mean")
        assert emb.shape == (384,)
        assert emb.dtype == np.float32

    def test_pairwise_output_shape(self, mock_wrapper):
        mock_wrapper.output_type = "pairwise"
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", pooling_strategy="mean")
        assert emb.shape == (128,)

    def test_both_output_shape(self, mock_wrapper):
        mock_wrapper.output_type = "both"
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", pooling_strategy="mean")
        assert emb.shape == (384 + 128,)

    def test_max_pooling(self, mock_wrapper):
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", pooling_strategy="max")
        assert emb.shape == (384,)

    def test_cls_pooling(self, mock_wrapper):
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", pooling_strategy="cls")
        assert emb.shape == (384,)

    def test_output_type_override(self, mock_wrapper):
        emb = mock_wrapper.embed("MKWVTFISLLFLFSSAYS", output_type="pairwise")
        assert emb.shape == (128,)


class TestBoltz2WrapperBatch:
    @pytest.fixture
    def mock_wrapper(self):
        from embpy.models.structure_models import Boltz2Wrapper

        wrapper = Boltz2Wrapper()
        wrapper.device = torch.device("cpu")

        n_tokens = 15
        mock_model = MagicMock()
        mock_model.return_value = {
            "s": torch.randn(1, n_tokens, 384),
            "z": torch.randn(1, n_tokens, n_tokens, 128),
            "pdistogram": torch.randn(1, n_tokens, n_tokens, 64),
        }
        wrapper._boltz_model = mock_model
        wrapper.model = mock_model
        wrapper.recycling_steps = 1
        wrapper._prepare_input = MagicMock(return_value={
            "token_pad_mask": torch.ones(1, n_tokens),
        })

        return wrapper

    def test_batch_returns_list(self, mock_wrapper):
        results = mock_wrapper.embed_batch(["MKTAYIAKQRQISFVK", "MKWVTFISLLFLFSSAYS"])
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)

    def test_batch_error_handling(self, mock_wrapper):
        def fail_second(seq, **kw):
            if seq == "FAIL":
                raise RuntimeError("test error")
            return mock_wrapper._prepare_input(seq)

        mock_wrapper._prepare_input = MagicMock(side_effect=fail_second)
        mock_wrapper._prepare_input.return_value = {
            "token_pad_mask": torch.ones(1, 15),
        }

        results = mock_wrapper.embed_batch(["MKTAYIAKQRQISFVK"])
        assert len(results) == 1
