"""Tests for DNA model wrappers (Enformer, Borzoi, Evo, Evo2) using mocks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models.dna_models import BorzoiWrapper, EnformerWrapper, EvoWrapper


class TestEnformerWrapper:
    def test_init_defaults(self):
        w = EnformerWrapper()
        assert w.model_name == "EleutherAI/enformer-official-rough"
        assert w.model_type == "dna"
        assert w.SEQUENCE_LENGTH == 196_608
        assert w.TRUNK_OUTPUT_DIM == 3072

    def test_init_custom_name(self):
        w = EnformerWrapper(model_path_or_name="custom/enformer")
        assert w.model_name == "custom/enformer"

    def test_embed_without_load_raises(self):
        w = EnformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = EnformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = EnformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_strategy_raises(self):
        w = EnformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_preprocess_pads_short_sequence(self):
        w = EnformerWrapper()
        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w._preprocess_sequence("ACGT")
            assert result.shape[1] == w.SEQUENCE_LENGTH

    def test_preprocess_truncates_long_sequence(self):
        w = EnformerWrapper()
        long_seq = "A" * (w.SEQUENCE_LENGTH + 100)
        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w._preprocess_sequence(long_seq)
            assert result.shape[1] == w.SEQUENCE_LENGTH

    def test_embed_with_mocked_model(self):
        w = EnformerWrapper()
        w.device = torch.device("cpu")

        num_bins = 896
        trunk_tensor = torch.randn(1, num_bins, 3072)
        w.model = MagicMock()
        w.model.return_value = (None, trunk_tensor)

        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w.embed("ACGT", pooling_strategy="mean")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3072,)
        assert not np.isnan(result).any()

    def test_embed_max_pooling(self):
        w = EnformerWrapper()
        w.device = torch.device("cpu")

        num_bins = 896
        trunk_tensor = torch.randn(1, num_bins, 3072)
        w.model = MagicMock()
        w.model.return_value = (None, trunk_tensor)

        with patch("embpy.models.dna_models.seq_indices_to_one_hot") as mock_one_hot:
            mock_one_hot.return_value = torch.zeros(1, w.SEQUENCE_LENGTH, 5)
            result = w.embed("ACGT", pooling_strategy="max")

        assert result.shape == (3072,)


class TestBorzoiWrapper:
    def test_init_defaults(self):
        w = BorzoiWrapper()
        assert w.model_name == "johahi/borzoi-replicate-0"
        assert w.model_type == "dna"
        assert w.SEQUENCE_LENGTH == 524_288
        assert w.NUM_CHANNELS == 4

    def test_embed_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = BorzoiWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = BorzoiWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_preprocess_short_sequence(self):
        w = BorzoiWrapper()
        result = w._preprocess_sequence("ACGT")
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_long_sequence(self):
        w = BorzoiWrapper()
        long_seq = "A" * (w.SEQUENCE_LENGTH + 100)
        result = w._preprocess_sequence(long_seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_exact_length(self):
        w = BorzoiWrapper()
        seq = "A" * w.SEQUENCE_LENGTH
        result = w._preprocess_sequence(seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_embed_with_mocked_model(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")

        hidden_dim = 512
        num_bins = 100
        w.TRUNK_OUTPUT_DIM = hidden_dim
        embs_tensor = torch.randn(1, hidden_dim, num_bins)

        mock_model = MagicMock()
        mock_model.get_embs_after_crop.return_value = embs_tensor
        w.model = mock_model

        result = w.embed("ACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()


class TestEvo2Wrapper:
    """Tests for the Evo2Wrapper (mocked, since evo2 may not be installed)."""

    def test_import_available(self):
        from embpy.models.dna_models import Evo2Wrapper

        assert Evo2Wrapper is not None

    def test_init_defaults(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        assert w.model_name == "evo2_7b"
        assert w.model_type == "dna"
        assert w.layer_name is None

    def test_init_custom_layer(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper(layer_name="blocks.10.mlp.l3")
        assert w.layer_name == "blocks.10.mlp.l3"

    def test_layer_defaults_mapping(self):
        from embpy.models.dna_models import Evo2Wrapper

        assert "evo2_7b" in Evo2Wrapper.LAYER_DEFAULTS
        assert "evo2_40b" in Evo2Wrapper.LAYER_DEFAULTS
        assert "evo2_1b_base" in Evo2Wrapper.LAYER_DEFAULTS

    def test_embed_without_load_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w._evo2_model = MagicMock()
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w._evo2_model = MagicMock()
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_load_without_evo2_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        with patch("embpy.models.dna_models._HAVE_EVO2", False):
            with pytest.raises(ImportError, match="evo2 package is not installed"):
                w.load(torch.device("cpu"))

    def test_embed_with_mocked_evo2(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper(model_path_or_name="evo2_7b")
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGTACGTAC", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_pooling(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGT", pooling_strategy="cls")
        assert result.shape == (hidden_dim,)

    def test_embed_max_pooling(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10
        embedding_tensor = torch.randn(1, seq_len, hidden_dim)

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))
        mock_evo2.return_value = (None, {"blocks.28.mlp.l3": embedding_tensor})
        w._evo2_model = mock_evo2

        result = w.embed("ACGT", pooling_strategy="max")
        assert result.shape == (hidden_dim,)

    def test_embed_missing_layer_raises(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = [0, 1, 2, 3]
        mock_evo2.return_value = (None, {"some.other.layer": torch.randn(1, 4, 100)})
        w._evo2_model = mock_evo2

        with pytest.raises(ValueError, match="Layer.*not found"):
            w.embed("ACGT")

    def test_embed_batch_with_mocked_evo2(self):
        from embpy.models.dna_models import Evo2Wrapper

        w = Evo2Wrapper()
        w.layer_name = "blocks.28.mlp.l3"
        w.device = torch.device("cpu")

        hidden_dim = 4096
        seq_len = 10

        mock_evo2 = MagicMock()
        mock_evo2.tokenizer.tokenize.return_value = list(range(seq_len))

        def mock_forward(ids, return_embeddings=False, layer_names=None):
            return (None, {"blocks.28.mlp.l3": torch.randn(1, seq_len, hidden_dim)})

        mock_evo2.side_effect = mock_forward
        w._evo2_model = mock_evo2

        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)


# =====================================================================
#                       EVO (v1 / v1.5) WRAPPER
# =====================================================================


class TestEvoWrapper:
    """Tests for EvoWrapper (mocked — evo-model may not be installed)."""

    # --- Initialization ---

    def test_init_defaults(self):
        w = EvoWrapper()
        assert w.model_name == "evo-1-8k-base"
        assert w.model_type == "dna"
        assert w.embedding_layer is None
        assert w._evo_model is None
        assert w._tokenizer is None

    def test_init_custom_name(self):
        w = EvoWrapper(model_path_or_name="evo-1-131k-base")
        assert w.model_name == "evo-1-131k-base"

    def test_init_custom_embedding_layer(self):
        w = EvoWrapper(embedding_layer=10)
        assert w.embedding_layer == 10

    def test_available_models_list(self):
        assert "evo-1-8k-base" in EvoWrapper.AVAILABLE_MODELS
        assert "evo-1-131k-base" in EvoWrapper.AVAILABLE_MODELS
        assert "evo-1.5-8k-base" in EvoWrapper.AVAILABLE_MODELS
        assert "evo-1-8k-crispr" in EvoWrapper.AVAILABLE_MODELS
        assert "evo-1-8k-transposon" in EvoWrapper.AVAILABLE_MODELS

    def test_pooling_strategies(self):
        assert EvoWrapper.available_pooling_strategies == ["mean", "max", "cls"]

    # --- Error handling (before load) ---

    def test_embed_without_load_raises(self):
        w = EvoWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = EvoWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_load_without_evo_raises(self):
        w = EvoWrapper()
        with patch("embpy.models.dna_models._HAVE_EVO", False):
            with pytest.raises(ImportError, match="evo-model package is not installed"):
                w.load(torch.device("cpu"))

    # --- Mocked load ---

    def _make_mock_sh_model(self, num_blocks: int = 32, hidden_dim: int = 4096) -> MagicMock:
        """Create a mock StripedHyena model with blocks."""
        mock_blocks = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        mock_model = MagicMock()
        mock_model.blocks = mock_blocks
        mock_model.eval = MagicMock(return_value=None)
        return mock_model

    def _make_loaded_wrapper(
        self,
        num_blocks: int = 32,
        hidden_dim: int = 4096,
        seq_len: int = 10,
        embedding_layer: int | None = None,
    ) -> EvoWrapper:
        """Create an EvoWrapper with a fully mocked model, ready for embed()."""
        w = EvoWrapper(embedding_layer=embedding_layer)
        w.device = torch.device("cpu")

        mock_model = self._make_mock_sh_model(num_blocks, hidden_dim)
        hidden_tensor = torch.randn(1, seq_len, hidden_dim)

        w._evo_model = mock_model
        w._tokenizer = MagicMock()
        w._tokenizer.tokenize.return_value = list(range(seq_len))
        w.model = mock_model

        if embedding_layer is None:
            w.embedding_layer = num_blocks // 2

        # Patch _extract_hidden_state to return a known tensor
        def patched_extract(input_ids: Any) -> torch.Tensor:
            return hidden_tensor

        w._extract_hidden_state = patched_extract  # type: ignore[assignment]
        return w

    def test_load_sets_default_embedding_layer(self):
        w = EvoWrapper()
        mock_model = self._make_mock_sh_model(num_blocks=32)
        mock_tokenizer = MagicMock()

        mock_evo_cls = MagicMock()
        mock_evo_instance = MagicMock()
        mock_evo_instance.model = mock_model
        mock_evo_instance.tokenizer = mock_tokenizer
        mock_evo_cls.return_value = mock_evo_instance

        with patch("embpy.models.dna_models._HAVE_EVO", True), patch("embpy.models.dna_models.EvoModel", mock_evo_cls):
            w.load(torch.device("cpu"))

        assert w.embedding_layer == 16  # 32 // 2
        assert w._evo_model is not None
        assert w._tokenizer is not None
        assert w.device == torch.device("cpu")

    def test_load_already_loaded_skips(self):
        w = EvoWrapper()
        w._evo_model = MagicMock()  # Pretend already loaded
        w.load(torch.device("cpu"))  # Should not raise

    def test_load_invalid_embedding_layer_raises(self):
        w = EvoWrapper(embedding_layer=999)
        mock_model = self._make_mock_sh_model(num_blocks=32)
        mock_evo_cls = MagicMock()
        mock_evo_instance = MagicMock()
        mock_evo_instance.model = mock_model
        mock_evo_instance.tokenizer = MagicMock()
        mock_evo_cls.return_value = mock_evo_instance

        with patch("embpy.models.dna_models._HAVE_EVO", True), patch("embpy.models.dna_models.EvoModel", mock_evo_cls):
            with pytest.raises(RuntimeError):
                w.load(torch.device("cpu"))

    def test_load_negative_embedding_layer_raises(self):
        w = EvoWrapper(embedding_layer=-1)
        mock_model = self._make_mock_sh_model(num_blocks=32)
        mock_evo_cls = MagicMock()
        mock_evo_instance = MagicMock()
        mock_evo_instance.model = mock_model
        mock_evo_instance.tokenizer = MagicMock()
        mock_evo_cls.return_value = mock_evo_instance

        with patch("embpy.models.dna_models._HAVE_EVO", True), patch("embpy.models.dna_models.EvoModel", mock_evo_cls):
            with pytest.raises(RuntimeError):
                w.load(torch.device("cpu"))

    # --- Embedding (mocked) ---

    def test_embed_mean_pooling(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        result = w.embed("ACGTACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert result.dtype == np.float32
        assert not np.isnan(result).any()

    def test_embed_max_pooling(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        result = w.embed("ACGTACGT", pooling_strategy="max")
        assert result.shape == (hidden_dim,)

    def test_embed_cls_pooling(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        result = w.embed("ACGTACGT", pooling_strategy="cls")
        assert result.shape == (hidden_dim,)

    def test_embed_invalid_pooling_raises(self):
        w = self._make_loaded_wrapper()
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_embed_layer_override(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)
        original_layer = w.embedding_layer

        result = w.embed("ACGT", embedding_layer=5)
        assert result.shape == (hidden_dim,)
        # Verify the original embedding_layer is restored
        assert w.embedding_layer == original_layer

    def test_embed_different_inputs_differ(self):
        hidden_dim = 64
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        # Override the mock to return different tensors for different calls
        call_count = [0]

        def varying_extract(input_ids):
            call_count[0] += 1
            torch.manual_seed(call_count[0])
            return torch.randn(1, 10, hidden_dim)

        w._extract_hidden_state = varying_extract  # type: ignore[assignment]

        emb1 = w.embed("ACGT")
        emb2 = w.embed("TTTT")
        assert not np.allclose(emb1, emb2)

    # --- Batch embedding ---

    def test_embed_batch_empty_returns_empty(self):
        w = EvoWrapper()
        w._evo_model = MagicMock()
        assert w.embed_batch([]) == []

    def test_embed_batch_multiple(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.shape == (hidden_dim,)

    def test_embed_batch_single(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        results = w.embed_batch(["ACGT"])
        assert len(results) == 1
        assert results[0].shape == (hidden_dim,)

    def test_embed_batch_with_layer_override(self):
        hidden_dim = 4096
        w = self._make_loaded_wrapper(hidden_dim=hidden_dim)

        results = w.embed_batch(["ACGT", "GCTA"], embedding_layer=5)
        assert len(results) == 2
        for r in results:
            assert r.shape == (hidden_dim,)

    # --- _extract_hidden_state (hook mechanism) ---

    def test_extract_hidden_state_captures_output(self):
        """Test that the forward hook correctly captures the block output."""
        hidden_dim = 64
        seq_len = 8
        num_blocks = 4
        w = EvoWrapper(embedding_layer=2)
        w.device = torch.device("cpu")

        # Build a real small model with actual blocks
        blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            block = torch.nn.Linear(hidden_dim, hidden_dim)
            blocks.append(block)

        # Wrap in a callable module mock that runs the blocks
        expected_output = torch.randn(1, seq_len, hidden_dim)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = blocks

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x, None

        fake_model = FakeModel()

        def patched_block_forward(x: Any) -> torch.Tensor:
            return expected_output.squeeze(0)  # Linear expects 2D

        fake_model.blocks[2].forward = patched_block_forward

        w._evo_model = fake_model
        w.embedding_layer = 2

        input_ids = torch.randn(1, seq_len, hidden_dim)
        result = w._extract_hidden_state(input_ids)
        assert isinstance(result, torch.Tensor)
