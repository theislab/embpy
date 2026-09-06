"""Tests for DNA model wrappers (Enformer, Borzoi, Evo, Evo2) using mocks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models.dna_models import (
    BorzoiWrapper,
    EnformerWrapper,
    EvoWrapper,
    CaduceusWrapper,
    GENALMWrapper,
    HyenaDNAWrapper,
    NucleotideTransformerWrapper,
)


def _base_output(hidden_dim: int, seq_len: int = 20) -> MagicMock:
    """Mimic BaseModelOutput: last_hidden_state present."""
    h = torch.randn(1, seq_len, hidden_dim)
    out = MagicMock()
    out.last_hidden_state = h
    out.hidden_states = (h, h)
    return out


def _tok(seq_len: int = 20, has_mask: bool = True) -> MagicMock:
    """Return a mock tokeniser matching the real HF contract.

    ``_hf_batched_embed`` calls the tokenizer with ``return_tensors=None``,
    which a real HF tokenizer answers with plain ``list[int]`` -- not tensors.
    It then does its own chunking, special-token insertion and padding, and
    builds the ``input_ids``/``attention_mask`` tensors itself. So the mock
    must hand back lists: a tensor here makes the ``if ids and ...`` coercion
    guard raise "Boolean value of Tensor with more than one value is
    ambiguous", which is not a failure mode real tokenizers can produce.

    The special-token ids are real ints rather than auto-created MagicMock
    attributes so the chunk-size arithmetic (``inner_max = ctx - head - tail``)
    is deterministic and the constructed chunk length is predictable.
    """
    def _encode(*args: Any, **kwargs: Any) -> dict:
        # A real tokenizer switches container type on return_tensors: "pt"
        # yields batched tensors, None yields plain python lists. Wrappers use
        # both -- embed() tokenizes straight to tensors, while the chunking
        # helper asks for lists so it can pad and add special tokens itself.
        if kwargs.get("return_tensors") == "pt":
            enc: dict = {"input_ids": torch.ones(1, seq_len, dtype=torch.long)}
            if has_mask:
                enc["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)
            return enc
        enc = {"input_ids": [1] * seq_len}
        if has_mask:
            enc["attention_mask"] = [1] * seq_len
        return enc

    mock = MagicMock(side_effect=_encode)
    mock.cls_token_id = 2
    mock.sep_token_id = 3
    mock.bos_token_id = None
    mock.eos_token_id = None
    mock.pad_token_id = 0
    mock.model_max_length = 512
    return mock


# Number of special tokens _tok() causes _hf_batched_embed to prepend/append
# (one cls head + one sep tail), i.e. len(chunk) == len(input_ids) + _TOK_SPECIALS.
_TOK_SPECIALS = 2


def _dyn_model(
    hidden_dim: int,
    *,
    last_hidden: bool = False,
    layers: int = 2,
    fill: float | None = None,
    layer_fills: tuple[float, ...] | None = None,
) -> MagicMock:
    """Model mock whose output is shaped from the input it actually receives.

    A real transformer returns ``(batch, n_tokens_in, hidden)``. A mock with a
    static ``return_value`` cannot do that, so once ``_hf_batched_embed`` adds
    special tokens (or batches several chunks together) the mocked hidden
    state no longer lines up with the attention mask the helper built, and
    pooling dies with a tensor-size mismatch. Deriving the shape from
    ``input_ids`` keeps the mock faithful for any input length or batch size.
    """

    def _forward(*args: Any, **kwargs: Any) -> MagicMock:
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        n_batch, n_tok = (int(ids.shape[0]), int(ids.shape[1])) if ids is not None else (1, 20)

        def _layer(value: float | None) -> torch.Tensor:
            if value is None:
                return torch.randn(n_batch, n_tok, hidden_dim)
            return torch.full((n_batch, n_tok, hidden_dim), float(value))

        if layer_fills is not None:
            stack = tuple(_layer(v) for v in layer_fills)
        else:
            # Distinct draws per layer so target_layer selection is observable.
            stack = tuple(_layer(fill) for _ in range(layers))

        out = MagicMock()
        out.hidden_states = stack
        out.last_hidden_state = stack[-1] if last_hidden else None
        return out

    model = MagicMock(side_effect=_forward)
    model.config = MagicMock()
    model.config.max_position_embeddings = 512
    return model

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

    def test_preprocess_n_is_all_zero_not_adenine(self):
        """N must encode as an all-zero column, never as adenine.

        Mapping unknown bases to index 0 silently turns every masked, soft-masked
        or IUPAC-ambiguous base into a real A, fabricating sequence content the
        caller never supplied.
        """
        w = BorzoiWrapper()
        result = w._preprocess_sequence("ACGTN")

        # The 5 informative columns sit at the centre of the padded window.
        start = (w.SEQUENCE_LENGTH - 5) // 2
        window = result[0, :, start : start + 5]

        assert window[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
        assert window[:, 1].tolist() == [0.0, 1.0, 0.0, 0.0]  # C
        assert window[:, 2].tolist() == [0.0, 0.0, 1.0, 0.0]  # G
        assert window[:, 3].tolist() == [0.0, 0.0, 0.0, 1.0]  # T
        assert window[:, 4].tolist() == [0.0, 0.0, 0.0, 0.0], (
            "N encoded as a real nucleotide instead of an all-zero column"
        )

    def test_preprocess_unknown_characters_are_all_zero(self):
        """IUPAC codes and soft-masked bases follow the same rule as N."""
        w = BorzoiWrapper()
        # R/Y/S/W are IUPAC ambiguity codes; lowercase acgt is soft-masking and
        # must survive the upper() call as real bases.
        result = w._preprocess_sequence("RYSWacgt")
        start = (w.SEQUENCE_LENGTH - 8) // 2
        window = result[0, :, start : start + 8]

        assert window[:, :4].sum() == 0.0, "IUPAC ambiguity codes must be all-zero"
        # Soft-masked acgt are still real nucleotides after uppercasing.
        assert window[:, 4:].sum() == 4.0
        assert torch.equal(window[:, 4:], torch.eye(4))

    def test_preprocess_n_matches_padding_encoding(self):
        """An explicit N and an implicit pad column must be indistinguishable."""
        w = BorzoiWrapper()
        result = w._preprocess_sequence("N")
        # Every column, informative or padding, is all-zero for a lone N.
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)
        assert result.sum() == 0.0

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

    def test_predict_profile_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.predict_profile("ACGT")

    def test_predict_profile_returns_tracks_by_bins(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")

        num_tracks, num_bins = 7611, 50
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.rand(1, num_tracks, num_bins)
        w.model = mock_model

        profile = w.predict_profile("ACGT")
        assert isinstance(profile, np.ndarray)
        assert profile.shape == (num_tracks, num_bins)
        mock_model.forward.assert_called_once()
        assert mock_model.forward.call_args.kwargs.get("is_human") is True

    def test_predict_profile_track_subset(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")

        mock_model = MagicMock()
        mock_model.forward.return_value = torch.rand(1, 10, 5)
        w.model = mock_model

        profile = w.predict_profile("ACGT", track_indices=[0, 2, 4])
        assert profile.shape == (3, 5)

    def test_predict_profile_undo_squashed_scale_without_package_raises(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.rand(1, 4, 5)
        w.model = mock_model

        with patch("embpy.models.dna_models._undo_squashed_scale", None):
            with pytest.raises(ImportError):
                w.predict_profile("ACGT", undo_squashed_scale=True)

    def test_embed_return_profile_runs_second_forward(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")

        hidden_dim, num_bins = 32, 10
        num_tracks = 7611
        mock_model = MagicMock()
        mock_model.get_embs_after_crop.return_value = torch.randn(1, hidden_dim, num_bins)
        mock_model.forward.return_value = torch.rand(1, num_tracks, num_bins)
        w.model = mock_model

        embedding, profile = w.embed("ACGT", return_profile=True)
        assert embedding.shape == (hidden_dim,)
        assert profile.shape == (num_tracks, num_bins)

    def test_embed_default_does_not_call_forward(self):
        w = BorzoiWrapper()
        w.device = torch.device("cpu")
        mock_model = MagicMock()
        mock_model.get_embs_after_crop.return_value = torch.randn(1, 32, 10)
        w.model = mock_model

        w.embed("ACGT")
        mock_model.forward.assert_not_called()

    def test_profile_offset_bp_computed_from_crop(self):
        w = BorzoiWrapper()
        mock_model = MagicMock()
        mock_model.crop.target_length = 16352  # 16384 - 32, as in real Borzoi
        w.model = mock_model
        assert w.profile_offset_bp == (524_288 - 16352 * 32) // 2

    def test_profile_offset_bp_without_load_raises(self):
        w = BorzoiWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = w.profile_offset_bp

    def test_get_track_metadata_returns_dataframe(self):
        pytest.importorskip("borzoi_pytorch")  # get_track_metadata loads the real package
        df = BorzoiWrapper.get_track_metadata()
        assert "identifier" in df.columns
        assert len(df) > 0

    def test_get_track_categories_splits_rna_by_source_path(self):
        import pandas as pd

        tm = pd.DataFrame({
            "description": ["ATAC:pbmc", "DNASE:liver", "CAGE:blood", "CHIP:h3k4me3", "RNA:liver", "RNA:blood"],
            "file": [
                "atac.bw", "dnase.bw", "cage.bw", "chip.bw",
                "/human/rna/recount3/liver.bw", "/human/rna/encode/blood.bw",
            ],
        })
        cats = BorzoiWrapper.get_track_categories(tm)
        assert list(cats) == ["ATAC", "DNASE", "CAGE", "CHIP", "RNA_GTEx", "RNA_ENCODE"]
        assert list(cats.index) == list(tm.index)

    def test_get_track_categories_raises_on_unmatched_rna_file_path(self):
        import pandas as pd

        tm = pd.DataFrame({
            "description": ["RNA:liver"],
            "file": ["some/path/with/no/human_rna_prefix/liver.bw"],
        })
        with pytest.raises(ValueError, match="RNA track"):
            BorzoiWrapper.get_track_categories(tm)

    def test_get_track_categories_defaults_to_bundled_metadata(self):
        cats = BorzoiWrapper.get_track_categories()
        tm = BorzoiWrapper.get_track_metadata()
        assert len(cats) == len(tm)
        assert set(cats.unique()) >= {"ATAC", "DNASE", "CAGE", "CHIP", "RNA_GTEx", "RNA_ENCODE"}


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
        assert EvoWrapper.available_pooling_strategies == ["mean", "max", "cls", "none"]

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
        """An out-of-range layer is a caller error, so ValueError -- not RuntimeError.

        This test previously asserted ``RuntimeError``, which pinned a masking bug:
        the check ran *inside* ``load``'s ``try``, so the generic handler rewrote a
        precise "embedding_layer=999 is out of range" into "Could not load Evo",
        sending the reader after a broken checkpoint instead of a wrong argument.
        """
        w = EvoWrapper(embedding_layer=999)
        mock_model = self._make_mock_sh_model(num_blocks=32)
        mock_evo_cls = MagicMock()
        mock_evo_instance = MagicMock()
        mock_evo_instance.model = mock_model
        mock_evo_instance.tokenizer = MagicMock()
        mock_evo_cls.return_value = mock_evo_instance

        with patch("embpy.models.dna_models._HAVE_EVO", True), patch("embpy.models.dna_models.EvoModel", mock_evo_cls):
            with pytest.raises(ValueError) as excinfo:
                w.load(torch.device("cpu"))
            assert "out of range" in str(excinfo.value)
            assert "32 blocks" in str(excinfo.value)

    def test_load_negative_embedding_layer_raises(self):
        """Same contract for a negative index -- see the note above."""
        w = EvoWrapper(embedding_layer=-1)
        mock_model = self._make_mock_sh_model(num_blocks=32)
        mock_evo_cls = MagicMock()
        mock_evo_instance = MagicMock()
        mock_evo_instance.model = mock_model
        mock_evo_instance.tokenizer = MagicMock()
        mock_evo_cls.return_value = mock_evo_instance

        with patch("embpy.models.dna_models._HAVE_EVO", True), patch("embpy.models.dna_models.EvoModel", mock_evo_cls):
            with pytest.raises(ValueError) as excinfo:
                w.load(torch.device("cpu"))
            assert "out of range" in str(excinfo.value)
            assert "32 blocks" in str(excinfo.value)

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


class TestGENALMWrapper:

    def test_init_defaults(self):
        w = GENALMWrapper()
        assert "gena-lm" in w.model_name
        assert w.model_type == "dna"
        assert w.model is None
        assert w.tokenizer is None

    def test_init_custom_name(self):
        w = GENALMWrapper(model_path_or_name="AIRI-Institute/gena-lm-bert-large-t2t")
        assert w.model_name == "AIRI-Institute/gena-lm-bert-large-t2t"

    def test_available_pooling_strategies(self):
        assert set(GENALMWrapper.available_pooling_strategies) >= {"mean", "max", "cls"}

    def test_embed_without_load_raises(self):
        w = GENALMWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = GENALMWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = GENALMWrapper()
        w.model = MagicMock()
        w.tokenizer = _tok()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = GENALMWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_load_without_transformers_raises(self):
        w = GENALMWrapper()
        with patch("embpy.models.dna_models._HAVE_TRANSFORMERS", False):
            with pytest.raises(ImportError, match="transformers"):
                w.load(torch.device("cpu"))

    def _loaded(self, hidden_dim: int = 768, seq_len: int = 20) -> GENALMWrapper:
        w = GENALMWrapper()
        w.device = torch.device("cpu")
        w.tokenizer = _tok(seq_len)
        w.model = _dyn_model(hidden_dim)
        return w

    def test_embed_mean_returns_correct_shape(self):
        hidden_dim = 768
        result = self._loaded(hidden_dim).embed("ACGTACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_returns_correct_shape(self):
        hidden_dim = 768
        result = self._loaded(hidden_dim).embed("ACGT", pooling_strategy="cls")
        assert result.shape == (hidden_dim,)

    def test_embed_max_returns_correct_shape(self):
        hidden_dim = 768
        result = self._loaded(hidden_dim).embed("ACGT", pooling_strategy="max")
        assert result.shape == (hidden_dim,)

    def test_embed_uses_hidden_states_when_no_last_hidden_state(self):
        hidden_dim = 768
        seq_len = 20
        w = self._loaded(hidden_dim, seq_len)

        # last layer is all-ones so we can verify it was used (not zero layer)
        w.model = _dyn_model(hidden_dim, layer_fills=(0.0, 1.0))

        result = w.embed("ACGT", pooling_strategy="mean")
        assert result.shape == (hidden_dim,)
        assert np.allclose(result, 1.0, atol=1e-4), (
            "Should have used hidden_states[-1] (all-ones), not hidden_states[0] (zeros)"
        )

    def test_embed_target_layer(self):
        hidden_dim = 768
        seq_len = 20
        w = self._loaded(hidden_dim, seq_len)

        # Distinct constant per layer so we can assert layer 0 was the one used.
        w.model = _dyn_model(hidden_dim, layer_fills=(7.0, -3.0))

        result = w.embed("ACGT", target_layer=0)
        assert result.shape == (hidden_dim,)
        assert np.allclose(result, 7.0, atol=1e-4), (
            "target_layer=0 should select hidden_states[0], not the last layer"
        )

    def test_embed_batch_multiple(self):
        hidden_dim = 768
        w = self._loaded(hidden_dim)
        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        assert all(r.shape == (hidden_dim,) for r in results)

    def test_embed_output_is_float32(self):
        result = self._loaded().embed("ACGT")
        assert result.dtype == np.float32


class TestNucleotideTransformerWrapper:

    def test_init_defaults(self):
        w = NucleotideTransformerWrapper()
        assert "nucleotide-transformer" in w.model_name
        assert w.model_type == "dna"
        assert w.model is None

    def test_init_v2_name(self):
        w = NucleotideTransformerWrapper(
            model_path_or_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        )
        assert "v2-50m" in w.model_name

    def test_init_v1_name(self):
        w = NucleotideTransformerWrapper(
            model_path_or_name="InstaDeepAI/nucleotide-transformer-500m-human-ref"
        )
        assert "500m" in w.model_name

    def test_available_pooling_strategies(self):
        assert set(NucleotideTransformerWrapper.available_pooling_strategies) >= {"mean", "max", "cls"}

    def test_embed_without_load_raises(self):
        w = NucleotideTransformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = NucleotideTransformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = NucleotideTransformerWrapper()
        w.model = MagicMock()
        w.tokenizer = _tok()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = NucleotideTransformerWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="bad")

    def test_load_without_transformers_raises(self):
        w = NucleotideTransformerWrapper()
        with patch("embpy.models.dna_models._HAVE_TRANSFORMERS", False):
            with pytest.raises(ImportError, match="transformers"):
                w.load(torch.device("cpu"))

    def _loaded(self, hidden_dim: int = 512, seq_len: int = 20) -> NucleotideTransformerWrapper:
        w = NucleotideTransformerWrapper()
        w.device = torch.device("cpu")
        w.tokenizer = _tok(seq_len)
        w.model = _dyn_model(hidden_dim)
        return w

    def test_embed_mean_returns_correct_shape(self):
        hidden_dim = 512
        result = self._loaded(hidden_dim).embed("ACGTACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="cls")
        assert result.shape == (512,)

    def test_embed_max_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="max")
        assert result.shape == (512,)

    def test_embed_uses_hidden_states_for_mlm_output(self):
        """NT v2 EsmForMaskedLM returns MaskedLMOutput — must use hidden_states[-1]."""
        hidden_dim = 512
        seq_len = 20
        w = self._loaded(hidden_dim, seq_len)

        out = MagicMock()
        out.last_hidden_state = None
        out.hidden_states = (
            torch.zeros(1, seq_len, hidden_dim),
            torch.ones(1, seq_len, hidden_dim),
        )
        w.model = MagicMock(return_value=out)

        result = w.embed("ACGT", pooling_strategy="mean")
        assert result.shape == (hidden_dim,)
        assert np.allclose(result, 1.0, atol=1e-4), (
            "Should use hidden_states[-1] (ones), not hidden_states[0] (zeros)"
        )

    def test_embed_target_layer(self):
        hidden_dim = 512
        seq_len = 20
        w = self._loaded(hidden_dim, seq_len)

        layer0 = torch.randn(1, seq_len, hidden_dim)
        layer1 = torch.randn(1, seq_len, hidden_dim)
        out = MagicMock()
        out.hidden_states = (layer0, layer1)
        out.last_hidden_state = None
        w.model = MagicMock(return_value=out)

        result = w.embed("ACGT", target_layer=0)
        assert result.shape == (hidden_dim,)

    def test_embed_batch_multiple(self):
        w = self._loaded()
        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        assert all(r.shape == (512,) for r in results)

    def test_embed_output_is_float32(self):
        result = self._loaded().embed("ACGT")
        assert result.dtype == np.float32

    def test_mask_weighted_mean_excludes_padding(self):
        hidden_dim = 4
        seq_len = 6
        w = NucleotideTransformerWrapper()
        w.device = torch.device("cpu")

        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": mask,
        }
        w.tokenizer = tok

        h = torch.ones(1, seq_len, hidden_dim)
        h[0, 4:, :] = -9999.0
        out = MagicMock()
        out.hidden_states = (h, h)
        out.last_hidden_state = None
        w.model = MagicMock(return_value=out)

        result = w.embed("ACGT", pooling_strategy="mean")
        assert result.shape == (hidden_dim,)
        assert np.allclose(result, 1.0, atol=1e-4), (
            "Padding tokens should be masked out from mean pooling"
        )


class TestHyenaDNAWrapper:

    def test_init_defaults(self):
        w = HyenaDNAWrapper()
        assert "hyenadna" in w.model_name
        assert w.model_type == "dna"
        assert w.model is None

    def test_init_large_1m(self):
        w = HyenaDNAWrapper(model_path_or_name="LongSafari/hyenadna-large-1m-seqlen-hf")
        assert "1m" in w.model_name

    def test_available_pooling_strategies(self):
        # HyenaDNA adds "last" (natural for causal models)
        assert set(HyenaDNAWrapper.available_pooling_strategies) >= {"mean", "max", "cls", "last"}

    def test_embed_without_load_raises(self):
        w = HyenaDNAWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = HyenaDNAWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = HyenaDNAWrapper()
        w.model = MagicMock()
        w.tokenizer = _tok(has_mask=False)
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = HyenaDNAWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def test_load_without_transformers_raises(self):
        w = HyenaDNAWrapper()
        with patch("embpy.models.dna_models._HAVE_TRANSFORMERS", False):
            with pytest.raises(ImportError, match="transformers"):
                w.load(torch.device("cpu"))

    def _loaded(self, hidden_dim: int = 256, seq_len: int = 10) -> HyenaDNAWrapper:
        w = HyenaDNAWrapper()
        w.device = torch.device("cpu")

        w.tokenizer = _tok(seq_len, has_mask=False)
        w.model = _dyn_model(hidden_dim)
        return w

    def test_embed_mean_returns_correct_shape(self):
        hidden_dim = 256
        result = self._loaded(hidden_dim).embed("ACGTACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="cls")
        assert result.shape == (256,)

    def test_embed_max_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="max")
        assert result.shape == (256,)

    def test_embed_last_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="last")
        assert result.shape == (256,)

    def test_embed_target_layer(self):
        hidden_dim = 256
        seq_len = 10
        w = HyenaDNAWrapper()
        w.device = torch.device("cpu")
        w.tokenizer = _tok(seq_len, has_mask=False)

        layer0 = torch.randn(1, seq_len, hidden_dim)
        layer1 = torch.randn(1, seq_len, hidden_dim)
        out = MagicMock()
        out.hidden_states = (layer0, layer1)
        out.last_hidden_state = None
        w.model = MagicMock(return_value=out)

        result = w.embed("ACGT", target_layer=0)
        assert result.shape == (hidden_dim,)

    def test_embed_batch_multiple(self):
        w = self._loaded()
        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        assert all(r.shape == (256,) for r in results)

    def test_embed_output_is_float32(self):
        result = self._loaded().embed("ACGT")
        assert result.dtype == np.float32


class TestCaduceusWrapper:

    def test_init_defaults(self):
        w = CaduceusWrapper()
        assert "caduceus" in w.model_name
        assert w.model_type == "dna"
        assert w.model is None

    def test_init_ps_variant(self):
        w = CaduceusWrapper(
            model_path_or_name="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
        )
        assert "ps" in w.model_name

    def test_available_pooling_strategies(self):
        assert set(CaduceusWrapper.available_pooling_strategies) >= {"mean", "max", "cls"}

    def test_embed_without_load_raises(self):
        w = CaduceusWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = CaduceusWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_batch_empty_returns_empty(self):
        w = CaduceusWrapper()
        w.model = MagicMock()
        w.tokenizer = _tok()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []

    def test_invalid_pooling_raises(self):
        w = CaduceusWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="bad")

    def test_load_without_transformers_raises(self):
        w = CaduceusWrapper()
        with patch("embpy.models.dna_models._HAVE_TRANSFORMERS", False):
            with pytest.raises(ImportError, match="transformers"):
                w.load(torch.device("cpu"))

    def _loaded(self, hidden_dim: int = 256, seq_len: int = 20) -> CaduceusWrapper:
        w = CaduceusWrapper()
        w.device = torch.device("cpu")

        w.tokenizer = _tok(seq_len, has_mask=True)
        w.model = _dyn_model(hidden_dim)
        return w

    def test_embed_mean_returns_correct_shape(self):
        hidden_dim = 256
        result = self._loaded(hidden_dim).embed("ACGTACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_dim,)
        assert not np.isnan(result).any()

    def test_embed_cls_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="cls")
        assert result.shape == (256,)

    def test_embed_max_returns_correct_shape(self):
        result = self._loaded().embed("ACGT", pooling_strategy="max")
        assert result.shape == (256,)

    def test_embed_target_layer(self):
        hidden_dim = 256
        seq_len = 20
        w = CaduceusWrapper()
        w.device = torch.device("cpu")
        w.tokenizer = _tok(seq_len)

        layer0 = torch.randn(1, seq_len, hidden_dim)
        layer1 = torch.randn(1, seq_len, hidden_dim)
        out = MagicMock()
        out.hidden_states = (layer0, layer1)
        out.last_hidden_state = None
        w.model = MagicMock(return_value=out)

        result = w.embed("ACGT", target_layer=0)
        assert result.shape == (hidden_dim,)

    def test_embed_batch_multiple(self):
        w = self._loaded()
        results = w.embed_batch(["ACGT", "GCTA", "TTTT"])
        assert len(results) == 3
        assert all(r.shape == (256,) for r in results)

    def test_embed_output_is_float32(self):
        result = self._loaded().embed("ACGT")
        assert result.dtype == np.float32

    def test_ph_and_ps_both_constructable(self):
        ph = CaduceusWrapper("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
        ps = CaduceusWrapper("kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16")
        assert "ph" in ph.model_name
        assert "ps" in ps.model_name