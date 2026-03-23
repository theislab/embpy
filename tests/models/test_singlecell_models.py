"""Tests for embpy.models.singlecell_models – SingleCellWrapper & friends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embpy.models.singlecell_models import (
    SCModelCard,
    ScGPTWrapper,
    GeneformerWrapper,
    UCEWrapper,
    TranscriptFormerWrapper,
    TahoeWrapper,
    Cell2SentenceWrapper,
    SingleCellWrapper,
    _SC_MODEL_REGISTRY,
    get_singlecell_wrapper,
    list_singlecell_models,
    singlecell_info,
)


# =====================================================================
# Helpers
# =====================================================================

_FAKE_EMB_DIM = 64
_N_CELLS = 10


def _make_fake_adata():
    """Minimal object that quacks like AnnData for mocking."""
    adata = MagicMock()
    adata.shape = (_N_CELLS, 2000)
    adata.n_obs = _N_CELLS
    return adata


def _make_mock_helical_model(emb_dim: int = _FAKE_EMB_DIM):
    """Return a mock helical model with process_data / get_embeddings."""
    model = MagicMock()
    fake_dataset = MagicMock()
    model.process_data.return_value = fake_dataset
    model.get_embeddings.return_value = np.random.randn(_N_CELLS, emb_dim).astype(
        np.float32
    )
    return model


# =====================================================================
# Registry / discovery
# =====================================================================


class TestRegistry:
    def test_list_models_returns_list(self) -> None:
        keys = list_singlecell_models()
        assert isinstance(keys, list)
        assert len(keys) > 0

    def test_known_keys_present(self) -> None:
        keys = list_singlecell_models()
        for expected in ["scgpt", "uce", "tahoe_1b", "cell2sentence_2b"]:
            assert expected in keys

    def test_singlecell_info_valid(self) -> None:
        card = singlecell_info("scgpt")
        assert isinstance(card, SCModelCard)
        assert card.key == "scgpt"

    def test_singlecell_info_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown single-cell model"):
            singlecell_info("nonexistent_model")

    def test_all_cards_have_description(self) -> None:
        for key in list_singlecell_models():
            card = singlecell_info(key)
            assert card.description, f"Missing description for {key}"

    def test_all_cards_have_wrapper_class(self) -> None:
        from embpy.models.singlecell_models import _WRAPPER_MAP

        for key, card in _SC_MODEL_REGISTRY.items():
            assert card.wrapper_class_name in _WRAPPER_MAP, (
                f"Card {key} references unknown wrapper {card.wrapper_class_name}"
            )


# =====================================================================
# Factory
# =====================================================================


class TestFactory:
    def test_get_wrapper_returns_correct_type(self) -> None:
        wrapper = get_singlecell_wrapper("scgpt")
        assert isinstance(wrapper, ScGPTWrapper)

    def test_get_wrapper_geneformer(self) -> None:
        wrapper = get_singlecell_wrapper("geneformer_v2_12L")
        assert isinstance(wrapper, GeneformerWrapper)

    def test_get_wrapper_uce(self) -> None:
        wrapper = get_singlecell_wrapper("uce")
        assert isinstance(wrapper, UCEWrapper)

    def test_get_wrapper_transcriptformer(self) -> None:
        wrapper = get_singlecell_wrapper("transcriptformer_metazoa")
        assert isinstance(wrapper, TranscriptFormerWrapper)

    def test_get_wrapper_tahoe(self) -> None:
        wrapper = get_singlecell_wrapper("tahoe_1b")
        assert isinstance(wrapper, TahoeWrapper)

    def test_get_wrapper_cell2sentence(self) -> None:
        wrapper = get_singlecell_wrapper("cell2sentence_2b")
        assert isinstance(wrapper, Cell2SentenceWrapper)

    def test_factory_invalid_key(self) -> None:
        with pytest.raises(ValueError, match="Unknown single-cell model"):
            get_singlecell_wrapper("bad_model")

    def test_factory_passes_batch_size(self) -> None:
        wrapper = get_singlecell_wrapper("scgpt", batch_size=64)
        assert wrapper.batch_size == 64


# =====================================================================
# SingleCellWrapper ABC
# =====================================================================


class TestSingleCellWrapperABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SingleCellWrapper()  # type: ignore[abstract]

    def test_repr(self) -> None:
        wrapper = get_singlecell_wrapper("scgpt")
        r = repr(wrapper)
        assert "ScGPTWrapper" in r

    def test_embedding_dim_before_load(self) -> None:
        wrapper = get_singlecell_wrapper("scgpt")
        assert wrapper.embedding_dim == 0

    def test_model_type(self) -> None:
        wrapper = get_singlecell_wrapper("scgpt")
        assert wrapper.model_type == "single_cell"


# =====================================================================
# scGPT
# =====================================================================


class TestScGPT:
    @patch("embpy.models.singlecell_models._require_helical")
    def test_load_and_embed(self, mock_helical: MagicMock) -> None:
        mock_model = _make_mock_helical_model()

        with patch(
            "embpy.models.singlecell_models.ScGPTWrapper.load"
        ) as mock_load:
            wrapper = ScGPTWrapper(batch_size=5)
            wrapper._model = mock_model
            wrapper.device = "cpu"

            adata = _make_fake_adata()
            embs = wrapper.embed_cells(adata)

            assert isinstance(embs, np.ndarray)
            assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)
            assert embs.dtype == np.float32
            mock_model.process_data.assert_called_once_with(adata)
            mock_model.get_embeddings.assert_called_once()

    def test_embed_before_load_raises(self) -> None:
        wrapper = ScGPTWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())

    def test_embedding_dim_after_mock_load(self) -> None:
        wrapper = ScGPTWrapper()
        wrapper._model = MagicMock()
        assert wrapper.embedding_dim == 512


# =====================================================================
# Geneformer
# =====================================================================


class TestGeneformer:
    def test_embed_with_mock(self) -> None:
        wrapper = GeneformerWrapper(model_name="gf-12L-38M-i4096")
        wrapper._model = _make_mock_helical_model()
        wrapper.device = "cpu"

        embs = wrapper.embed_cells(_make_fake_adata())
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)

    def test_embed_before_load_raises(self) -> None:
        wrapper = GeneformerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())


# =====================================================================
# UCE
# =====================================================================


class TestUCE:
    def test_embed_with_mock(self) -> None:
        wrapper = UCEWrapper()
        wrapper._model = _make_mock_helical_model(1280)
        wrapper.device = "cpu"

        embs = wrapper.embed_cells(_make_fake_adata())
        assert embs.shape == (_N_CELLS, 1280)

    def test_embedding_dim(self) -> None:
        wrapper = UCEWrapper()
        wrapper._model = MagicMock()
        assert wrapper.embedding_dim == 1280

    def test_embed_before_load_raises(self) -> None:
        wrapper = UCEWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())


# =====================================================================
# TranscriptFormer
# =====================================================================


class TestTranscriptFormer:
    def test_embed_with_mock(self) -> None:
        wrapper = TranscriptFormerWrapper(model_name="TF-Metazoa")
        mock_model = _make_mock_helical_model()
        wrapper._model = mock_model
        wrapper.device = "cpu"

        adata = _make_fake_adata()
        embs = wrapper.embed_cells(adata)
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)
        mock_model.process_data.assert_called_once_with([adata])

    def test_embed_before_load_raises(self) -> None:
        wrapper = TranscriptFormerWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())


# =====================================================================
# Tahoe
# =====================================================================


class TestTahoe:
    def test_embed_with_mock(self) -> None:
        wrapper = TahoeWrapper(model_name="70m")
        wrapper._model = _make_mock_helical_model()
        wrapper.device = "cpu"

        embs = wrapper.embed_cells(_make_fake_adata())
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)

    def test_embed_before_load_raises(self) -> None:
        wrapper = TahoeWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())


# =====================================================================
# Cell2Sentence
# =====================================================================


class TestCell2Sentence:
    def test_embed_with_mock(self) -> None:
        wrapper = Cell2SentenceWrapper(model_name="c2s-scale-2b")
        wrapper._model = _make_mock_helical_model()
        wrapper.device = "cpu"

        embs = wrapper.embed_cells(_make_fake_adata())
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)

    def test_embed_before_load_raises(self) -> None:
        wrapper = Cell2SentenceWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())


# =====================================================================
# _require_helical
# =====================================================================


class TestRequireHelical:
    def test_import_error_message(self) -> None:
        from embpy.models.singlecell_models import _require_helical

        with patch.dict("sys.modules", {"helical": None}):
            with pytest.raises(ImportError, match="helical"):
                _require_helical()
