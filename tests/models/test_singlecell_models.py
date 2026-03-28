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


# =====================================================================
# PCAEmbedding
# =====================================================================


class TestPCAEmbedding:
    def _make_adata_with_layers(self):
        """AnnData with counts and log_normalized layers."""
        import scipy.sparse as sp
        from anndata import AnnData as AD

        rng = np.random.default_rng(42)
        n, g = _N_CELLS, 200
        X = np.abs(rng.standard_normal((n, g))).astype(np.float32) * 100
        adata = AD(X=X)
        adata.layers["counts"] = X.copy()
        adata.layers["log_normalized"] = np.log1p(X)
        adata.var["highly_variable"] = np.array(
            [True] * 50 + [False] * (g - 50),
        )
        return adata

    def test_init_defaults(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = PCAEmbedding()
        assert wrapper.n_components == 50
        assert wrapper.use_hvg is True
        assert wrapper.model_type == "single_cell"

    def test_load_is_noop(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = PCAEmbedding()
        wrapper.load("cpu")
        assert wrapper.device == "cpu"

    def test_embed_cells_shape(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata_with_layers()
        wrapper = PCAEmbedding(n_components=10)
        wrapper.load("cpu")
        embs = wrapper.embed_cells(adata)
        assert embs.shape == (_N_CELLS, 10)
        assert embs.dtype == np.float32

    def test_embed_cells_with_hvg(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata_with_layers()
        wrapper = PCAEmbedding(n_components=10, use_hvg=True)
        wrapper.load("cpu")
        embs = wrapper.embed_cells(adata)
        assert embs.shape[0] == _N_CELLS

    def test_embed_cells_without_hvg(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata_with_layers()
        wrapper = PCAEmbedding(n_components=10, use_hvg=False)
        wrapper.load("cpu")
        embs = wrapper.embed_cells(adata)
        assert embs.shape[0] == _N_CELLS

    def test_embed_cells_from_x(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata_with_layers()
        wrapper = PCAEmbedding(n_components=5, layer=None)
        wrapper.load("cpu")
        embs = wrapper.embed_cells(adata)
        assert embs.shape == (_N_CELLS, 5)

    def test_embedding_dim_property(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = PCAEmbedding(n_components=30)
        assert wrapper.embedding_dim == 30

    def test_registry_entry(self) -> None:
        keys = list_singlecell_models()
        assert "pca" in keys

    def test_factory(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = get_singlecell_wrapper("pca")
        assert isinstance(wrapper, PCAEmbedding)


# =====================================================================
# ScVIToolsWrapper
# =====================================================================


class TestScVIToolsWrapper:
    def test_init_defaults(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper()
        assert wrapper.model_class_name == "SCVI"
        assert wrapper.n_latent == 30
        assert wrapper.model_type == "single_cell"

    def test_init_custom_params(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper(
            model_class="SCANVI",
            n_latent=20,
            n_layers=3,
            max_epochs=100,
        )
        assert wrapper.model_class_name == "SCANVI"
        assert wrapper.n_latent == 20
        assert wrapper.n_layers == 3

    def test_load_is_noop(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper()
        wrapper.load("cuda")
        assert wrapper.device == "cuda"

    def test_embedding_dim_property(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper(n_latent=15)
        assert wrapper.embedding_dim == 15

    def test_invalid_model_class(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper(model_class="INVALID")
        with pytest.raises((ValueError, ImportError)):
            wrapper.embed_cells(_make_fake_adata())

    def test_registry_entries(self) -> None:
        keys = list_singlecell_models()
        for k in ("scvi", "scanvi", "totalvi"):
            assert k in keys, f"'{k}' missing from registry"

    def test_factory_scvi(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = get_singlecell_wrapper("scvi")
        assert isinstance(wrapper, ScVIToolsWrapper)

    def test_factory_scanvi(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = get_singlecell_wrapper("scanvi")
        assert isinstance(wrapper, ScVIToolsWrapper)
