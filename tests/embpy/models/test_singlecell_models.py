"""Tests for embpy.models.singlecell_models – SingleCellWrapper & friends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embpy.models.singlecell_models import (
    _SC_MODEL_REGISTRY,
    Cell2SentenceWrapper,
    GeneformerWrapper,
    ScGPTWrapper,
    SCModelCard,
    SingleCellWrapper,
    StackWrapper,
    StateEmbeddingWrapper,
    TahoeWrapper,
    TranscriptFormerWrapper,
    UCEWrapper,
    get_singlecell_wrapper,
    list_singlecell_models,
    resolve_singlecell_preprocessing,
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
    model.get_embeddings.return_value = np.random.randn(_N_CELLS, emb_dim).astype(np.float32)
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
        for expected in ["scgpt", "uce", "tahoe_1b", "cell2sentence_2b", "state", "stack"]:
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

    def test_preprocessing_policy_metadata(self) -> None:
        pca = singlecell_info("pca")
        scgpt = singlecell_info("scgpt")

        assert pca.default_preprocessing == "standard"
        assert pca.input_layer == "log_normalized"
        assert pca.uses_hvg is True
        assert scgpt.default_preprocessing == "raw"
        assert scgpt.input_layer == "X"

    def test_resolve_singlecell_preprocessing_auto(self) -> None:
        raw_mode, raw_plan = resolve_singlecell_preprocessing(["scgpt"], "auto")
        mixed_mode, mixed_plan = resolve_singlecell_preprocessing(["scgpt", "pca"], "auto")

        assert raw_mode == "raw"
        assert mixed_mode == "standard"
        assert raw_plan["model_requirements"]["scgpt"]["input_layer"] == "X"
        assert mixed_plan["model_requirements"]["pca"]["uses_hvg"] is True


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

        with patch("embpy.models.singlecell_models.ScGPTWrapper.load"):
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


# =====================================================================
# StateEmbeddingWrapper
# =====================================================================


class TestStateEmbeddingWrapper:
    def test_init_defaults(self) -> None:
        wrapper = StateEmbeddingWrapper()
        assert wrapper.model_name == "state"
        assert wrapper._checkpoint is None
        assert wrapper._model_folder is None
        assert wrapper._inferer is None

    def test_init_with_checkpoint(self) -> None:
        wrapper = StateEmbeddingWrapper(
            checkpoint="/path/to/ckpt.ckpt",
            model_folder="/path/to/folder",
        )
        assert wrapper._checkpoint == "/path/to/ckpt.ckpt"
        assert wrapper._model_folder == "/path/to/folder"

    def test_embed_before_load_raises(self) -> None:
        wrapper = StateEmbeddingWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())

    def test_load_missing_package_raises(self) -> None:
        wrapper = StateEmbeddingWrapper(checkpoint="/fake.ckpt")
        with patch.dict("sys.modules", {"state": None, "state.emb": None}):
            with pytest.raises(ImportError, match="arc-state"):
                wrapper.load("cpu")

    def test_load_no_checkpoint_raises(self) -> None:
        wrapper = StateEmbeddingWrapper()
        with patch("embpy.models.singlecell_models.StateEmbeddingWrapper.load") as mock_load:
            mock_load.side_effect = ValueError("Either checkpoint or model_folder")
            with pytest.raises(ValueError, match="checkpoint or model_folder"):
                wrapper.load("cpu")

    @patch("embpy.models.singlecell_models.StateEmbeddingWrapper.load")
    def test_embed_with_mock_inferer(self, mock_load: MagicMock) -> None:
        wrapper = StateEmbeddingWrapper(checkpoint="/fake.ckpt")
        mock_inferer = MagicMock()
        fake_embs = np.random.randn(_N_CELLS, _FAKE_EMB_DIM).astype(np.float32)
        mock_inferer.encode_adata.return_value = fake_embs
        wrapper._inferer = mock_inferer
        wrapper._model = MagicMock()

        adata = _make_fake_adata()
        adata.write_h5ad = MagicMock()

        with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
            mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/fake")
            mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
            embs = wrapper.embed_cells(adata)

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)
        assert embs.dtype == np.float32
        mock_inferer.encode_adata.assert_called_once()

    def test_registry_entry(self) -> None:
        keys = list_singlecell_models()
        assert "state" in keys

    def test_registry_card(self) -> None:
        card = singlecell_info("state")
        assert card.wrapper_class_name == "StateEmbeddingWrapper"
        assert "Arc Institute" in card.description

    def test_factory(self) -> None:
        wrapper = get_singlecell_wrapper("state")
        assert isinstance(wrapper, StateEmbeddingWrapper)

    def test_repr(self) -> None:
        wrapper = StateEmbeddingWrapper()
        r = repr(wrapper)
        assert "StateEmbeddingWrapper" in r


# =====================================================================
# StackWrapper
# =====================================================================


class TestStackWrapper:
    def test_init_defaults(self) -> None:
        wrapper = StackWrapper()
        assert wrapper.model_name == "stack"
        assert wrapper._checkpoint is None
        assert wrapper._genelist is None

    def test_init_with_paths(self) -> None:
        wrapper = StackWrapper(
            checkpoint="/path/to/stack.ckpt",
            genelist="/path/to/genes.pkl",
            gene_name_col="gene_symbols",
        )
        assert wrapper._checkpoint == "/path/to/stack.ckpt"
        assert wrapper._genelist == "/path/to/genes.pkl"
        assert wrapper._gene_name_col == "gene_symbols"

    def test_embed_before_load_raises(self) -> None:
        wrapper = StackWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.embed_cells(_make_fake_adata())

    def test_load_missing_package_raises(self) -> None:
        wrapper = StackWrapper(checkpoint="/fake.ckpt", genelist="/fake.pkl")
        with patch.dict("sys.modules", {"stack": None, "stack.cli": None, "stack.cli.embedding": None}):
            with pytest.raises(ImportError, match="arc-stack"):
                wrapper.load("cpu")

    def test_load_missing_checkpoint_raises(self) -> None:
        wrapper = StackWrapper(genelist="/fake.pkl")
        with pytest.raises((ValueError, ImportError)):
            wrapper.load("cpu")

    def test_load_missing_genelist_raises(self) -> None:
        wrapper = StackWrapper(checkpoint="/fake.ckpt")
        with pytest.raises((ValueError, ImportError)):
            wrapper.load("cpu")

    @patch("embpy.models.singlecell_models.StackWrapper.load")
    def test_embed_with_mock(self, mock_load: MagicMock) -> None:
        wrapper = StackWrapper(checkpoint="/fake.ckpt", genelist="/fake.pkl")
        wrapper._model = MagicMock()

        fake_embs = np.random.randn(_N_CELLS, _FAKE_EMB_DIM).astype(np.float32)

        adata = _make_fake_adata()
        adata.write_h5ad = MagicMock()

        with patch(
            "embpy.models.singlecell_models.StackWrapper.embed_cells",
            return_value=fake_embs,
        ):
            embs = wrapper.embed_cells(adata)

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (_N_CELLS, _FAKE_EMB_DIM)
        assert embs.dtype == np.float32

    def test_registry_entry(self) -> None:
        keys = list_singlecell_models()
        assert "stack" in keys

    def test_registry_card(self) -> None:
        card = singlecell_info("stack")
        assert card.wrapper_class_name == "StackWrapper"
        assert "Arc Institute" in card.description

    def test_factory(self) -> None:
        wrapper = get_singlecell_wrapper("stack")
        assert isinstance(wrapper, StackWrapper)

    def test_factory_passes_kwargs(self) -> None:
        wrapper = get_singlecell_wrapper(
            "stack",
            checkpoint="/path/to/ckpt",
            genelist="/path/to/genes.pkl",
        )
        assert isinstance(wrapper, StackWrapper)
        assert wrapper._checkpoint == "/path/to/ckpt"
        assert wrapper._genelist == "/path/to/genes.pkl"

    def test_repr(self) -> None:
        wrapper = StackWrapper()
        r = repr(wrapper)
        assert "StackWrapper" in r


# =====================================================================
# Encoder-decoder capability flags
# =====================================================================


class TestCapabilityFlags:
    """Verify the SCModelCard / wrapper-class decode+generation metadata
    stays in sync with the actual method implementations.
    """

    def test_card_flags_encoder_decoders(self) -> None:
        for key in ("pca", "scvi", "scanvi", "totalvi", "state"):
            card = singlecell_info(key)
            assert card.supports_decode is True, f"{key} should advertise decode"
            assert card.supports_generation is False, f"{key} should NOT advertise generation"

    def test_card_flags_stack(self) -> None:
        card = singlecell_info("stack")
        # Stack has in-context generation, not a pure latent decoder.
        assert card.supports_generation is True
        assert card.supports_decode is False

    def test_card_flags_encoder_only(self) -> None:
        for key in (
            "scgpt",
            "geneformer_v2_12L",
            "uce",
            "transcriptformer_metazoa",
            "tahoe_70m",
            "cell2sentence_2b",
        ):
            card = singlecell_info(key)
            assert card.supports_decode is False, f"{key} should NOT decode"
            assert card.supports_generation is False, f"{key} should NOT generate"

    def test_card_and_wrapper_flags_agree(self) -> None:
        """SCModelCard flags should match wrapper class flags for every
        registered model.
        """
        from embpy.models.singlecell_models import _WRAPPER_MAP

        for key, card in _SC_MODEL_REGISTRY.items():
            wrapper_cls = _WRAPPER_MAP[card.wrapper_class_name]
            assert wrapper_cls.supports_decode == card.supports_decode, (
                f"{key}: wrapper.supports_decode={wrapper_cls.supports_decode} "
                f"but card.supports_decode={card.supports_decode}"
            )
            assert wrapper_cls.supports_generation == card.supports_generation, (
                f"{key}: wrapper.supports_generation="
                f"{wrapper_cls.supports_generation} but card="
                f"{card.supports_generation}"
            )


# =====================================================================
# Base-class decode_cells / generate_cells stubs
# =====================================================================


class TestBaseStubs:
    """Wrappers that do not override decode_cells / generate_cells must
    raise NotImplementedError with an informative message.
    """

    def test_encoder_only_wrapper_decode_raises(self) -> None:
        """scGPT is an encoder-only foundation model: decode_cells must
        refuse with a clear message."""
        wrapper = ScGPTWrapper()
        with pytest.raises(NotImplementedError, match="decode_cells"):
            wrapper.decode_cells(np.zeros((2, 4), dtype=np.float32))

    def test_encoder_only_wrapper_generate_raises(self) -> None:
        wrapper = UCEWrapper()
        with pytest.raises(NotImplementedError, match="generate_cells"):
            wrapper.generate_cells(
                _make_fake_adata(),
                _make_fake_adata(),
            )

    def test_pca_does_not_generate(self) -> None:
        """Encoder-decoder != in-context generator; PCA.generate must
        still refuse.
        """
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = PCAEmbedding()
        with pytest.raises(NotImplementedError, match="generate_cells"):
            wrapper.generate_cells(
                _make_fake_adata(),
                _make_fake_adata(),
            )

    def test_stack_does_not_decode(self) -> None:
        """Stack exposes generate_cells, not a pure decode_cells."""
        wrapper = StackWrapper()
        with pytest.raises(NotImplementedError, match="decode_cells"):
            wrapper.decode_cells(np.zeros((2, 4), dtype=np.float32))


# =====================================================================
# PCAEmbedding decode round-trip
# =====================================================================


class TestPCADecode:
    def _make_adata(self, n_cells: int = _N_CELLS, n_genes: int = 50):
        from anndata import AnnData as AD

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        adata = AD(X=X)
        adata.layers["log_normalized"] = X.copy()
        # First 60% of genes are HVG.
        hvg_cut = int(n_genes * 0.6)
        adata.var["highly_variable"] = np.array(
            [True] * hvg_cut + [False] * (n_genes - hvg_cut),
        )
        return adata

    def test_decode_before_embed_raises(self) -> None:
        from embpy.models.singlecell_models import PCAEmbedding

        wrapper = PCAEmbedding(n_components=5)
        wrapper.load()
        with pytest.raises(RuntimeError, match="embed_cells"):
            wrapper.decode_cells(np.zeros((3, 5), dtype=np.float32))

    def test_roundtrip_shape_full_gene_panel(self) -> None:
        """Decoded matrix should have n_genes columns regardless of HVG
        restriction.
        """
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata(n_genes=50)
        wrapper = PCAEmbedding(n_components=5, use_hvg=True, backend="cpu")
        wrapper.load()
        z = wrapper.embed_cells(adata)
        assert z.shape == (_N_CELLS, 5)

        recon = wrapper.decode_cells(z)
        assert recon.shape == (_N_CELLS, 50)
        assert recon.dtype == np.float32

    def test_roundtrip_non_hvg_columns_are_zero(self) -> None:
        """When use_hvg=True, decode should zero-pad non-HVG columns."""
        from embpy.models.singlecell_models import PCAEmbedding

        n_genes = 40
        adata = self._make_adata(n_genes=n_genes)
        wrapper = PCAEmbedding(n_components=5, use_hvg=True, backend="cpu")
        wrapper.load()
        z = wrapper.embed_cells(adata)
        recon = wrapper.decode_cells(z)

        hvg_mask = adata.var["highly_variable"].values.astype(bool)
        non_hvg = ~hvg_mask
        assert np.allclose(recon[:, non_hvg], 0.0)

    def test_roundtrip_no_hvg_full_reconstruction(self) -> None:
        """Without HVG restriction, the full gene matrix participates in
        both fit and inverse; with 5 components on 50 genes, the recon
        should be close but not exact. We only check shape + finiteness
        here.
        """
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata(n_genes=20)
        wrapper = PCAEmbedding(n_components=5, use_hvg=False, backend="cpu")
        wrapper.load()
        z = wrapper.embed_cells(adata)
        recon = wrapper.decode_cells(z)

        assert recon.shape == (_N_CELLS, 20)
        assert np.all(np.isfinite(recon))

    def test_decode_caches_ignore_unused_args(self) -> None:
        """PCA decode ignores gene_names/adata/kwargs cleanly."""
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata(n_genes=30)
        wrapper = PCAEmbedding(n_components=3, use_hvg=False, backend="cpu")
        wrapper.load()
        z = wrapper.embed_cells(adata)

        recon_a = wrapper.decode_cells(z)
        recon_b = wrapper.decode_cells(
            z,
            gene_names=["ignored"] * 30,
            adata=None,
            extra="ignored",
        )
        np.testing.assert_allclose(recon_a, recon_b)

    def test_reconstruction_approaches_identity_when_rank_matches(self) -> None:
        """If n_components equals n_features and use_hvg=False and scale=
        False, PCA + inverse_transform is a near-identity operation.
        """
        from embpy.models.singlecell_models import PCAEmbedding

        adata = self._make_adata(n_genes=10)
        wrapper = PCAEmbedding(
            n_components=10,
            use_hvg=False,
            scale=False,
            backend="cpu",
        )
        wrapper.load()
        z = wrapper.embed_cells(adata)
        recon = wrapper.decode_cells(z)

        original = np.asarray(adata.layers["log_normalized"], dtype=np.float32)
        np.testing.assert_allclose(recon, original, atol=1e-4)


# =====================================================================
# ScVIToolsWrapper decode
# =====================================================================


class TestScVIDecode:
    def test_decode_before_train_raises(self) -> None:
        from embpy.models.singlecell_models import ScVIToolsWrapper

        wrapper = ScVIToolsWrapper()
        with pytest.raises(RuntimeError, match="embed_cells"):
            wrapper.decode_cells(np.zeros((3, 30), dtype=np.float32))

    def test_decode_routes_through_generative_module(self) -> None:
        """Simulate a trained scvi model so decode_cells exercises the
        tensor-packing path and the ``px_rate`` unwrapping."""
        import torch

        from embpy.models.singlecell_models import ScVIToolsWrapper

        n_cells, n_latent, n_genes = 4, 8, 12

        fake_module = MagicMock()
        # Track what generative was called with.
        fake_rate = torch.ones(n_cells, n_genes, dtype=torch.float32)
        fake_module.generative = MagicMock(
            return_value={"px_rate": fake_rate},
        )
        # Provide a param iterable so next(parameters()) works.
        dummy_param = torch.zeros(1)
        fake_module.parameters = MagicMock(return_value=iter([dummy_param]))

        fake_model = MagicMock()
        fake_model.module = fake_module

        wrapper = ScVIToolsWrapper(n_latent=n_latent)
        wrapper._trained_model = fake_model
        wrapper._trained_adata = MagicMock()

        latent = (
            np.random.default_rng(0)
            .standard_normal(
                (n_cells, n_latent),
            )
            .astype(np.float32)
        )
        out = wrapper.decode_cells(latent)

        assert out.shape == (n_cells, n_genes)
        assert out.dtype == np.float32
        # Generative must have been called with z, library, batch_index.
        fake_module.generative.assert_called_once()
        call_kwargs = fake_module.generative.call_args.kwargs
        assert "z" in call_kwargs
        assert "library" in call_kwargs
        assert "batch_index" in call_kwargs
        assert call_kwargs["z"].shape == (n_cells, n_latent)
        assert call_kwargs["library"].shape == (n_cells, 1)
        assert call_kwargs["batch_index"].shape == (n_cells, 1)

    def test_decode_scanvi_adds_y_label(self) -> None:
        """scANVI's generative takes an extra ``y`` argument."""
        import torch

        from embpy.models.singlecell_models import ScVIToolsWrapper

        n_cells, n_latent, n_genes = 3, 6, 10
        fake_module = MagicMock()
        fake_module.generative = MagicMock(
            return_value={"px_rate": torch.zeros(n_cells, n_genes)},
        )
        fake_module.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        fake_model = MagicMock()
        fake_model.module = fake_module

        wrapper = ScVIToolsWrapper(model_class="SCANVI", n_latent=n_latent)
        wrapper._trained_model = fake_model

        wrapper.decode_cells(np.zeros((n_cells, n_latent), dtype=np.float32))
        kwargs = fake_module.generative.call_args.kwargs
        assert "y" in kwargs
        assert kwargs["y"].shape == (n_cells, 1)

    def test_decode_unwraps_distribution(self) -> None:
        """When generative returns a torch Distribution under 'px', decode
        should read its ``.mean``."""
        import torch

        from embpy.models.singlecell_models import ScVIToolsWrapper

        class _FakeDist:
            def __init__(self, mean: torch.Tensor) -> None:
                self.mean = mean

        n_cells, n_latent, n_genes = 2, 4, 5
        fake_mean = torch.full((n_cells, n_genes), 3.14)
        fake_module = MagicMock()
        fake_module.generative = MagicMock(
            return_value={"px": _FakeDist(fake_mean)},
        )
        fake_module.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        fake_model = MagicMock()
        fake_model.module = fake_module

        wrapper = ScVIToolsWrapper(n_latent=n_latent)
        wrapper._trained_model = fake_model

        out = wrapper.decode_cells(np.zeros((n_cells, n_latent), dtype=np.float32))
        np.testing.assert_allclose(out, 3.14, atol=1e-5)

    def test_decode_missing_output_key_raises(self) -> None:
        """If neither px_rate nor px is present, decode_cells must raise
        a clear error."""
        import torch

        from embpy.models.singlecell_models import ScVIToolsWrapper

        fake_module = MagicMock()
        fake_module.generative = MagicMock(return_value={"unexpected": 1})
        fake_module.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        fake_model = MagicMock()
        fake_model.module = fake_module

        wrapper = ScVIToolsWrapper()
        wrapper._trained_model = fake_model

        with pytest.raises(RuntimeError, match="px_rate"):
            wrapper.decode_cells(np.zeros((2, 30), dtype=np.float32))


# =====================================================================
# StateEmbeddingWrapper decode
# =====================================================================


class TestStateDecode:
    def test_decode_before_load_raises(self) -> None:
        wrapper = StateEmbeddingWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.decode_cells(
                np.zeros((2, 8), dtype=np.float32),
                gene_names=["G1", "G2", "G3"],
            )

    def test_decode_requires_gene_names(self) -> None:
        """STATE's decoder is gene-parametric: gene_names is required."""
        wrapper = StateEmbeddingWrapper()
        wrapper._inferer = MagicMock()
        with pytest.raises(ValueError, match="gene_names"):
            wrapper.decode_cells(np.zeros((2, 8), dtype=np.float32))

    def test_decode_routes_through_inferer(self) -> None:
        """Mock state.emb.Inference.decode_from_adata and check the
        wrapper assembles the scaffold + concatenates batches correctly.
        """
        wrapper = StateEmbeddingWrapper()
        n_cells, emb_dim, n_genes = 5, 16, 7
        gene_names = [f"G{i}" for i in range(n_genes)]

        fake_batches = [
            np.ones((3, n_genes), dtype=np.float32),
            np.full((2, n_genes), 2.0, dtype=np.float32),
        ]

        def fake_decode(adata, genes, emb_key, read_depth, batch_size):
            # Check the wrapper produced the right scaffold.
            assert list(adata.var_names) == gene_names
            assert list(genes) == gene_names
            assert emb_key in adata.obsm
            assert adata.obsm[emb_key].shape == (n_cells, emb_dim)
            yield from fake_batches

        mock_inferer = MagicMock()
        mock_inferer.decode_from_adata = fake_decode
        wrapper._inferer = mock_inferer

        z = (
            np.random.default_rng(0)
            .standard_normal(
                (n_cells, emb_dim),
            )
            .astype(np.float32)
        )
        out = wrapper.decode_cells(z, gene_names=gene_names)

        assert out.shape == (n_cells, n_genes)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[:3], 1.0)
        np.testing.assert_allclose(out[3:], 2.0)

    def test_decode_uses_adata_var_names_fallback(self) -> None:
        """If gene_names is None but adata is passed, its var_names win."""
        from anndata import AnnData as AD

        wrapper = StateEmbeddingWrapper()

        captured: dict = {}

        def fake_decode(adata_arg, genes, emb_key, read_depth, batch_size):
            captured["genes"] = list(genes)
            yield np.ones((2, len(genes)), dtype=np.float32)

        mock_inferer = MagicMock()
        mock_inferer.decode_from_adata = fake_decode
        wrapper._inferer = mock_inferer

        adata = AD(X=np.zeros((2, 3), dtype=np.float32))
        adata.var_names = ["A", "B", "C"]

        wrapper.decode_cells(
            np.zeros((2, 8), dtype=np.float32),
            gene_names=None,
            adata=adata,
        )
        assert captured["genes"] == ["A", "B", "C"]

    def test_decode_handles_trailing_singleton_dim(self) -> None:
        """Some checkpoints emit batches with a trailing singleton; the
        wrapper should squeeze that."""
        wrapper = StateEmbeddingWrapper()
        n_cells, n_genes = 4, 5

        def fake_decode(*args, **kwargs):
            yield np.ones((n_cells, n_genes, 1), dtype=np.float32)

        mock_inferer = MagicMock()
        mock_inferer.decode_from_adata = fake_decode
        wrapper._inferer = mock_inferer

        out = wrapper.decode_cells(
            np.zeros((n_cells, 4), dtype=np.float32),
            gene_names=[f"G{i}" for i in range(n_genes)],
        )
        assert out.shape == (n_cells, n_genes)


# =====================================================================
# StackWrapper generate_cells
# =====================================================================


class TestStackGenerate:
    def test_generate_requires_checkpoint_and_genelist(self) -> None:
        wrapper = StackWrapper()
        with pytest.raises(ValueError, match="checkpoint"):
            wrapper.generate_cells(_make_fake_adata(), _make_fake_adata())

    def test_generate_import_error_propagates(self) -> None:
        wrapper = StackWrapper(checkpoint="/fake.ckpt", genelist="/fake.pkl")
        with patch.dict(
            "sys.modules",
            {
                "stack": None,
                "stack.cli": None,
                "stack.cli.generation": None,
            },
        ):
            with pytest.raises(ImportError, match="arc-stack"):
                wrapper.generate_cells(
                    _make_fake_adata(),
                    _make_fake_adata(),
                )

    def test_generate_routes_through_generate(self) -> None:
        """Mock stack.cli.generation.generate and check the wrapper
        concatenates the per-split predictions into a single array.
        """
        import types

        from anndata import AnnData as AD

        wrapper = StackWrapper(
            checkpoint="/fake.ckpt",
            genelist="/fake.pkl",
        )
        wrapper.device = "cpu"

        n_test, n_genes = 6, 4
        pred_adata = AD(X=np.ones((n_test, n_genes), dtype=np.float32))
        fake_generate = MagicMock(return_value={"donor_A": pred_adata})

        # Build a stack.cli.generation module tree in sys.modules.
        stack_mod = types.ModuleType("stack")
        cli_mod = types.ModuleType("stack.cli")
        gen_mod = types.ModuleType("stack.cli.generation")
        gen_mod.generate = fake_generate
        stack_mod.cli = cli_mod
        cli_mod.generation = gen_mod

        base = AD(X=np.zeros((8, 20), dtype=np.float32))
        base.obs["donor"] = ["donor_A"] * 8
        test = AD(X=np.zeros((n_test, 20), dtype=np.float32))

        with patch.dict(
            "sys.modules",
            {
                "stack": stack_mod,
                "stack.cli": cli_mod,
                "stack.cli.generation": gen_mod,
            },
        ):
            out = wrapper.generate_cells(
                base,
                test,
                split_column="donor",
                split_values=["donor_A"],
            )

        assert out.shape == (n_test, n_genes)
        assert out.dtype == np.float32
        # generate() must have received our split config + forwarded kwargs.
        kwargs = fake_generate.call_args.kwargs
        assert kwargs["split_column"] == "donor"
        assert kwargs["split_values"] == ["donor_A"]
        assert kwargs["genelist_path"] == "/fake.pkl"
        assert kwargs["checkpoint_path"] == "/fake.ckpt"

    def test_generate_auto_single_donor_shortcut(self) -> None:
        """When split_column is None the wrapper injects a placeholder
        split column on the base adata and calls generate() with it.
        """
        import types

        from anndata import AnnData as AD

        wrapper = StackWrapper(
            checkpoint="/fake.ckpt",
            genelist="/fake.pkl",
        )

        n_test, n_genes = 3, 5
        pred_adata = AD(X=np.ones((n_test, n_genes), dtype=np.float32))
        fake_generate = MagicMock(return_value={"ALL": pred_adata})

        stack_mod = types.ModuleType("stack")
        cli_mod = types.ModuleType("stack.cli")
        gen_mod = types.ModuleType("stack.cli.generation")
        gen_mod.generate = fake_generate
        stack_mod.cli = cli_mod
        cli_mod.generation = gen_mod

        base = AD(X=np.zeros((4, n_genes), dtype=np.float32))
        test = AD(X=np.zeros((n_test, n_genes), dtype=np.float32))

        with patch.dict(
            "sys.modules",
            {
                "stack": stack_mod,
                "stack.cli": cli_mod,
                "stack.cli.generation": gen_mod,
            },
        ):
            out = wrapper.generate_cells(base, test)

        assert out.shape == (n_test, n_genes)
        kwargs = fake_generate.call_args.kwargs
        assert kwargs["split_column"] == "__embpy_split__"
        assert kwargs["split_values"] == ["ALL"]


# =====================================================================
# BioEmbedder.decode_cells / generate_cells public API
# =====================================================================


class TestBioEmbedderDecode:
    def _fresh_embedder(self):
        from embpy.embedder import BioEmbedder

        e = BioEmbedder(device="cpu")
        return e

    def _make_processed_adata(self):
        """Small AnnData that BioEmbedder.embed_cells(models=['pca']) can
        consume without running the full preprocessing pipeline (we pass
        preprocessing='none' to skip it)."""
        from anndata import AnnData as AD

        rng = np.random.default_rng(42)
        n, g = 12, 30
        X = rng.standard_normal((n, g)).astype(np.float32)
        adata = AD(X=X)
        adata.layers["log_normalized"] = X.copy()
        adata.layers["counts"] = np.abs(X).astype(np.float32)
        adata.var["highly_variable"] = np.array(
            [True] * 20 + [False] * 10,
        )
        return adata

    def test_decode_unknown_model_raises(self) -> None:
        e = self._fresh_embedder()
        with pytest.raises(ValueError, match="Unknown single-cell model"):
            e.decode_cells(
                latent=np.zeros((2, 3), dtype=np.float32),
                model="nonexistent_xyz",
            )

    def test_decode_non_decoder_model_raises(self) -> None:
        """scGPT is an encoder-only model; decode_cells must refuse."""
        e = self._fresh_embedder()
        with pytest.raises(ValueError, match="does not expose a decoder"):
            e.decode_cells(
                latent=np.zeros((2, 512), dtype=np.float32),
                model="scgpt",
            )

    def test_decode_without_latent_or_adata_raises(self) -> None:
        e = self._fresh_embedder()
        with pytest.raises(ValueError, match="latent"):
            e.decode_cells(model="pca")

    def test_decode_missing_obsm_key_raises(self) -> None:
        from anndata import AnnData as AD

        e = self._fresh_embedder()
        adata = AD(X=np.zeros((4, 10), dtype=np.float32))
        with pytest.raises(KeyError, match="X_pca"):
            e.decode_cells(adata=adata, model="pca")

    def test_decode_no_cached_wrapper_raises(self) -> None:
        """Calling decode_cells before embed_cells should raise a clear
        RuntimeError pointing the user to the embed call.
        """
        e = self._fresh_embedder()
        with pytest.raises(RuntimeError, match="embed_cells"):
            e.decode_cells(
                latent=np.zeros((2, 5), dtype=np.float32),
                model="pca",
            )

    def test_embed_then_decode_pca_roundtrip(self) -> None:
        """After embed_cells(models=['pca']), decode_cells should work on
        the cached wrapper and produce a (n_cells, n_genes) matrix."""
        e = self._fresh_embedder()
        adata = self._make_processed_adata()

        out_adata = e.embed_cells(
            adata,
            models=["pca"],
            preprocessing="none",
            n_pca_components=5,
            copy=True,
        )
        assert "X_pca" in out_adata.obsm

        decoded = e.decode_cells(adata=out_adata, model="pca")
        assert decoded.shape == (out_adata.n_obs, out_adata.n_vars)

    def test_decode_writes_to_layer(self) -> None:
        """write_layer kwarg should persist the decoded matrix to
        adata.layers[...]."""
        e = self._fresh_embedder()
        adata = self._make_processed_adata()

        out_adata = e.embed_cells(
            adata,
            models=["pca"],
            preprocessing="none",
            n_pca_components=5,
            copy=True,
        )
        e.decode_cells(
            adata=out_adata,
            model="pca",
            write_layer="X_pca_decoded",
        )
        assert "X_pca_decoded" in out_adata.layers
        assert out_adata.layers["X_pca_decoded"].shape == (
            out_adata.n_obs,
            out_adata.n_vars,
        )

    def test_generate_non_generator_model_raises(self) -> None:
        e = self._fresh_embedder()
        with pytest.raises(ValueError, match="does not expose"):
            e.generate_cells(
                _make_fake_adata(),
                _make_fake_adata(),
                model="pca",
            )

    def test_generate_without_wrapper_raises(self) -> None:
        """Stack requires a manually-constructed wrapper (checkpoint +
        genelist) because embed_cells cannot thread those through.
        """
        e = self._fresh_embedder()
        with pytest.raises(RuntimeError, match="wrapper"):
            e.generate_cells(
                _make_fake_adata(),
                _make_fake_adata(),
                model="stack",
            )

    def test_generate_uses_user_wrapper(self) -> None:
        """If the user passes their own wrapper, generate_cells should
        call its generate_cells and return the result."""
        e = self._fresh_embedder()
        fake_wrapper = MagicMock(spec=StackWrapper)
        expected = np.ones((4, 3), dtype=np.float32)
        fake_wrapper.generate_cells.return_value = expected

        out = e.generate_cells(
            _make_fake_adata(),
            _make_fake_adata(),
            model="stack",
            wrapper=fake_wrapper,
            split_column="donor",
        )
        np.testing.assert_array_equal(out, expected)
        fake_wrapper.generate_cells.assert_called_once()
        kwargs = fake_wrapper.generate_cells.call_args.kwargs
        assert kwargs["split_column"] == "donor"
