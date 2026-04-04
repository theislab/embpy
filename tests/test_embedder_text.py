"""Tests for BioEmbedder.embed_description and embed_descriptions_batch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbedDescription:
    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_calls_text_resolver_then_embed_text(self, mock_tr_cls, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        mock_tr = MagicMock()
        mock_tr.get_combined_description.return_value = "TP53 is a tumor suppressor."
        mock_tr_cls.return_value = mock_tr

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        mock_model = MagicMock()
        mock_model.model_type = "text"
        mock_model.embed.return_value = np.ones(384, dtype=np.float32)
        embedder.model_cache["minilm_l6_v2"] = mock_model

        emb = embedder.embed_description("TP53", model="minilm_l6_v2")

        mock_tr.get_combined_description.assert_called_once_with(
            "TP53", entity_type="auto", sources="all",
        )
        mock_model.embed.assert_called_once()
        assert emb.shape == (384,)

    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_entity_type_passed_through(self, mock_tr_cls, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        mock_tr = MagicMock()
        mock_tr.get_combined_description.return_value = "Aspirin description."
        mock_tr_cls.return_value = mock_tr

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        mock_model = MagicMock()
        mock_model.model_type = "text"
        mock_model.embed.return_value = np.ones(384, dtype=np.float32)
        embedder.model_cache["minilm_l6_v2"] = mock_model

        embedder.embed_description("aspirin", model="minilm_l6_v2", entity_type="molecule")

        call_kwargs = mock_tr.get_combined_description.call_args
        assert call_kwargs[1]["entity_type"] == "molecule"

    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_sources_passed_through(self, mock_tr_cls, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        mock_tr = MagicMock()
        mock_tr.get_combined_description.return_value = "desc"
        mock_tr_cls.return_value = mock_tr

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        mock_model = MagicMock()
        mock_model.model_type = "text"
        mock_model.embed.return_value = np.ones(384, dtype=np.float32)
        embedder.model_cache["minilm_l6_v2"] = mock_model

        embedder.embed_description("TP53", model="minilm_l6_v2", sources=["mygene", "wikipedia"])

        call_kwargs = mock_tr.get_combined_description.call_args
        assert call_kwargs[1]["sources"] == ["mygene", "wikipedia"]


class TestEmbedDescriptionsBatch:
    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_batch_returns_list(self, mock_tr_cls, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        mock_tr = MagicMock()
        mock_tr.get_combined_description.return_value = "Gene description."
        mock_tr_cls.return_value = mock_tr

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        mock_model = MagicMock()
        mock_model.model_type = "text"
        mock_model.embed_batch.return_value = [
            np.ones(384, dtype=np.float32),
            np.ones(384, dtype=np.float32),
        ]
        embedder.model_cache["minilm_l6_v2"] = mock_model

        results = embedder.embed_descriptions_batch(
            ["TP53", "BRCA1"], model="minilm_l6_v2",
        )
        assert len(results) == 2
        assert all(r is not None for r in results)

    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_batch_handles_failures(self, mock_tr_cls, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        mock_tr = MagicMock()

        def side_effect(ident, **kwargs):
            if ident == "FAKEGENE":
                raise RuntimeError("API error")
            return "Description text."

        mock_tr.get_combined_description.side_effect = side_effect
        mock_tr_cls.return_value = mock_tr

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        mock_model = MagicMock()
        mock_model.model_type = "text"
        mock_model.embed_batch.return_value = [np.ones(384, dtype=np.float32)]
        embedder.model_cache["minilm_l6_v2"] = mock_model

        results = embedder.embed_descriptions_batch(
            ["TP53", "FAKEGENE"], model="minilm_l6_v2",
        )
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None
