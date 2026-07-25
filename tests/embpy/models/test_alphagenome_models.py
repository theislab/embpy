"""Tests for AlphaGenomeWrapper (Google DeepMind cloud API client) using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from embpy.models import alphagenome_models as ag
from embpy.models.alphagenome_models import AlphaGenomeWrapper


def _mock_dna_client() -> MagicMock:
    """A MagicMock standing in for the `alphagenome.models.dna_client` module."""
    mock = MagicMock()
    mock.ModelVersion.FOLD_0 = "FOLD_0_ENUM"
    mock.Organism.HOMO_SAPIENS = "HUMAN_ENUM"
    mock.Organism.MUS_MUSCULUS = "MOUSE_ENUM"
    return mock


class TestAlphaGenomeWrapperInit:
    def test_init_defaults(self):
        w = AlphaGenomeWrapper()
        assert w.model_name == "alphagenome"
        assert w.model_type == "dna"
        assert w.organism == "human"
        assert w.model_version is None
        assert w.context_window == AlphaGenomeWrapper.DEFAULT_CONTEXT_WINDOW
        assert w.client is None

    def test_init_mouse_organism(self):
        w = AlphaGenomeWrapper(organism="mouse")
        assert w.organism == "mouse"

    def test_init_invalid_organism_raises(self):
        with pytest.raises(ValueError, match="Unknown organism"):
            AlphaGenomeWrapper(organism="fly")

    def test_init_custom_context_window(self):
        w = AlphaGenomeWrapper(context_window=1024)
        assert w.context_window == 1024


class TestAlphaGenomeWrapperLoad:
    def test_load_without_package_raises(self):
        w = AlphaGenomeWrapper()
        with patch.object(ag, "dna_client", None):
            with pytest.raises(ImportError, match="not installed"):
                w.load()

    def test_load_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ALPHAGENOME_API_KEY", raising=False)
        w = AlphaGenomeWrapper()
        with patch.object(ag, "dna_client", _mock_dna_client()):
            with pytest.raises(ValueError, match="ALPHAGENOME_API_KEY"):
                w.load()

    def test_load_creates_client(self, monkeypatch):
        monkeypatch.setenv("ALPHAGENOME_API_KEY", "test-key")
        mock_client_module = _mock_dna_client()
        sentinel_client = MagicMock()
        mock_client_module.create.return_value = sentinel_client

        w = AlphaGenomeWrapper()
        with patch.object(ag, "dna_client", mock_client_module):
            w.load()

        assert w.client is sentinel_client
        assert w.model is sentinel_client
        mock_client_module.create.assert_called_once_with("test-key", model_version=None)

    def test_load_resolves_model_version(self, monkeypatch):
        monkeypatch.setenv("ALPHAGENOME_API_KEY", "test-key")
        mock_client_module = _mock_dna_client()

        w = AlphaGenomeWrapper(model_version="FOLD_0")
        with patch.object(ag, "dna_client", mock_client_module):
            w.load()

        mock_client_module.create.assert_called_once_with("test-key", model_version="FOLD_0_ENUM")

    def test_load_already_initialized_is_noop(self, monkeypatch):
        monkeypatch.setenv("ALPHAGENOME_API_KEY", "test-key")
        mock_client_module = _mock_dna_client()

        w = AlphaGenomeWrapper()
        existing_client = MagicMock()
        w.client = existing_client
        with patch.object(ag, "dna_client", mock_client_module):
            w.load()

        mock_client_module.create.assert_not_called()
        assert w.client is existing_client


class TestAlphaGenomeWrapperEmbed:
    def test_embed_without_load_raises(self):
        w = AlphaGenomeWrapper()
        with pytest.raises(RuntimeError, match="not initialized"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = AlphaGenomeWrapper()
        with pytest.raises(RuntimeError, match="not initialized"):
            w.embed_batch(["ACGT"])

    def test_embed_invalid_pooling_raises(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def _mock_prediction(self, values: np.ndarray) -> MagicMock:
        prediction = MagicMock()
        prediction.rna_seq.values = values
        return prediction

    def test_embed_mean_pooling(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        w.client.predict_sequence.return_value = self._mock_prediction(values)

        with patch.object(ag, "dna_client", _mock_dna_client()), patch.object(ag, "dna_output", MagicMock()):
            result = w.embed("ACGT", pooling_strategy="mean")

        np.testing.assert_allclose(result, values.mean(axis=0))

    def test_embed_max_pooling(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        w.client.predict_sequence.return_value = self._mock_prediction(values)

        with patch.object(ag, "dna_client", _mock_dna_client()), patch.object(ag, "dna_output", MagicMock()):
            result = w.embed("ACGT", pooling_strategy="max")

        np.testing.assert_allclose(result, values.max(axis=0))

    def test_embed_none_pooling_returns_raw_array(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        w.client.predict_sequence.return_value = self._mock_prediction(values)

        with patch.object(ag, "dna_client", _mock_dna_client()), patch.object(ag, "dna_output", MagicMock()):
            result = w.embed("ACGT", pooling_strategy="none")

        np.testing.assert_allclose(result, values)

    def test_embed_passes_organism_enum(self):
        w = AlphaGenomeWrapper(organism="mouse")
        w.client = MagicMock()
        values = np.zeros((2, 2), dtype=np.float32)
        w.client.predict_sequence.return_value = self._mock_prediction(values)
        mock_dna_client = _mock_dna_client()

        with patch.object(ag, "dna_client", mock_dna_client), patch.object(ag, "dna_output", MagicMock()):
            w.embed("ACGT")

        assert w.client.predict_sequence.call_args.kwargs["organism"] == "MOUSE_ENUM"

    def test_embed_batch_calls_embed_per_sequence(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        values = np.zeros((2, 2), dtype=np.float32)
        w.client.predict_sequence.return_value = self._mock_prediction(values)

        with patch.object(ag, "dna_client", _mock_dna_client()), patch.object(ag, "dna_output", MagicMock()):
            results = w.embed_batch(["ACGT", "TTTT", "GGGG"])

        assert len(results) == 3
        assert w.client.predict_sequence.call_count == 3

    def test_embed_batch_empty_returns_empty(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        assert w.embed_batch([]) == []


class TestAlphaGenomeWrapperTrackMetadata:
    def test_get_track_metadata_without_load_raises(self):
        w = AlphaGenomeWrapper()
        with pytest.raises(RuntimeError, match="not initialized"):
            w.get_track_metadata()

    def test_get_track_metadata_specific_output_type(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        rna_df = pd.DataFrame({"identifier": ["a", "b"]})
        metadata = MagicMock()
        metadata.rna_seq = rna_df
        w.client.output_metadata.return_value = metadata

        with patch.object(ag, "dna_client", _mock_dna_client()):
            result = w.get_track_metadata(output_type="RNA_SEQ")

        assert result is rna_df

    def test_get_track_metadata_concatenates_all(self):
        import types

        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        df1 = pd.DataFrame({"identifier": ["a"]})
        df2 = pd.DataFrame({"identifier": ["b"]})
        metadata = types.SimpleNamespace(rna_seq=df1, atac=df2, not_a_frame="ignored")
        w.client.output_metadata.return_value = metadata

        with patch.object(ag, "dna_client", _mock_dna_client()):
            result = w.get_track_metadata()

        assert len(result) == 2
        assert set(result["identifier"]) == {"a", "b"}


class TestAlphaGenomeWrapperVariantEffect:
    def _mock_variant_scorers(self, df: pd.DataFrame) -> MagicMock:
        mock = MagicMock()
        mock.tidy_scores.return_value = df
        return mock

    def test_predict_variant_effect_without_load_raises(self):
        w = AlphaGenomeWrapper()
        with pytest.raises(RuntimeError, match="not initialized"):
            w.predict_variant_effect("chr1", 100, "A", "T")

    def test_predict_variant_effect_default_interval(self):
        w = AlphaGenomeWrapper(context_window=100)
        w.client = MagicMock()
        w.client.score_variant.return_value = MagicMock()
        df = pd.DataFrame(
            {
                "raw_score": [0.5, -0.3],
                "track_name": ["trackA", "trackB"],
                "ontology_curie": ["UBERON:1", "UBERON:2"],
            }
        )
        mock_genome = MagicMock()

        with (
            patch.object(ag, "dna_client", _mock_dna_client()),
            patch.object(ag, "dna_output", MagicMock()),
            patch.object(ag, "genome", mock_genome),
            patch.object(ag, "variant_scorers", self._mock_variant_scorers(df)),
        ):
            scores, track_names = w.predict_variant_effect("chr1", 100, "A", "T")

        np.testing.assert_allclose(scores, [0.5, -0.3])
        assert track_names == ["trackA", "trackB"]
        # default interval centered on pos=100 with context_window=100 -> half=50
        mock_genome.Interval.assert_called_once_with(chromosome="chr1", start=50, end=150)

    def test_predict_variant_effect_explicit_interval(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        w.client.score_variant.return_value = MagicMock()
        df = pd.DataFrame({"raw_score": [1.0], "track_name": ["t"]})
        mock_genome = MagicMock()

        with (
            patch.object(ag, "dna_client", _mock_dna_client()),
            patch.object(ag, "dna_output", MagicMock()),
            patch.object(ag, "genome", mock_genome),
            patch.object(ag, "variant_scorers", self._mock_variant_scorers(df)),
        ):
            w.predict_variant_effect("chr1", 100, "A", "T", interval_start=10, interval_end=20)

        mock_genome.Interval.assert_called_once_with(chromosome="chr1", start=10, end=20)

    def test_predict_variant_effect_ontology_filter(self):
        w = AlphaGenomeWrapper()
        w.client = MagicMock()
        w.client.score_variant.return_value = MagicMock()
        df = pd.DataFrame(
            {
                "raw_score": [0.5, -0.3],
                "track_name": ["trackA", "trackB"],
                "ontology_curie": ["UBERON:1", "UBERON:2"],
            }
        )

        with (
            patch.object(ag, "dna_client", _mock_dna_client()),
            patch.object(ag, "dna_output", MagicMock()),
            patch.object(ag, "genome", MagicMock()),
            patch.object(ag, "variant_scorers", self._mock_variant_scorers(df)),
        ):
            scores, track_names = w.predict_variant_effect(
                "chr1", 100, "A", "T", ontology_terms=["UBERON:1"]
            )

        assert track_names == ["trackA"]
        np.testing.assert_allclose(scores, [0.5])
