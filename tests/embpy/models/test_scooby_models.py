"""Tests for ScoobyWrapper (gagneurlab/scooby single-cell DNA model) using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embpy.models import scooby_models as sm
from embpy.models.scooby_models import ScoobyWrapper


class TestScoobyWrapperInit:
    def test_init_defaults_resolve_onek1k_checkpoint(self):
        w = ScoobyWrapper()
        assert w.model_name == "lauradmartens/onek1k-scooby"
        assert w.model_type == "dna"
        assert w.cell_emb_dim == 10
        assert w.n_tracks == 2
        assert w.use_transform_borzoi_emb is True
        assert w.clip_soft == 5.0
        assert w.embedding_dim == 1920

    def test_init_neurips_checkpoint(self):
        w = ScoobyWrapper(model_path_or_name="johahi/neurips-scooby")
        assert w.cell_emb_dim == 14
        assert w.n_tracks == 3

    def test_init_epicardioids_checkpoint(self):
        w = ScoobyWrapper(model_path_or_name="lauradmartens/epicardioids-scooby")
        assert w.cell_emb_dim == 50
        assert w.n_tracks == 3

    def test_init_unknown_checkpoint_without_hparams_raises(self):
        with pytest.raises(ValueError, match="could not be inferred"):
            ScoobyWrapper(model_path_or_name="someorg/custom-scooby")

    def test_init_unknown_checkpoint_with_explicit_hparams(self):
        w = ScoobyWrapper(model_path_or_name="someorg/custom-scooby", cell_emb_dim=20, n_tracks=3)
        assert w.cell_emb_dim == 20
        assert w.n_tracks == 3
        assert w.use_transform_borzoi_emb is True  # default fallback
        assert w.clip_soft == 5.0  # default fallback

    def test_init_explicit_hparams_override_known_checkpoint(self):
        w = ScoobyWrapper(clip_soft=3.0)
        assert w.clip_soft == 3.0
        assert w.cell_emb_dim == 10  # still resolved from KNOWN_CHECKPOINTS

    def test_track_names_two_tracks(self):
        w = ScoobyWrapper()  # onek1k -> n_tracks=2
        assert w.TRACK_NAMES == ["RNA:+", "RNA:-"]

    def test_track_names_three_tracks(self):
        w = ScoobyWrapper(model_path_or_name="johahi/neurips-scooby")
        assert w.TRACK_NAMES == ["RNA:+", "RNA:-", "ATAC"]

    def test_track_names_generic_fallback(self):
        w = ScoobyWrapper(model_path_or_name="someorg/custom-scooby", cell_emb_dim=5, n_tracks=4)
        assert w.TRACK_NAMES == ["track_0", "track_1", "track_2", "track_3"]

    def test_get_track_metadata(self):
        w = ScoobyWrapper()
        df = w.get_track_metadata()
        assert list(df["identifier"]) == ["RNA:+", "RNA:-"]


class TestScoobyWrapperLoad:
    def test_load_without_package_raises(self):
        w = ScoobyWrapper()
        with patch.object(sm, "Scooby", None):
            with pytest.raises(ImportError, match="not installed"):
                w.load(torch.device("cpu"))

    def test_load_already_loaded_is_noop(self):
        w = ScoobyWrapper()
        w.model = MagicMock()
        mock_scooby_cls = MagicMock()
        with patch.object(sm, "Scooby", mock_scooby_cls):
            w.load(torch.device("cpu"))
        mock_scooby_cls.from_pretrained.assert_not_called()

    def test_load_success(self):
        w = ScoobyWrapper()
        device = torch.device("cpu")
        mock_scooby_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.to.return_value = mock_instance
        mock_instance.eval.return_value = mock_instance
        mock_scooby_cls.from_pretrained.return_value = mock_instance

        with patch.object(sm, "Scooby", mock_scooby_cls):
            w.load(device)

        assert w.model is mock_instance
        assert w.device is device
        mock_scooby_cls.from_pretrained.assert_called_once_with(
            "lauradmartens/onek1k-scooby",
            cell_emb_dim=10,
            embedding_dim=1920,
            n_tracks=2,
            return_center_bins_only=True,
            disable_cache=False,
            use_transform_borzoi_emb=True,
        )

    def test_load_failure_wraps_in_runtime_error(self):
        w = ScoobyWrapper()
        mock_scooby_cls = MagicMock()
        mock_scooby_cls.from_pretrained.side_effect = OSError("network error")

        with patch.object(sm, "Scooby", mock_scooby_cls):
            with pytest.raises(RuntimeError, match="Could not load Scooby"):
                w.load(torch.device("cpu"))

        assert w.model is None
        assert w.device is None


class TestScoobyWrapperPreprocess:
    def test_preprocess_pads_short_sequence(self):
        w = ScoobyWrapper()
        result = w._preprocess_sequence("ACGT")
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_truncates_long_sequence(self):
        w = ScoobyWrapper()
        long_seq = "A" * (w.SEQUENCE_LENGTH + 100)
        result = w._preprocess_sequence(long_seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)

    def test_preprocess_exact_length(self):
        w = ScoobyWrapper()
        seq = "A" * w.SEQUENCE_LENGTH
        result = w._preprocess_sequence(seq)
        assert result.shape == (1, 4, w.SEQUENCE_LENGTH)


class TestScoobyWrapperCellEmbeddings:
    def test_prepare_single_embedding_is_unsqueezed(self):
        w = ScoobyWrapper()  # cell_emb_dim=10
        emb = np.random.rand(10)
        result = w._prepare_cell_embeddings(emb)
        assert result.shape == (1, 1, 10)

    def test_prepare_multi_cell_embeddings(self):
        w = ScoobyWrapper()
        emb = np.random.rand(5, 10)
        result = w._prepare_cell_embeddings(emb)
        assert result.shape == (1, 5, 10)

    def test_prepare_wrong_dim_raises(self):
        w = ScoobyWrapper()
        emb = np.random.rand(7)  # wrong last-dim size (expects 10)
        with pytest.raises(ValueError, match="cell_embeddings must have shape"):
            w._prepare_cell_embeddings(emb)


class TestScoobyWrapperProfileOffset:
    def test_profile_offset_bp_without_load_raises(self):
        w = ScoobyWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = w.profile_offset_bp

    def test_profile_offset_bp_computed_from_crop(self):
        w = ScoobyWrapper()
        mock_model = MagicMock()
        mock_model.crop.target_length = 16352  # matches Borzoi's crop, same trunk
        w.model = mock_model
        assert w.profile_offset_bp == (524_288 - 16352 * 32) // 2


class TestScoobyWrapperPredictProfile:
    def _mock_model(self, raw_tensor: torch.Tensor) -> MagicMock:
        model = MagicMock()
        model.forward_cell_embs_only.return_value = (MagicMock(), MagicMock())
        model.forward_sequence_w_convs.return_value = raw_tensor
        return model

    def test_without_load_raises(self):
        w = ScoobyWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.predict_profile("ACGT", cell_embeddings=np.random.rand(10))

    def test_single_cell_returns_tracks_by_bins(self):
        w = ScoobyWrapper()  # n_tracks=2
        w.device = torch.device("cpu")
        num_bins = 5
        raw = torch.randn(1, num_bins, 1 * w.n_tracks)  # (1, num_bins, num_cells*n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", lambda x, clip_soft: x):
            profile = w.predict_profile("ACGT", cell_embeddings=np.random.rand(10))

        assert isinstance(profile, np.ndarray)
        assert profile.shape == (w.n_tracks, num_bins)

    def test_pseudobulk_aggregation_sums_across_cells(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins, num_cells = 5, 3
        raw = torch.randn(1, num_bins, num_cells * w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", lambda x, clip_soft: x):
            profile = w.predict_profile(
                "ACGT", cell_embeddings=np.random.rand(num_cells, 10), aggregate="pseudobulk"
            )

        assert profile.shape == (w.n_tracks, num_bins)

    def test_aggregate_none_returns_per_cell(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins, num_cells = 5, 3
        raw = torch.randn(1, num_bins, num_cells * w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", lambda x, clip_soft: x):
            profile = w.predict_profile(
                "ACGT", cell_embeddings=np.random.rand(num_cells, 10), aggregate="none"
            )

        assert profile.shape == (num_cells, w.n_tracks, num_bins)

    def test_invalid_aggregate_raises(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins, num_cells = 5, 2
        raw = torch.randn(1, num_bins, num_cells * w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", lambda x, clip_soft: x):
            with pytest.raises(ValueError, match="Invalid aggregate"):
                w.predict_profile(
                    "ACGT", cell_embeddings=np.random.rand(num_cells, 10), aggregate="bogus"
                )

    def test_track_indices_subsets_output(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins = 5
        raw = torch.randn(1, num_bins, w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", lambda x, clip_soft: x):
            profile = w.predict_profile(
                "ACGT", cell_embeddings=np.random.rand(10), track_indices=[0]
            )

        assert profile.shape == (1, num_bins)

    def test_undo_squashed_scale_false_skips_transform(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins = 5
        raw = torch.randn(1, num_bins, w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale") as mock_undo:
            w.predict_profile("ACGT", cell_embeddings=np.random.rand(10), undo_squashed_scale=False)

        mock_undo.assert_not_called()

    def test_undo_squashed_scale_without_package_raises(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        num_bins = 5
        raw = torch.randn(1, num_bins, w.n_tracks)
        w.model = self._mock_model(raw)

        with patch.object(sm, "_scooby_undo_squashed_scale", None):
            with pytest.raises(ImportError):
                w.predict_profile("ACGT", cell_embeddings=np.random.rand(10), undo_squashed_scale=True)

    def test_unexpected_output_shape_raises(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        raw = torch.randn(1, 5)  # wrong: only 2 dims, expected 3
        w.model = self._mock_model(raw)

        with pytest.raises(RuntimeError, match="Unexpected Scooby output"):
            w.predict_profile("ACGT", cell_embeddings=np.random.rand(10))


class TestScoobyWrapperEmbed:
    def test_embed_without_load_raises(self):
        w = ScoobyWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("ACGT")

    def test_embed_batch_without_load_raises(self):
        w = ScoobyWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["ACGT"])

    def test_embed_invalid_pooling_raises(self):
        w = ScoobyWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        with pytest.raises(ValueError, match="Invalid pooling"):
            w.embed("ACGT", pooling_strategy="invalid")

    def _mock_trunk_model(self, trunk_tensor: torch.Tensor) -> MagicMock:
        model = MagicMock()
        model.forward_seq_to_emb.return_value = trunk_tensor
        return model

    def test_embed_mean_pooling(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        embedding_dim, num_bins = 1920, 10
        trunk = torch.randn(1, embedding_dim, num_bins)
        w.model = self._mock_trunk_model(trunk)

        result = w.embed("ACGT", pooling_strategy="mean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedding_dim,)
        assert not np.isnan(result).any()

    def test_embed_max_pooling(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        trunk = torch.randn(1, 16, 10)
        w.model = self._mock_trunk_model(trunk)

        result = w.embed("ACGT", pooling_strategy="max")
        assert result.shape == (16,)

    def test_embed_median_pooling(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        trunk = torch.randn(1, 16, 10)
        w.model = self._mock_trunk_model(trunk)

        result = w.embed("ACGT", pooling_strategy="median")
        assert result.shape == (16,)

    def test_embed_none_pooling_returns_per_bin(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        embedding_dim, num_bins = 16, 10
        trunk = torch.randn(1, embedding_dim, num_bins)
        w.model = self._mock_trunk_model(trunk)

        result = w.embed("ACGT", pooling_strategy="none")
        assert result.shape == (num_bins, embedding_dim)

    def test_embed_unexpected_trunk_shape_raises(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        w.model = self._mock_trunk_model(torch.randn(1, 16))  # wrong: 2 dims

        with pytest.raises(RuntimeError, match="Unexpected Scooby trunk output"):
            w.embed("ACGT")

    def test_embed_batch_calls_embed_per_sequence(self):
        w = ScoobyWrapper()
        w.device = torch.device("cpu")
        trunk = torch.randn(1, 1920, 10)
        w.model = self._mock_trunk_model(trunk)

        results = w.embed_batch(["ACGT", "TTTT"])
        assert len(results) == 2
        assert w.model.forward_seq_to_emb.call_count == 2

    def test_embed_batch_empty_returns_empty(self):
        w = ScoobyWrapper()
        w.model = MagicMock()
        w.device = torch.device("cpu")
        assert w.embed_batch([]) == []
