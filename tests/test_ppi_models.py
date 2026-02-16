"""Tests for embpy.models.ppi_models – PPI GNN embeddings."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# torch_geometric may not be installed – skip the entire module if missing
pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from embpy.models.ppi_models import (
    GNNEncoder,
    PPIGNNWrapper,
    _fetch_string_network,
    load_string_links_file,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def small_interactions() -> pd.DataFrame:
    """A tiny PPI network with 5 nodes and 6 edges."""
    return pd.DataFrame(
        {
            "protein1": ["TP53", "TP53", "BRCA1", "EGFR", "EGFR", "KRAS"],
            "protein2": ["BRCA1", "EGFR", "EGFR", "KRAS", "MYC", "MYC"],
            "combined_score": [900, 800, 700, 600, 500, 400],
        }
    )


@pytest.fixture
def wrapper(small_interactions: pd.DataFrame) -> PPIGNNWrapper:
    """A PPIGNNWrapper with a built graph, ready for training."""
    w = PPIGNNWrapper(
        gnn_type="gcn",
        input_dim=16,
        hidden_dim=32,
        output_dim=16,
        num_layers=2,
        dropout=0.0,
        score_threshold=0,
    )
    w.load(torch.device("cpu"))
    w.build_graph(interactions_df=small_interactions)
    return w


@pytest.fixture
def trained_wrapper(wrapper: PPIGNNWrapper) -> PPIGNNWrapper:
    """A PPIGNNWrapper that has been trained for a few epochs."""
    wrapper.train_embeddings(epochs=10, lr=0.01, log_every=100)
    return wrapper


# =====================================================================
# GNNEncoder tests
# =====================================================================


class TestGNNEncoder:
    def test_forward_shape(self) -> None:
        enc = GNNEncoder(
            num_nodes=5, input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, gnn_type="gcn",
        )
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = enc(edge_index)
        assert out.shape == (5, 8)

    def test_gnn_types(self) -> None:
        for gnn_type in ("gcn", "sage", "gat"):
            enc = GNNEncoder(
                num_nodes=4, input_dim=8, hidden_dim=16, output_dim=8,
                num_layers=2, gnn_type=gnn_type,
            )
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = enc(edge_index)
            assert out.shape == (4, 8), f"Failed for gnn_type={gnn_type}"

    def test_invalid_gnn_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown gnn_type"):
            GNNEncoder(num_nodes=4, input_dim=8, hidden_dim=8, output_dim=8,
                       num_layers=2, gnn_type="invalid")

    def test_min_layers(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be >= 2"):
            GNNEncoder(num_nodes=4, input_dim=8, hidden_dim=8, output_dim=8,
                       num_layers=1, gnn_type="gcn")

    def test_three_layers(self) -> None:
        enc = GNNEncoder(
            num_nodes=5, input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=3, gnn_type="gcn",
        )
        assert len(enc.convs) == 3

    def test_dropout_in_training(self) -> None:
        enc = GNNEncoder(
            num_nodes=5, input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, gnn_type="gcn", dropout=0.5,
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        enc.train()
        out_train = enc(edge_index)
        enc.eval()
        out_eval = enc(edge_index)
        # Outputs may differ due to dropout
        assert out_train.shape == out_eval.shape


# =====================================================================
# PPIGNNWrapper – graph building
# =====================================================================


class TestBuildGraph:
    def test_from_dataframe(self, small_interactions: pd.DataFrame) -> None:
        w = PPIGNNWrapper(
            input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, score_threshold=0,
        )
        w.load(torch.device("cpu"))
        w.build_graph(interactions_df=small_interactions)

        assert w.num_nodes == 5
        assert w.num_edges > 0
        assert "TP53" in w.node_to_idx
        assert "MYC" in w.node_to_idx
        assert w.encoder is not None

    def test_score_filtering(self) -> None:
        df = pd.DataFrame({
            "protein1": ["A", "B", "C"],
            "protein2": ["B", "C", "D"],
            "combined_score": [900, 300, 100],
        })
        w = PPIGNNWrapper(
            input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, score_threshold=500,
        )
        w.load(torch.device("cpu"))
        w.build_graph(interactions_df=df)
        # Only 1 edge survives (A-B with score=900)
        assert w.num_nodes == 2
        assert "A" in w.node_to_idx
        assert "B" in w.node_to_idx
        assert "C" not in w.node_to_idx

    def test_empty_after_filter_raises(self) -> None:
        df = pd.DataFrame({
            "protein1": ["A"],
            "protein2": ["B"],
            "combined_score": [100],
        })
        w = PPIGNNWrapper(
            input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, score_threshold=999,
        )
        w.load(torch.device("cpu"))
        with pytest.raises(ValueError, match="No edges remain"):
            w.build_graph(interactions_df=df)

    def test_no_source_raises(self) -> None:
        w = PPIGNNWrapper(input_dim=8, hidden_dim=16, output_dim=8, num_layers=2)
        w.load(torch.device("cpu"))
        with pytest.raises(ValueError, match="exactly one"):
            w.build_graph()

    def test_multiple_sources_raises(self, small_interactions: pd.DataFrame) -> None:
        w = PPIGNNWrapper(input_dim=8, hidden_dim=16, output_dim=8, num_layers=2)
        w.load(torch.device("cpu"))
        with pytest.raises(ValueError, match="exactly one"):
            w.build_graph(
                interactions_df=small_interactions,
                gene_ids=["TP53"],
            )

    def test_from_file(self, tmp_path) -> None:
        # Write a mock STRING links file
        links = "protein1 protein2 combined_score\n9606.ENSP001 9606.ENSP002 900\n"
        fpath = tmp_path / "links.txt"
        fpath.write_text(links)

        w = PPIGNNWrapper(
            input_dim=8, hidden_dim=16, output_dim=8,
            num_layers=2, score_threshold=0,
        )
        w.load(torch.device("cpu"))
        w.build_graph(string_links_file=str(fpath))
        # Species prefix stripped: "ENSP001", "ENSP002"
        assert w.num_nodes == 2

    def test_graph_genes_property(self, wrapper: PPIGNNWrapper) -> None:
        genes = wrapper.graph_genes
        assert sorted(genes) == ["BRCA1", "EGFR", "KRAS", "MYC", "TP53"]


# =====================================================================
# PPIGNNWrapper – training
# =====================================================================


class TestTraining:
    def test_train_returns_losses(self, wrapper: PPIGNNWrapper) -> None:
        losses = wrapper.train_embeddings(epochs=5, lr=0.01)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self, wrapper: PPIGNNWrapper) -> None:
        losses = wrapper.train_embeddings(epochs=50, lr=0.01)
        # The loss should generally trend downward
        assert losses[-1] < losses[0]

    def test_train_before_build_raises(self) -> None:
        w = PPIGNNWrapper(input_dim=8, hidden_dim=16, output_dim=8, num_layers=2)
        w.load(torch.device("cpu"))
        with pytest.raises(RuntimeError, match="build_graph"):
            w.train_embeddings(epochs=5)


# =====================================================================
# PPIGNNWrapper – embedding
# =====================================================================


class TestEmbed:
    def test_embed_single(self, trained_wrapper: PPIGNNWrapper) -> None:
        emb = trained_wrapper.embed("TP53")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (16,)
        assert emb.dtype == np.float32

    def test_embed_all_genes(self, trained_wrapper: PPIGNNWrapper) -> None:
        for gene in trained_wrapper.graph_genes:
            emb = trained_wrapper.embed(gene)
            assert emb.shape == (16,)

    def test_embed_unknown_gene_raises(self, trained_wrapper: PPIGNNWrapper) -> None:
        with pytest.raises(ValueError, match="not found in the PPI graph"):
            trained_wrapper.embed("NONEXISTENT_GENE")

    def test_embed_batch(self, trained_wrapper: PPIGNNWrapper) -> None:
        genes = ["TP53", "BRCA1", "EGFR"]
        embs = trained_wrapper.embed_batch(genes)
        assert len(embs) == 3
        for emb in embs:
            assert emb.shape == (16,)

    def test_embed_batch_missing_gene_returns_zeros(
        self, trained_wrapper: PPIGNNWrapper
    ) -> None:
        embs = trained_wrapper.embed_batch(["TP53", "NONEXISTENT"])
        assert embs[0].shape == (16,)
        assert np.allclose(embs[1], 0.0)

    def test_embed_caching(self, trained_wrapper: PPIGNNWrapper) -> None:
        emb1 = trained_wrapper.embed("TP53")
        emb2 = trained_wrapper.embed("TP53")
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_before_train_uses_random_init(
        self, wrapper: PPIGNNWrapper
    ) -> None:
        # Embedding before training should still work (uses random init)
        emb = wrapper.embed("TP53")
        assert emb.shape == (16,)

    def test_different_genes_different_embeddings(
        self, trained_wrapper: PPIGNNWrapper
    ) -> None:
        emb_tp53 = trained_wrapper.embed("TP53")
        emb_myc = trained_wrapper.embed("MYC")
        assert not np.allclose(emb_tp53, emb_myc)


# =====================================================================
# PPIGNNWrapper – checkpoint save / load
# =====================================================================


class TestCheckpoint:
    def test_save_and_load(self, trained_wrapper: PPIGNNWrapper) -> None:
        emb_before = trained_wrapper.embed("TP53").copy()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        trained_wrapper.save_checkpoint(path)

        # Load into a fresh wrapper
        loaded = PPIGNNWrapper()
        loaded.load(torch.device("cpu"))
        loaded._load_checkpoint(path)

        emb_after = loaded.embed("TP53")
        np.testing.assert_array_almost_equal(emb_before, emb_after)
        assert loaded.num_nodes == trained_wrapper.num_nodes
        assert loaded.graph_genes == trained_wrapper.graph_genes

    def test_save_before_train_raises(self) -> None:
        w = PPIGNNWrapper(input_dim=8, hidden_dim=16, output_dim=8, num_layers=2)
        w.load(torch.device("cpu"))
        with pytest.raises(RuntimeError, match="No trained model"):
            w.save_checkpoint("/tmp/test.pt")

    def test_load_from_model_name(self, trained_wrapper: PPIGNNWrapper) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        trained_wrapper.save_checkpoint(path)

        # model_path_or_name triggers auto-load in load()
        loaded = PPIGNNWrapper(model_path_or_name=path)
        loaded.load(torch.device("cpu"))

        emb = loaded.embed("TP53")
        assert emb.shape == (16,)


# =====================================================================
# load_string_links_file
# =====================================================================


class TestLoadStringFile:
    def test_strips_species_prefix(self, tmp_path) -> None:
        content = (
            "protein1 protein2 combined_score\n"
            "9606.ENSP00000269305 9606.ENSP00000275493 900\n"
            "9606.ENSP00000269305 9606.ENSP00000338799 700\n"
        )
        fpath = tmp_path / "links.txt"
        fpath.write_text(content)
        df = load_string_links_file(str(fpath))
        assert "ENSP00000269305" in df["protein1"].values
        assert "ENSP00000275493" in df["protein2"].values
        assert "9606." not in str(df["protein1"].iloc[0])


# =====================================================================
# _fetch_string_network (mocked)
# =====================================================================


class TestFetchStringNetwork:
    @patch("embpy.models.ppi_models.requests.post")
    def test_fetches_and_parses(self, mock_post: MagicMock) -> None:
        tsv_data = (
            "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\n"
            "9606.ENSP001\t9606.ENSP002\tTP53\tBRCA1\t900\n"
        )
        mock_resp = MagicMock()
        mock_resp.text = tsv_data
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        df = _fetch_string_network(["TP53", "BRCA1"], species=9606)
        assert "protein1" in df.columns
        assert "protein2" in df.columns
        assert df["protein1"].iloc[0] == "TP53"
        assert df["protein2"].iloc[0] == "BRCA1"
