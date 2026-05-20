"""Tests for embpy.tl.weighted_protein_embedding -- WeightedProteinEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


EMB_DIM = 64


def _mock_embedder():
    """Create a mock BioEmbedder with controllable behavior."""
    embedder = MagicMock()
    embedder.protein_resolver = MagicMock()
    embedder.protein_resolver.get_canonical_sequence = MagicMock(
        return_value="MEEPQSDPSVEPPLSQETFSDLWKLLP",
    )
    embedder.protein_resolver.resolve_uniprot_id = MagicMock(
        return_value="P04637",
    )

    def _embed_protein(identifier, model, id_type="symbol", organism="human",
                       pooling_strategy="mean", isoform="canonical", **kw):
        if isoform == "all":
            return {
                "P04637": np.random.randn(EMB_DIM).astype(np.float32),
                "P04637-2": np.random.randn(EMB_DIM).astype(np.float32),
                "P04637-3": np.random.randn(EMB_DIM).astype(np.float32),
            }
        return np.random.randn(EMB_DIM).astype(np.float32)

    embedder.embed_protein = MagicMock(side_effect=_embed_protein)

    mock_model = MagicMock()
    mock_model.model_type = "protein"
    mock_model.tokenizer = MagicMock()
    mock_model.device = "cpu"

    import torch
    hidden = torch.randn(1, 27, EMB_DIM)
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden
    mock_model.model = MagicMock(return_value=mock_outputs)
    mock_model.tokenizer.return_value = {
        "input_ids": torch.ones(1, 27, dtype=torch.long),
        "attention_mask": torch.ones(1, 27, dtype=torch.long),
    }

    embedder._get_model = MagicMock(return_value=mock_model)
    return embedder


@pytest.fixture
def wpe():
    from embpy.tl.weighted_protein_embedding import WeightedProteinEmbedder
    return WeightedProteinEmbedder(_mock_embedder(), organism="human")


# =====================================================================
# TPM-weighted isoform average
# =====================================================================


class TestTPMWeighted:
    def test_equal_weights(self, wpe):
        emb = wpe.tpm_weighted_embedding("TP53", model="esm2_650M")
        assert emb.shape == (EMB_DIM,)
        assert emb.dtype == np.float32

    def test_explicit_tpm(self, wpe):
        emb = wpe.tpm_weighted_embedding(
            "TP53", model="esm2_650M",
            tpm_values={"P04637": 90.0, "P04637-2": 10.0, "P04637-3": 0.0},
        )
        assert emb.shape == (EMB_DIM,)

    def test_zero_weights_fallback(self, wpe):
        emb = wpe.tpm_weighted_embedding(
            "TP53", model="esm2_650M",
            tpm_values={"P04637": 0.0, "P04637-2": 0.0, "P04637-3": 0.0},
        )
        assert emb.shape == (EMB_DIM,)


# =====================================================================
# Annotation-weighted pooling
# =====================================================================


class TestAnnotationWeighted:
    @patch("embpy.resources.protein_annotator.ProteinAnnotator._fetch_uniprot_entry")
    def test_annotation_weighted(self, mock_fetch, wpe):
        mock_fetch.return_value = {
            "features": [
                {"type": "Active site", "location": {"start": {"value": 5}, "end": {"value": 5}}, "description": ""},
                {"type": "Binding site", "location": {"start": {"value": 10}, "end": {"value": 15}}, "description": "", "ligand": {"name": "DNA"}},
            ],
        }
        emb = wpe.annotation_weighted_embedding(
            "TP53", model="esm2_650M", site_boost=3.0,
        )
        assert emb.shape == (EMB_DIM,)


# =====================================================================
# Expression-context embedding
# =====================================================================


class TestExpressionContext:
    def test_with_explicit_vector(self, wpe):
        ctx = np.random.randn(54).astype(np.float32)
        emb = wpe.expression_context_embedding(
            "TP53", model="esm2_650M", expression_vector=ctx,
        )
        assert emb.shape == (EMB_DIM + 54,)

    @patch("embpy.resources.gene_annotator.GeneAnnotator.get_tissue_expression")
    def test_with_gtex(self, mock_gtex, wpe):
        mock_gtex.return_value = [
            {"tissue": f"tissue_{i}", "median_tpm": float(i)} for i in range(54)
        ]
        emb = wpe.expression_context_embedding(
            "TP53", model="esm2_650M", use_gtex=True,
        )
        assert emb.shape[0] > EMB_DIM


# =====================================================================
# embed_perturbation (unified)
# =====================================================================


class TestEmbedPerturbation:
    def test_canonical(self, wpe):
        emb = wpe.embed_perturbation("TP53", model="esm2_650M", strategy="canonical")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (EMB_DIM,)

    def test_tpm_weighted(self, wpe):
        emb = wpe.embed_perturbation(
            "TP53", model="esm2_650M", strategy="tpm_weighted",
            tpm_values={"P04637": 50.0, "P04637-2": 50.0},
        )
        assert emb.shape == (EMB_DIM,)

    def test_expression_context(self, wpe):
        ctx = np.ones(10, dtype=np.float32)
        emb = wpe.embed_perturbation(
            "TP53", model="esm2_650M", strategy="expression_context",
            expression_vector=ctx,
        )
        assert emb.shape == (EMB_DIM + 10,)

    def test_invalid_strategy(self, wpe):
        with pytest.raises(ValueError, match="Unknown strategy"):
            wpe.embed_perturbation("TP53", model="esm2_650M", strategy="invalid")
