"""Shared fixtures and mock helpers for the embpy test suite."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


@pytest.fixture
def cpu_device():
    """Provide a CPU torch device."""
    return torch.device("cpu")


@pytest.fixture
def mock_tensor_2d():
    """A (seq_len=5, hidden_dim=8) tensor for testing pooling."""
    return torch.randn(5, 8)


@pytest.fixture
def mock_tensor_3d():
    """A (batch=2, seq_len=5, hidden_dim=8) tensor for testing batch pooling."""
    return torch.randn(2, 5, 8)


@pytest.fixture
def sample_dna_sequence():
    """A short DNA sequence for testing."""
    return "ACGTACGTACGTACGT"


@pytest.fixture
def sample_protein_sequence():
    """A short protein sequence for testing."""
    return "MTEYKLVVVGAGGVGKSALT"


@pytest.fixture
def sample_smiles():
    """A valid SMILES string (ethanol)."""
    return "CCO"


@pytest.fixture
def sample_smiles_list():
    """A list of valid SMILES strings."""
    return ["CCO", "CCC", "c1ccccc1"]


def make_mock_model_wrapper(model_type: str = "dna", embedding_dim: int = 8):
    """Create a mock BaseModelWrapper with controllable behavior."""
    mock = MagicMock()
    mock.model_type = model_type
    mock.available_pooling_strategies = ["mean", "max", "cls"]

    def _embed(input, pooling_strategy="mean", **kwargs):
        return np.random.randn(embedding_dim).astype(np.float32)

    def _embed_batch(inputs, pooling_strategy="mean", **kwargs):
        return [np.random.randn(embedding_dim).astype(np.float32) for _ in inputs]

    mock.embed = MagicMock(side_effect=_embed)
    mock.embed_batch = MagicMock(side_effect=_embed_batch)
    mock.load = MagicMock()

    return mock
