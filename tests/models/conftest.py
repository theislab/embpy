"""Shared fixtures for model wrapper tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def mock_hf_outputs():
    """Create a mock HuggingFace model output with last_hidden_state."""
    batch_size, seq_len, hidden_dim = 1, 10, 768
    output = MagicMock()
    output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    output.pooler_output = torch.randn(batch_size, hidden_dim)
    output.hidden_states = None
    return output


@pytest.fixture
def mock_hf_outputs_with_hidden():
    """Create a mock HuggingFace model output with hidden states."""
    batch_size, seq_len, hidden_dim, n_layers = 1, 10, 768, 6
    output = MagicMock()
    output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    output.pooler_output = torch.randn(batch_size, hidden_dim)
    output.hidden_states = tuple(
        torch.randn(batch_size, seq_len, hidden_dim) for _ in range(n_layers)
    )
    return output
