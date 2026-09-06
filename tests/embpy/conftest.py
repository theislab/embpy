"""Shared fixtures and mock helpers for the embpy test suite.

Importing this conftest must NOT require ``torch``: the core test suite is
designed to run on a lightweight ``pip install embpy[test]`` install with no
deep-learning stack. Tests that genuinely need ``torch`` live under
``tests/embpy/models/`` and in ``test_embedder.py``; when ``torch`` is absent
those files are skipped wholesale via ``collect_ignore`` below, and the few
torch-using fixtures here import ``torch`` lazily so plain collection stays
dependency-free.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the torch-only portions of the suite when torch is not installed, so the
# core suite still collects and runs on a lightweight install.
try:
    import torch  # noqa: F401

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

# Files that import torch (or BioEmbedder/model wrappers, which pull torch) at
# MODULE level, or whose every test needs torch at runtime. Skipped wholesale
# when torch is absent. Paths are relative to this conftest's directory
# (tests/embpy/); naming the `models` directory also skips its torch conftest.
_TORCH_ONLY_FILES = [
    "test_embedder.py",
    "test_embedder_text.py",
    "test_hpa_morphology_batch.py",
    "test_weighted_protein_embedding.py",
    # The registry maps model keys to wrapper classes, so importing it pulls
    # the (torch-based) model wrappers.
    "test_registry_split.py",
    "models",
    "io/test_bioembedder_embed_standardized.py",
]
collect_ignore: list[str] = [] if _HAVE_TORCH else list(_TORCH_ONLY_FILES)


def pytest_collection_modifyitems(config, items):
    """Skip individual ``@pytest.mark.requires_torch`` tests when torch is absent.

    Used for tests that live alongside core (torch-free) tests in the same
    file -- e.g. the BioEmbedder cases in ``test_multi_species.py`` -- so the
    surrounding core tests still run on a lightweight install.
    """
    if _HAVE_TORCH:
        return
    skip_torch = pytest.mark.skip(reason="requires the optional 'torch' dependency (pip install embpy[models])")
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip_torch)


@pytest.fixture
def cpu_device():
    """Provide a CPU torch device."""
    import torch

    return torch.device("cpu")


@pytest.fixture
def mock_tensor_2d():
    """A (seq_len=5, hidden_dim=8) tensor for testing pooling."""
    import torch

    return torch.randn(5, 8)


@pytest.fixture
def mock_tensor_3d():
    """A (batch=2, seq_len=5, hidden_dim=8) tensor for testing batch pooling."""
    import torch

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
