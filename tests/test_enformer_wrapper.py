import logging

import numpy as np
import pytest
import torch

# Attempt to import the specific wrapper
try:
    from embpy.models.dna_models import EnformerWrapper

    ENFORMER_PYTORCH_INSTALLED = True
except ImportError:
    ENFORMER_PYTORCH_INSTALLED = False
    EnformerWrapper = None  # Define as None if import fails

# --- Test Setup ---

# Mark all tests in this file to be skipped if enformer-pytorch is not installed
pytestmark = pytest.mark.skipif(
    not ENFORMER_PYTORCH_INSTALLED, reason="enformer-pytorch package not found, skipping Enformer tests"
)

# TODO: Add a real suquence, generate the embedding using pretrain enformer and then verify against the wrapper


# Determine device for testing (prefer GPU if available)
def get_test_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Add MPS check if desired, but CPU is a safe fallback
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for Enformer tests: {TEST_DEVICE}")


# Fixture to load the model once for relevant tests
# Scope="module" means it runs once per test file execution
@pytest.fixture(scope="module")
def loaded_enformer_wrapper():
    """Fixture to provide a loaded EnformerWrapper instance."""
    if not EnformerWrapper:  # Should be caught by pytestmark, but as safeguard
        pytest.skip("EnformerWrapper class not available.")

    # Using the default model: "EleutherAI/enformer-official-rough"
    wrapper = EnformerWrapper()
    try:
        wrapper.load(device=TEST_DEVICE)
        # Check if model actually loaded
        if wrapper.model is None:
            pytest.skip(f"Failed to load Enformer model on {TEST_DEVICE}, skipping embedding tests.")
        return wrapper
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"Failed to load Enformer model due to error: {e}, skipping embedding tests.")


# --- Test Cases ---


def test_enformer_init():
    """Test if EnformerWrapper can be initialized."""
    if not EnformerWrapper:
        pytest.fail("EnformerWrapper class not available.")
    wrapper = EnformerWrapper()
    assert wrapper is not None
    assert wrapper.model_type == "dna"
    assert wrapper.model is None  # Model shouldn't be loaded on init
    assert wrapper.device is None
    assert wrapper.SEQUENCE_LENGTH == 196608
    # Updated to match the actual model output dimension
    assert wrapper.TRUNK_OUTPUT_DIM == 3072


def test_enformer_load(loaded_enformer_wrapper):
    """Test if the model loads correctly using the fixture."""
    # The fixture itself handles loading and skipping if failed
    assert loaded_enformer_wrapper is not None
    assert loaded_enformer_wrapper.model is not None
    assert loaded_enformer_wrapper.device == TEST_DEVICE
    assert isinstance(loaded_enformer_wrapper.model, torch.nn.Module)


def test_enformer_preprocess():
    """Test the _preprocess_sequence method."""
    if not EnformerWrapper:
        pytest.fail("EnformerWrapper class not available.")
    wrapper = EnformerWrapper()  # No need to load model for preprocessing test
    target_len = wrapper.SEQUENCE_LENGTH

    # 1. Sequence shorter than target length -> Should pad
    short_seq = "ACGT" * 10
    processed_short = wrapper._preprocess_sequence(short_seq)
    assert isinstance(processed_short, torch.Tensor)
    # Updated to match one-hot encoding shape (batch, seq_len, num_nucleotides)
    assert processed_short.shape == (1, target_len, 4)
    # Check if it's a float tensor (model expects float)
    assert processed_short.dtype == torch.float

    # 2. Sequence exactly target length -> No padding/truncation
    exact_seq = "N" * target_len
    processed_exact = wrapper._preprocess_sequence(exact_seq)
    assert isinstance(processed_exact, torch.Tensor)
    assert processed_exact.shape == (1, target_len, 4)
    # Check if it's a one-hot tensor for N (index 4)
    # All columns should be 0
    assert torch.all(processed_exact[:, :, :] == 0.0)

    # 3. Sequence longer than target length -> Should truncate from center
    long_seq = "ACGT" * (target_len // 4 + 100)  # Make it longer
    processed_long = wrapper._preprocess_sequence(long_seq)
    assert isinstance(processed_long, torch.Tensor)
    assert processed_long.shape == (1, target_len, 4)


def test_enformer_embed_single(loaded_enformer_wrapper):
    """Test embedding a single sequence."""
    wrapper = loaded_enformer_wrapper  # Get loaded model from fixture
    # Use a sequence that requires padding/truncation to test preprocessing integration
    test_seq = "A" * (wrapper.SEQUENCE_LENGTH // 2)

    embedding = wrapper.embed(input=test_seq, pooling_strategy="mean")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (wrapper.TRUNK_OUTPUT_DIM,)  # Check output dimension
    assert not np.isnan(embedding).any()  # Check for NaNs

    # Test max pooling
    embedding_max = wrapper.embed(test_seq, pooling_strategy="max")
    assert isinstance(embedding_max, np.ndarray)
    assert embedding_max.shape == (wrapper.TRUNK_OUTPUT_DIM,)
    assert not np.isnan(embedding_max).any()


def test_enformer_preprocess_trims_center():
    """Make sure long sequences are truncated around the center."""
    wrapper = EnformerWrapper()
    target_len = wrapper.SEQUENCE_LENGTH

    # build a long sequence whose content is easily distinguishable
    # e.g. first half all “A”, second half all “C”
    half = target_len + 50
    long_seq = "A" * half + "C" * half  # total length = 2*half

    # preprocess the long sequence
    processed_long = wrapper._preprocess_sequence(long_seq)
    assert isinstance(processed_long, torch.Tensor)
    assert processed_long.shape == (1, target_len, 4)

    # compute the expected center substring
    total_len = len(long_seq)
    start = (total_len - target_len) // 2
    center_seq = long_seq[start : start + target_len]

    # preprocess exactly that center substring
    processed_center = wrapper._preprocess_sequence(center_seq)
    assert isinstance(processed_center, torch.Tensor)
    assert processed_center.shape == (1, target_len, 4)

    # they should be identical one-hot encodings
    assert torch.equal(processed_long, processed_center)


def test_enformer_embed_batch(loaded_enformer_wrapper):
    """Test embedding a batch of sequences."""
    wrapper = loaded_enformer_wrapper
    target_len = wrapper.SEQUENCE_LENGTH
    batch_seqs = [
        "G" * target_len,  # Exact length
        "T" * (target_len // 2),  # Shorter
        "ACGT" * (target_len // 4 + 10),  # Longer
        "N" * target_len,  # Exact length Ns
    ]
    batch_size = len(batch_seqs)

    embeddings = wrapper.embed_batch(batch_seqs, pooling_strategy="mean")

    assert isinstance(embeddings, list)
    assert len(embeddings) == batch_size
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (wrapper.TRUNK_OUTPUT_DIM,)
        assert not np.isnan(emb).any()

    # Test max pooling on batch
    embeddings_max = wrapper.embed_batch(batch_seqs, pooling_strategy="max")
    assert isinstance(embeddings_max, list)
    assert len(embeddings_max) == batch_size
    for emb in embeddings_max:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (wrapper.TRUNK_OUTPUT_DIM,)
        assert not np.isnan(emb).any()


def test_enformer_invalid_pooling(loaded_enformer_wrapper):
    """Test that invalid pooling strategy raises ValueError."""
    wrapper = loaded_enformer_wrapper
    test_seq = "N" * wrapper.SEQUENCE_LENGTH
    with pytest.raises(ValueError, match="Invalid pooling strategy:"):
        wrapper.embed(test_seq, pooling_strategy="min")  # Minimum not supported

    with pytest.raises(ValueError, match="Invalid pooling strategy:"):
        wrapper.embed_batch([test_seq], pooling_strategy="some_invalid_pooling")
