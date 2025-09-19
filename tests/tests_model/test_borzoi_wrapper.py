import logging

import numpy as np
import pytest
import torch
from borzoi_pytorch import Borzoi

# TODO: Add a real suquence, generate the embedding using pretrain borzoi and then verify against the wrapper

# Attempt to import BorzoiWrapper
try:
    from embpy.models.dna_models import BorzoiWrapper

    BORZOI_PYTORCH_INSTALLED = True
except ImportError:
    BORZOI_PYTORCH_INSTALLED = False
    BorzoiWrapper = None

# Skip entire file if Borzoi isn’t installed
pytestmark = pytest.mark.skipif(
    not BORZOI_PYTORCH_INSTALLED, reason="borzoi_pytorch package not found, skipping Borzoi tests"
)


# pick device
def get_test_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for Borzoi tests: {TEST_DEVICE}")


@pytest.fixture(scope="module")
def loaded_borzoi_wrapper():
    """Load Borzoi once per module."""
    if not BorzoiWrapper:
        pytest.skip("BorzoiWrapper not available")
    wrapper = BorzoiWrapper()
    try:
        wrapper.load(device=TEST_DEVICE)
        if wrapper.model is None:
            pytest.skip(f"Failed to load Borzoi model on {TEST_DEVICE}")
        return wrapper
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Could not load Borzoi model: {e}")


def test_borzoi_init():
    """Can initialize without loading."""
    if not BorzoiWrapper:
        pytest.fail("BorzoiWrapper class not available.")
    w = BorzoiWrapper()
    assert w.model is None
    assert w.device is None
    assert w.model_type == "dna"
    assert w.SEQUENCE_LENGTH == 524_288
    assert w.NUM_CHANNELS == 4
    assert hasattr(w, "ALPHABET_MAP")


def test_borzoi_load(loaded_borzoi_wrapper):
    """Model loads onto device and has correct TRUNK_OUTPUT_DIM."""
    w = loaded_borzoi_wrapper
    assert w.model is not None
    assert isinstance(w.model, torch.nn.Module)
    assert w.device == TEST_DEVICE
    # hidden_dim should be set from config or default 512
    assert isinstance(w.TRUNK_OUTPUT_DIM, int)
    assert w.available_pooling_strategies == ["mean", "max", "median"]


def test_borzoi_preprocess_shapes_and_dtype():
    """_preprocess_sequence yields correct shape/dtype for short, exact, long."""
    w = BorzoiWrapper()
    L = w.SEQUENCE_LENGTH

    # short -> pad
    short = "ACGT" * 10
    t_short = w._preprocess_sequence(short)
    assert isinstance(t_short, torch.Tensor)
    assert t_short.shape == (1, w.NUM_CHANNELS, L)
    assert t_short.dtype == torch.float32

    # long -> center-crop only
    long = "ACGT" * (L // 4 + 100)
    t_long = w._preprocess_sequence(long)
    assert t_long.shape == (1, w.NUM_CHANNELS, L)


def test_borzoi_preprocess_trims_center():
    """Ensure center cropping of long sequences."""
    w = BorzoiWrapper()
    L = w.SEQUENCE_LENGTH
    half = L + 50
    seq = "A" * half + "C" * half  # total = 2*half
    out_full = w._preprocess_sequence(seq)
    assert out_full.shape == (1, 4, L)

    start = (len(seq) - L) // 2
    center_seq = seq[start : start + L]
    out_center = w._preprocess_sequence(center_seq)
    assert torch.equal(out_full, out_center)


def test_borzoi_embed_single(loaded_borzoi_wrapper):
    """Embed a single sequence (mean & max) and check output shape + no NaNs."""
    w = loaded_borzoi_wrapper
    half_seq = "A" * (w.SEQUENCE_LENGTH // 2)
    emb_mean = w.embed(input=half_seq, pooling_strategy="mean")
    assert isinstance(emb_mean, np.ndarray)
    assert emb_mean.shape == (w.TRUNK_OUTPUT_DIM,)
    assert not np.isnan(emb_mean).any()

    emb_max = w.embed(input=half_seq, pooling_strategy="max")
    assert emb_max.shape == (w.TRUNK_OUTPUT_DIM,)
    assert not np.isnan(emb_max).any()


def test_borzoi_embed_batch(loaded_borzoi_wrapper):
    """Embed a batch of sequences (mean & max)."""
    w = loaded_borzoi_wrapper
    L = w.SEQUENCE_LENGTH
    seqs = [
        "G" * L,
        "T" * (L // 2),
        "ACGT" * (L // 4 + 10),
        "N" * L,
    ]
    out_mean = w.embed_batch(seqs, pooling_strategy="mean")
    assert isinstance(out_mean, list) and len(out_mean) == len(seqs)
    for emb in out_mean:
        assert emb.shape == (w.TRUNK_OUTPUT_DIM,)
        assert not np.isnan(emb).any()

    out_max = w.embed_batch(seqs, pooling_strategy="max")
    assert len(out_max) == len(seqs)
    for emb in out_max:
        assert emb.shape == (w.TRUNK_OUTPUT_DIM,)
        assert not np.isnan(emb).any()


def test_borzoi_invalid_pooling(loaded_borzoi_wrapper):
    """Invalid pooling strategies should raise ValueError."""
    w = loaded_borzoi_wrapper
    seq = "A" * w.SEQUENCE_LENGTH
    with pytest.raises(ValueError, match="Invalid pooling"):
        w.embed(input=seq, pooling_strategy="median")
    with pytest.raises(ValueError, match="Invalid pooling"):
        w.embed_batch([seq], pooling_strategy="unsupported")


def test_borzoi_direct_vs_wrapper(loaded_borzoi_wrapper):
    """
    Compare mean‐pooled embeddings from:
      1) the BorzoiWrapper.embed(...)
      2) the raw Borzoi.from_pretrained(...).get_embs_after_crop(...)
    for a fixed test sequence.
    """
    w = loaded_borzoi_wrapper
    L = w.SEQUENCE_LENGTH

    # create a test sequence exactly L bases long (no padding/truncation)
    seq = ("ACGT" * (L // 4))[:L]

    # 1) wrapper result
    wrapper_mean = w.embed(input=seq, pooling_strategy="mean")
    assert isinstance(wrapper_mean, np.ndarray)

    # 2) direct Borzoi API
    # preprocess into one‐hot tensor and move to device
    oh = w._preprocess_sequence(seq).to(w.device)  # (1,4,L)
    # load a fresh Borzoi instance
    direct_model = Borzoi.from_pretrained(w.model_name).to(w.device).eval()
    with torch.no_grad():
        embs = direct_model.get_embs_after_crop(oh)  # (1, hidden_dim, bins)
    trunk = embs.squeeze(0)  # (hidden_dim, bins)
    direct_mean = trunk.mean(dim=1).cpu().numpy()  # (hidden_dim,)

    # verify shapes match
    assert wrapper_mean.shape == direct_mean.shape, "Shape mismatch between wrapper and direct model"

    # compare numerical values
    assert np.allclose(wrapper_mean, direct_mean, atol=1e-6), (
        f"Max abs diff = {np.max(np.abs(wrapper_mean - direct_mean)):.3e}"
    )
