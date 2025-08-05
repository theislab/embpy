import logging

import numpy as np
import pytest
import torch

# Attempt to import ESMCWrapper and the raw ESMC SDK
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    from embpy.models.protein_models import ESMCWrapper

    ESMC_SDK_AVAILABLE = True
except ImportError:
    ESMC_SDK_AVAILABLE = False
    ESMCWrapper = None

# Skip all tests in this file if the ESMC SDK or wrapper not available
pytestmark = pytest.mark.skipif(
    not ESMC_SDK_AVAILABLE, reason="ESMC SDK or ESMCWrapper not available, skipping ESMC tests"
)


def get_test_device():
    """Prefer GPU if available, else CPU (or MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for ESMC tests: {TEST_DEVICE}")

# Two common ESMC checkpoints
ESMC_MODELS = {
    "esmc_300m": "esmc_300m",
    "esmc_600m": "esmc_600m",
}


@pytest.mark.parametrize("short_name, model_id", list(ESMC_MODELS.items()))
def test_esmc_load_all_sizes(short_name, model_id):
    """
    Parametrized test that each ESMCWrapper variant can be initialized and loaded.
    """
    wrapper = ESMCWrapper(model_path_or_name=model_id)
    assert wrapper.model_name == model_id
    assert wrapper.client is None
    assert wrapper.device is None

    wrapper.load(device=TEST_DEVICE)

    assert wrapper.client is not None, f"{short_name}: client failed to load"
    assert wrapper.device == TEST_DEVICE
    # SDK client should be in eval mode
    assert not wrapper.client.training, f"{short_name}: client should be in eval() mode"


def test_esmc_load_invalid_name():
    """Empty or invalid model name should raise on load()."""
    w_empty = ESMCWrapper(model_path_or_name="")
    with pytest.raises(RuntimeError, match="ESMC SDK not installed|Could not load ESMC client"):
        w_empty.load(device=TEST_DEVICE)

    w_bad = ESMCWrapper(model_path_or_name="no_such_model")
    # first call might raise from the SDK or our wrapper
    with pytest.raises(RuntimeError, match="Could not load ESMC client"):
        w_bad.load(device=TEST_DEVICE)


def test_esmc_embed_without_load():
    """Calling embed() before load() should raise RuntimeError."""
    w = ESMCWrapper(model_path_or_name="esmc_300m")
    with pytest.raises(RuntimeError, match="client not loaded"):
        w.embed("AAAAA", pooling_strategy="mean")


@pytest.mark.parametrize("pooling", ["mean", "max", "cls"])
def test_esmc_embed_single(pooling):
    """
    Embed a single sequence with different pooling strategies.
    """
    w = ESMCWrapper(model_path_or_name="esmc_300m")
    w.load(TEST_DEVICE)

    seq = "MTEYK"  # short dummy sequence
    emb = w.embed(seq, pooling_strategy=pooling)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert not np.isnan(emb).any()


def test_esmc_invalid_pooling():
    """Invalid pooling_strategy should raise ValueError."""
    w = ESMCWrapper(model_path_or_name="esmc_300m")
    w.load(TEST_DEVICE)
    with pytest.raises(ValueError, match="Invalid pooling"):
        w.embed("AAAAA", pooling_strategy="median")


def test_esmc_embed_batch():
    """
    Embedding a batch of sequences returns a list of correct embeddings.
    """
    w = ESMCWrapper(model_path_or_name="esmc_300m")
    w.load(TEST_DEVICE)

    seqs = ["MTEYK", "GGGGG", "AAAAA"]
    embs = w.embed_batch(seqs, pooling_strategy="mean")
    assert isinstance(embs, list)
    assert len(embs) == len(seqs)
    for emb in embs:
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert not np.isnan(emb).any()


@pytest.mark.parametrize("pooling", ["mean", "max", "cls"])
def test_esmc_direct_vs_wrapper(pooling):
    """
    Compare embeddings from:
      1) Raw ESMC SDK
      2) ESMCWrapper.embed(...)
    """
    # load raw client
    raw_client = ESMC.from_pretrained("esmc_300m").to(TEST_DEVICE).eval()
    # load wrapper
    w = ESMCWrapper(model_path_or_name="esmc_300m")
    w.load(TEST_DEVICE)

    seq = "MTEYKLVVVG"  # dummy test sequence
    prot = ESMProtein(sequence=seq)
    tensor = raw_client.encode(prot)
    logits_o = raw_client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
    raw_emb = logits_o.embeddings
    if raw_emb.dim() == 3 and raw_emb.shape[0] == 1:
        raw_emb = raw_emb.squeeze(0)  # (L, hidden_dim)

    # pool raw
    if pooling == "cls":
        direct = raw_emb[0]
    elif pooling == "max":
        direct = raw_emb.max(dim=0)[0]
    else:  # mean
        direct = raw_emb.mean(dim=0)
    direct = direct.cpu().numpy()

    # via wrapper
    wrapper_emb = w.embed(seq, pooling_strategy=pooling)

    # compare
    assert direct.shape == wrapper_emb.shape
    assert np.allclose(wrapper_emb, direct, atol=1e-6), f"Pooling={pooling} mismatch"
