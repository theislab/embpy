import logging

import numpy as np
import pytest
import torch

# Attempt to import ESM2Wrapper and transformers
try:
    from transformers import AutoModel, AutoTokenizer

    from embpy.models.protein_models import ESM2Wrapper

    TRANSFORMERS_INSTALLED = True
except ImportError:
    TRANSFORMERS_INSTALLED = False
    ESM2Wrapper = None

# Skip all tests in this file if the ESM2Wrapper or transformers are not available
pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_INSTALLED, reason="transformers or ESM2Wrapper not available, skipping ESM2 tests"
)


def get_test_device():
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for ESM2 tests: {TEST_DEVICE}")

# Mapping of shorthand names to Hugging Face model identifiers
ESM2_MODELS = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
}


@pytest.mark.parametrize("short_name, hf_model_id", list(ESM2_MODELS.items()))
def test_esm2_load_all_sizes(short_name, hf_model_id):
    """
    Parametrized test that each ESM2Wrapper variant (8M, 35M, 150M, 650M, 3B)
    can be initialized with the correct HF model name and successfully loaded.
    """
    wrapper = ESM2Wrapper(model_path_or_name=hf_model_id)
    assert wrapper.model_name == hf_model_id
    assert wrapper.model is None
    assert wrapper.device is None

    wrapper.load(device=TEST_DEVICE)

    assert wrapper.tokenizer is not None, f"{short_name}: tokenizer failed to load"
    assert wrapper.model is not None, f"{short_name}: model failed to load"
    assert wrapper.device == TEST_DEVICE
    assert not wrapper.model.training, f"{short_name}: model should be in eval() mode"

    # smoke‐test: tokenize a short sequence
    seq = "M" * 10
    toks = wrapper._preprocess_sequence(seq)
    assert toks["input_ids"].device == TEST_DEVICE

    # end‐to‐end forward pass
    emb = wrapper.embed(input=seq, pooling_strategy="mean")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert not np.isnan(emb).any()


def test_esm2_load_invalid_name():
    """Passing an empty or invalid model name should raise appropriate errors."""
    w_empty = ESM2Wrapper(model_path_or_name="")
    with pytest.raises(ValueError, match="model_path_or_name must be provided"):
        w_empty.load(device=TEST_DEVICE)

    w_bad = ESM2Wrapper(model_path_or_name="this-model-does-not-exist")
    with pytest.raises(RuntimeError, match="Could not load ESM2 model"):
        w_bad.load(device=TEST_DEVICE)


# TODO: Further verification of the architecture of ESM2 to make sure we are extracting the best embeddings


@pytest.mark.parametrize("short_name, hf_model_id", list(ESM2_MODELS.items()))
def test_esm2_hf_backend_embedding_consistency(short_name, hf_model_id):
    """
    Compare, for each model size, the embedding from:
      1) Direct HF Transformers (AutoTokenizer + AutoModel)
      2) ESM2Wrapper with backend='hf'
    """
    seq = "MTEYKLVVVG"

    # 1) Direct HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id).to(TEST_DEVICE).eval()

    toks = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    input_ids = toks["input_ids"].to(TEST_DEVICE)
    attention_mask = toks.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(TEST_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    direct_emb = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()

    # 2) Via wrapper
    wrapper = ESM2Wrapper(model_path_or_name=hf_model_id, backend="hf")
    wrapper.load(device=TEST_DEVICE)
    wrapper_emb = wrapper.embed(input=seq, pooling_strategy="mean")

    # Compare shapes and values
    assert isinstance(wrapper_emb, np.ndarray)
    assert wrapper_emb.shape == direct_emb.shape, f"{short_name}: shape mismatch"
    assert np.allclose(wrapper_emb, direct_emb, atol=1e-6), (
        f"{short_name}: wrapper embedding differs from direct HF embedding"
    )
