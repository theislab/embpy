import logging

import numpy as np
import pytest
import torch

# Attempt to import TextLLMWrapper and transformers
try:
    from transformers import AutoModel, AutoTokenizer

    from embpy.models.text_models import TextLLMWrapper

    TRANSFORMERS_INSTALLED = True
except ImportError:
    TRANSFORMERS_INSTALLED = False
    TextLLMWrapper = None

# Skip all tests in this file if the TextLLMWrapper or transformers are not available
pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_INSTALLED, reason="transformers or TextLLMWrapper not available, skipping text model tests"
)


def get_test_device():
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for TextLLM tests: {TEST_DEVICE}")

# Test models with different characteristics
TEXT_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert_mini": "prajjwal1/bert-mini",
    "roberta": "roberta-base",
}


def test_textllm_init():
    """TextLLMWrapper initializes without loading model or tokenizer."""
    wrapper = TextLLMWrapper()
    assert wrapper.model is None
    assert wrapper.tokenizer is None
    assert wrapper.device is None
    assert wrapper.max_length is None
    assert wrapper.cls_token_position == 0


def test_textllm_init_with_params():
    """TextLLMWrapper initializes with custom parameters."""
    wrapper = TextLLMWrapper(model_path_or_name="bert-base-uncased", max_length=256, cls_token_position=1)
    assert wrapper.model_name == "bert-base-uncased"
    assert wrapper.max_length == 256
    assert wrapper.cls_token_position == 1


def test_textllm_load_and_smoke():
    """TextLLMWrapper.load sets tokenizer, model, device, and eval mode; basic embed works."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)
    assert wrapper.tokenizer is not None
    assert wrapper.model is not None
    assert wrapper.device == TEST_DEVICE
    assert not wrapper.model.training
    assert wrapper.max_length is not None
    assert wrapper.max_length > 0

    # Smoke test: basic embedding
    text = "Hello world"
    embedding = wrapper.embed(text, pooling_strategy="mean")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert not np.isnan(embedding).any()


def test_textllm_load_invalid_name():
    """Passing an invalid model name should raise appropriate errors."""
    wrapper = TextLLMWrapper(model_path_or_name="this-model-does-not-exist")
    with pytest.raises(RuntimeError, match="Could not load text model"):
        wrapper.load(device=TEST_DEVICE)


def test_textllm_embed_before_load():
    """Calling embed before load should raise RuntimeError."""
    wrapper = TextLLMWrapper()
    with pytest.raises(RuntimeError, match="Text model not loaded"):
        wrapper.embed("Hello world")


def test_textllm_preprocess_before_load():
    """Calling _preprocess_text_hf before load should raise RuntimeError."""
    wrapper = TextLLMWrapper()
    with pytest.raises(RuntimeError, match="Hugging Face Tokenizer not loaded"):
        wrapper._preprocess_text_hf("Hello world")


@pytest.mark.parametrize("pooling_strategy", ["mean", "max", "cls"])
def test_textllm_embed_single(pooling_strategy):
    """Embed single text with different pooling strategies."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)
    hidden_dim = wrapper.model.config.hidden_size

    text = "The quick brown fox jumps over the lazy dog"
    embedding = wrapper.embed(text, pooling_strategy=pooling_strategy)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape == (hidden_dim,)
    assert not np.isnan(embedding).any()


def test_textllm_embed_invalid_pooling():
    """Invalid pooling_strategy should raise ValueError."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)
    with pytest.raises(ValueError, match="Invalid pooling strategy"):
        wrapper.embed("Hello world", pooling_strategy="invalid")


def test_textllm_embed_batch():
    """Embedding a batch of texts returns a list of correct embeddings."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)
    hidden_dim = wrapper.model.config.hidden_size

    texts = ["Hello world", "The quick brown fox", "Machine learning is fascinating"]
    embeddings = wrapper.embed_batch(texts, pooling_strategy="mean")

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (hidden_dim,)
        assert not np.isnan(emb).any()


def test_textllm_embed_batch_empty():
    """Embedding empty batch should return empty list."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)
    embeddings = wrapper.embed_batch([])
    assert embeddings == []


@pytest.mark.parametrize("model_name", list(TEXT_MODELS.values()))
def test_textllm_different_models(model_name):
    """Test that different text models can be loaded and used."""
    wrapper = TextLLMWrapper(model_path_or_name=model_name)
    wrapper.load(TEST_DEVICE)

    text = "This is a test sentence"
    embedding = wrapper.embed(text, pooling_strategy="mean")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert not np.isnan(embedding).any()


def test_textllm_max_length_configuration():
    """Test that max_length configuration works correctly."""
    # Test with custom max_length
    wrapper = TextLLMWrapper(max_length=128)
    wrapper.load(TEST_DEVICE)
    assert wrapper.max_length == 128
    assert wrapper.tokenizer.model_max_length == 128

    # Test with very long text (should be truncated)
    long_text = "word " * 200  # Much longer than 128 tokens
    embedding = wrapper.embed(long_text, pooling_strategy="mean")
    assert isinstance(embedding, np.ndarray)
    assert not np.isnan(embedding).any()


def test_textllm_cls_token_position():
    """Test that CLS token position configuration works."""
    wrapper = TextLLMWrapper(cls_token_position=0)
    wrapper.load(TEST_DEVICE)

    text = "Test sentence"
    embedding = wrapper.embed(text, pooling_strategy="cls")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert not np.isnan(embedding).any()


def test_textllm_target_layer():
    """Test that target_layer parameter works for extracting specific layer embeddings."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)

    text = "Test sentence"
    # Extract from layer -2 (second to last layer)
    embedding = wrapper.embed(text, pooling_strategy="mean", target_layer=-2)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert not np.isnan(embedding).any()


@pytest.mark.parametrize("text", ["Hello world", "The quick brown fox jumps over the lazy dog"])
def test_textllm_hf_backend_embedding_consistency(text):
    """
    Compare mask-aware mean-pooled embeddings from HF Transformers vs. TextLLMWrapper.

    Parameters
    ----------
    text : str
        A text string to embed.

    Raises
    ------
    AssertionError
        If the wrapper's embedding differs from direct HF by >1e-6.
    """
    model_id = "distilbert-base-uncased"

    # 1) Direct HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(TEST_DEVICE).eval()

    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = toks["input_ids"].to(TEST_DEVICE)
    attention_mask = toks["attention_mask"].to(TEST_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    # Mask-aware mean pooling
    mask = attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
    sum_embeddings = torch.sum(last_hidden * mask, dim=1)
    sum_mask = torch.sum(mask, dim=1)
    direct_emb = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()

    # 2) Via wrapper
    wrapper = TextLLMWrapper(model_path_or_name=model_id)
    wrapper.load(device=TEST_DEVICE)
    wrapper_emb = wrapper.embed(input=text, pooling_strategy="mean")

    # Compare shapes and values
    assert isinstance(wrapper_emb, np.ndarray)
    assert wrapper_emb.shape == direct_emb.shape, "Shape mismatch"
    assert np.allclose(wrapper_emb, direct_emb, atol=1e-6), (
        f"Wrapper embedding differs from direct HF embedding for text='{text}'"
    )


def test_textllm_cls_pooling_consistency():
    """
    Compare CLS token embeddings from HF Transformers vs. TextLLMWrapper.
    """
    model_id = "distilbert-base-uncased"
    text = "Hello world"

    # 1) Direct HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(TEST_DEVICE).eval()

    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = toks["input_ids"].to(TEST_DEVICE)
    attention_mask = toks["attention_mask"].to(TEST_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    # CLS token is at position 0
    direct_cls = last_hidden[0, 0, :].cpu().numpy()

    # 2) Via wrapper
    wrapper = TextLLMWrapper(model_path_or_name=model_id, cls_token_position=0)
    wrapper.load(device=TEST_DEVICE)
    wrapper_cls = wrapper.embed(input=text, pooling_strategy="cls")

    # Compare shapes and values
    assert isinstance(wrapper_cls, np.ndarray)
    assert wrapper_cls.shape == direct_cls.shape, "CLS shape mismatch"
    assert np.allclose(wrapper_cls, direct_cls, atol=1e-6), "CLS embedding mismatch"


def test_textllm_batch_vs_single_consistency():
    """Test that batch embedding gives same results as individual embeddings."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)

    texts = ["Hello world", "The quick brown fox"]

    # Individual embeddings
    individual_embs = [wrapper.embed(text, pooling_strategy="mean") for text in texts]

    # Batch embedding
    batch_embs = wrapper.embed_batch(texts, pooling_strategy="mean")

    assert len(individual_embs) == len(batch_embs)
    for ind_emb, batch_emb in zip(individual_embs, batch_embs, strict=False):
        assert np.allclose(ind_emb, batch_emb, atol=1e-6), "Batch vs individual embedding mismatch"


@pytest.mark.parametrize("pooling_strategy", ["mean", "max", "cls"])
def test_textllm_batch_different_pooling(pooling_strategy):
    """Test batch embedding with different pooling strategies."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)

    texts = ["Short text", "This is a much longer text that should test the attention mask handling"]
    embeddings = wrapper.embed_batch(texts, pooling_strategy=pooling_strategy)

    assert len(embeddings) == 2
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert not np.isnan(emb).any()


def test_textllm_memory_cleanup():
    """Test that memory is properly cleaned up after embedding."""
    wrapper = TextLLMWrapper()
    wrapper.load(TEST_DEVICE)

    # Get initial memory if CUDA
    if TEST_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

    # Embed some text
    text = "This is a test for memory cleanup"
    embedding = wrapper.embed(text, pooling_strategy="mean")

    # Check memory is cleaned up
    if TEST_DEVICE.type == "cuda":
        final_memory = torch.cuda.memory_allocated()
        # Memory should be close to initial (allowing for some variance)
        assert final_memory <= initial_memory + 1024 * 1024  # 1MB tolerance

    assert isinstance(embedding, np.ndarray)
