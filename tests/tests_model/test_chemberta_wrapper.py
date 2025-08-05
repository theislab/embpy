import logging

import numpy as np
import pytest
import torch

# Attempt to import ChembertaWrapper and HF Transformers
try:
    from transformers import AutoModel, AutoTokenizer

    from embpy.models.molecule_models import ChembertaWrapper

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    ChembertaWrapper = None

# Skip tests if ChembertaWrapper or transformers aren't available
pytestmark = pytest.mark.skipif(
    not HF_AVAILABLE, reason="transformers or ChembertaWrapper not available, skipping ChemBERTa tests"
)


def get_test_device():
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for ChemBERTa tests: {TEST_DEVICE}")


def test_chemberta_init():
    """ChembertaWrapper initializes without loading model or tokenizer."""
    w = ChembertaWrapper()
    assert w.model is None
    assert w.tokenizer is None
    assert w.device is None


def test_chemberta_load_and_smoke():
    """ChembertaWrapper.load sets tokenizer, model, device, and eval mode; basic embed works."""
    w = ChembertaWrapper()
    w.load(TEST_DEVICE)
    assert w.tokenizer is not None
    assert w.model is not None
    assert w.device == TEST_DEVICE
    assert not w.model.training

    # retrieve hidden size
    hidden_dim = w.model.config.hidden_size

    # Test cls pooling if pooler_output exists
    try:
        emb_pooler = w.embed("CCO", pooling_strategy="cls", use_pooler=True)
        assert isinstance(emb_pooler, np.ndarray)
        assert emb_pooler.shape == (hidden_dim,)
        assert not np.isnan(emb_pooler).any()
    except ValueError:
        # some ChemBERTa variants may not expose pooler_output
        pass


@pytest.mark.parametrize(
    "pooling,use_pooler",
    [
        ("cls", False),
        ("mean", False),
        ("max", False),
        ("cls", True),  # pooler may be None for some models, handled in code
    ],
)
def test_chemberta_embed_single(pooling, use_pooler):
    """Embed single SMILES with different pooling strategies and optional pooler."""
    w = ChembertaWrapper()
    w.load(TEST_DEVICE)
    hidden_dim = w.model.config.hidden_size

    if use_pooler and not hasattr(w.model.config, "pooler_output"):
        pytest.skip("Model has no pooler_output")

    emb = w.embed("CCO", pooling_strategy=pooling, use_pooler=use_pooler)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.shape == (hidden_dim,)
    assert not np.isnan(emb).any()


def test_chemberta_embed_invalid_pooling():
    """Invalid pooling_strategy should raise ValueError."""
    w = ChembertaWrapper()
    w.load(TEST_DEVICE)
    with pytest.raises(ValueError, match="pooling_strategy"):
        w.embed("CCO", pooling_strategy="median")


def test_chemberta_preprocess_error():
    """_preprocess_smiles should error if load() not called."""
    w = ChembertaWrapper()
    with pytest.raises(RuntimeError, match="Call load"):
        w._preprocess_smiles("CCO")


def test_chemberta_embed_batch():
    """Embedding a batch of SMILES returns a list of correct embeddings."""
    w = ChembertaWrapper()
    w.load(TEST_DEVICE)
    hidden_dim = w.model.config.hidden_size
    smiles_list = ["CCO", "CCC", "CN(C)C"]
    embs = w.embed_batch(smiles_list, pooling_strategy="mean", use_pooler=False)
    assert isinstance(embs, list)
    assert len(embs) == len(smiles_list)
    for emb in embs:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (hidden_dim,)
        assert not np.isnan(emb).any()


@pytest.mark.parametrize("smiles", ["CCO", "C1=CC=CC=C1", "CN(C)C(=O)O"])
def test_chemberta_hf_backend_embedding_consistency(smiles):
    """
    Compare mask‐aware mean‐pooled embeddings from HF Transformers vs. ChembertaWrapper.

    Parameters
    ----------
    smiles : str
        A SMILES string to embed.

    Raises
    ------
    AssertionError
        If the wrapper’s output differs from direct HF output by >1e-6.
    """
    model_id = "seyonec/ChemBERTa-zinc-base-v1"

    # Direct HF Transformers path
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(TEST_DEVICE).eval()

    toks = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
    input_ids = toks["input_ids"].to(TEST_DEVICE)
    attention_mask = toks["attention_mask"].to(TEST_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

    mask = attention_mask.squeeze(0).unsqueeze(-1)  # (seq_len,1)
    summed = (last_hidden * mask).sum(dim=0)  # (hidden_dim,)
    direct_emb = (summed / mask.sum(dim=0)).cpu().numpy()  # mask‐aware mean

    # Through ChembertaWrapper
    wrapper = ChembertaWrapper(model_path_or_name=model_id)
    wrapper.load(TEST_DEVICE)
    wrapper_emb = wrapper.embed(
        input=smiles,
        pooling_strategy="mean",
        use_pooler=False,
    )

    assert isinstance(wrapper_emb, np.ndarray)
    assert wrapper_emb.shape == direct_emb.shape, "Shape mismatch"
    assert np.allclose(wrapper_emb, direct_emb, atol=1e-6), f"Mean‐pooled embedding mismatch for SMILES={smiles}"


@pytest.mark.parametrize("smiles", ["CCO", "C1=CC=CC=C1"])
def test_chemberta_hf_backend_pooler_consistency(smiles):
    """
    Compare pooler_output from HF Transformers vs. ChembertaWrapper when use_pooler=True.

    Parameters
    ----------
    smiles : str
        A SMILES string to embed.

    Raises
    ------
    AssertionError
        If the wrapper’s pooler_output differs from direct HF by >1e-6.
    """
    model_id = "seyonec/ChemBERTa-zinc-base-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(TEST_DEVICE).eval()

    toks = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
    input_ids = toks["input_ids"].to(TEST_DEVICE)
    attention_mask = toks["attention_mask"].to(TEST_DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(outputs, "pooler_output"), "HF model has no pooler_output"
        direct_pooler = outputs.pooler_output.squeeze(0).cpu().numpy()

    wrapper_pooler = ChembertaWrapper(model_path_or_name=model_id)
    wrapper_pooler.load(TEST_DEVICE)
    emb = wrapper_pooler.embed(
        input=smiles,
        pooling_strategy="cls",  # ignored when use_pooler=True
        use_pooler=True,
    )

    assert isinstance(emb, np.ndarray)
    assert emb.shape == direct_pooler.shape, "Pooler shape mismatch"
    assert np.allclose(emb, direct_pooler, atol=1e-6), f"Pooler output mismatch for SMILES={smiles}"
