import logging

import numpy as np
import pytest
import torch

# Attempt to import MolformerWrapper and HF Transformers
try:
    from transformers import AutoModel, AutoTokenizer

    from embpy.models.molecule_models import MolformerWrapper

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    MolformerWrapper = None

# Skip tests if MolformerWrapper or transformers aren't available
pytestmark = pytest.mark.skipif(
    not HF_AVAILABLE, reason="transformers or MolformerWrapper not available, skipping MolFormer tests"
)

# TODO: Verify if everything makes sense, every test and so on


def get_test_device():
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TEST_DEVICE = get_test_device()
logging.info(f"Using device for MolFormer tests: {TEST_DEVICE}")


def test_molformer_init():
    """MolformerWrapper initializes without loading model or tokenizer."""
    w = MolformerWrapper()
    assert w.model is None
    assert w.tokenizer is None
    assert w.device is None


def test_molformer_load_and_smoke():
    """
    MolformerWrapper.load should set tokenizer, model, device, and eval mode.
    Smoke-test that a basic cls embed works.
    """
    w = MolformerWrapper()
    w.load(TEST_DEVICE)
    assert w.tokenizer is not None
    assert w.model is not None
    assert w.device == TEST_DEVICE
    assert not w.model.training

    hidden_dim = w.model.config.hidden_size
    # cls pooling
    emb_cls = w.embed("CCO", pooling_strategy="cls")
    assert isinstance(emb_cls, np.ndarray)
    assert emb_cls.shape == (hidden_dim,)
    assert not np.isnan(emb_cls).any()


@pytest.mark.parametrize("pooling", ["cls", "mean", "max"])
def test_molformer_embed_single(pooling):
    """
    Embed a single SMILES with various pooling strategies.

    Parameters
    ----------
    pooling : str
        One of "cls", "mean", "max".

    Raises
    ------
    AssertionError
        If the embedding has wrong shape or contains NaNs.
    """
    w = MolformerWrapper()
    w.load(TEST_DEVICE)
    hidden_dim = w.model.config.hidden_size

    emb = w.embed("CCO", pooling_strategy=pooling)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.shape == (hidden_dim,)
    assert not np.isnan(emb).any()


def test_molformer_embed_invalid_pooling():
    """Invalid pooling string should raise ValueError."""
    w = MolformerWrapper()
    w.load(TEST_DEVICE)
    with pytest.raises(ValueError, match="Invalid pooling"):
        w.embed("CCO", pooling_strategy="median")


def test_molformer_preprocess_error():
    """_preprocess_smiles should error if load() not called."""
    w = MolformerWrapper()
    with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
        w._preprocess_smiles("CCO")


def test_molformer_embed_batch():
    """
    Embedding a batch of SMILES returns correct list of embeddings.

    Raises
    ------
    AssertionError
        If the list length or embedding shapes are incorrect.
    """
    w = MolformerWrapper()
    w.load(TEST_DEVICE)
    hidden_dim = w.model.config.hidden_size
    smiles_list = ["CCO", "CCC", "CN(C)C"]
    embs = w.embed_batch(smiles_list, pooling_strategy="mean")
    assert isinstance(embs, list)
    assert len(embs) == len(smiles_list)
    for emb in embs:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (hidden_dim,)
        assert not np.isnan(emb).any()


@pytest.mark.parametrize("smiles", ["CCO", "C1=CC=CC=C1"])
def test_molformer_hf_backend_mean_consistency(smiles):
    """
    Compare mean-pooled embeddings from HF Transformers vs. MolformerWrapper.

    Parameters
    ----------
    smiles : str
        SMILES string to embed.

    Raises
    ------
    AssertionError
        If the wrapper’s mean-pooled embeddings differ from direct HF.
    """
    model_id = "ibm/MoLFormer-XL-both-10pct"

    # Direct HF path
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(TEST_DEVICE).eval()

    toks = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
    toks = {k: v.to(TEST_DEVICE) for k, v in toks.items()}

    with torch.no_grad():
        out = model(**toks)
        last_hidden = out.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

    # mask-aware mean pooling
    mask = toks["attention_mask"].squeeze(0).unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=0)
    direct_emb = (summed / mask.sum(dim=0)).cpu().numpy()

    # Wrapper path
    w = MolformerWrapper(model_path_or_name=model_id)
    w.load(TEST_DEVICE)
    wrapper_emb = w.embed(smiles, pooling_strategy="mean")

    assert isinstance(wrapper_emb, np.ndarray)
    assert wrapper_emb.shape == direct_emb.shape
    assert np.allclose(wrapper_emb, direct_emb, atol=1e-6), f"Mean-pooled mismatch for SMILES={smiles}"


@pytest.mark.parametrize("smiles", ["CCO", "C1=CC=CC=C1"])
def test_molformer_hf_backend_cls_consistency(smiles):
    """
    Compare cls-pooled embeddings from HF Transformers vs. MolformerWrapper.

    Parameters
    ----------
    smiles : str
        SMILES string to embed.

    Raises
    ------
    AssertionError
        If the wrapper’s cls-pooled embeddings differ from direct HF.
    """
    model_id = "ibm/MoLFormer-XL-both-10pct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(TEST_DEVICE).eval()

    toks = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
    toks = {k: v.to(TEST_DEVICE) for k, v in toks.items()}

    with torch.no_grad():
        out = model(**toks)
        direct_cls = out.pooler_output.squeeze(0).cpu().numpy()

    w = MolformerWrapper(model_path_or_name=model_id)
    w.load(TEST_DEVICE)
    wrapper_cls = w.embed(smiles, pooling_strategy="cls")

    assert isinstance(wrapper_cls, np.ndarray)
    assert wrapper_cls.shape == direct_cls.shape
    assert np.allclose(wrapper_cls, direct_cls, atol=1e-6), f"CLS-pooled mismatch for SMILES={smiles}"
