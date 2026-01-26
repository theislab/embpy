# Placeholder for small molecule models (e.g., ChemBERTa, MolFormer)
import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .base import BaseModelWrapper

# TODO: If we use the transformers library we can access to other hidden states, since we are using it here for molecules we can add this option to the package


class ChembertaWrapper(BaseModelWrapper):
    """
    Wrapper for ChemBERTa models for SMILES strings.

    Implements:
      - embed(smiles, pooling_strategy, use_pooler) → np.ndarray
      - embed_batch([...], pooling_strategy, use_pooler) → list[np.ndarray]

    Pooling strategies:
      • cls  → take token 0 from last_hidden_state
      • mean → mean over all tokens (mask‐aware)
      • max  → max  over all tokens (mask‐aware)

    You can also set `use_pooler=True` to get `outputs.pooler_output` (if your model has one).
    """

    model_type = "molecule"
    available_pooling_strategies = ["cls", "mean", "max"]

    def __init__(self, model_path_or_name: str = "DeepChem/ChemBERTa-77M-MTR", **kwargs):
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModel | None = None
        self.device: torch.device | None = None
        self.max_len: int | None = None

    def load(self, device: torch.device):
        """Load the dataset"""
        if self.model is not None:
            return
        logging.info(f"Loading ChemBERTa '{self.model_name}'…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        self.device = device

        # derive a safe max sequence length (RoBERTa-style ≈512)
        mmax = getattr(self.model.config, "max_position_embeddings", None)
        if mmax is None or mmax > 512:
            mmax = 512
        self.max_len = int(mmax)
        # make tokenizer aware (some HF tokenizers have "no max" by default)
        self.tokenizer.model_max_length = self.max_len
        logging.info(f"Using model_max_length={self.max_len}")

    def _token_length(self, smiles: str) -> int:
        """Tokenized length incl. special tokens, without truncation/padding."""
        toks = self.tokenizer(
            smiles,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(toks["input_ids"])

    def _preprocess_smiles(self, smiles: str) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Call load() first.")
        # we’ll still pass max_length for safety even though we pre-check length
        return self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        use_pooler: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | None:  # << allow None for skipped items
        """Embeddings function, which generates the embeddings for a specific provided model"""
        if self.model is None or self.device is None:
            raise RuntimeError("Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"pooling_strategy must be one of {self.available_pooling_strategies}")

        # --- Skip too-long SMILES ---
        L = self._token_length(input)
        if L > self.max_len:
            logging.warning(f"Skipping SMILES ({L} tokens > max {self.max_len}): {input[:80]}...")
            return None

        # 1) tokenize & move to device
        enc = self._preprocess_smiles(input)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 2) forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
            pooler_out = getattr(outputs, "pooler_output", None)

        # 3) use pooler if requested
        if use_pooler and pooler_out is not None:
            return pooler_out[0].cpu().numpy()

        # 4) squeeze batch dim
        hidden = last_hidden.squeeze(0)  # (seq_len, hidden_dim)

        # 5) masking-aware pooling
        if pooling_strategy == "cls":
            vec = hidden[0]
        else:
            mask = attention_mask.squeeze(0).unsqueeze(-1)  # (seq_len, 1)
            if pooling_strategy == "mean":
                summed = (hidden * mask).sum(dim=0)
                vec = summed / mask.sum(dim=0).clamp(min=1)  # safe denom
            else:  # 'max'
                neg_inf = torch.finfo(hidden.dtype).min
                masked = hidden.masked_fill(mask == 0, neg_inf)
                vec = masked.max(dim=0).values

        return vec.cpu().numpy()

    def embed_batch(
        self,
        input: list[str],
        pooling_strategy: str = "mean",
        use_pooler: bool = False,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Embed a batch of SMILES strings.

        Returns a list of 1D numpy arrays.
        """
        return [self.embed(s, pooling_strategy=pooling_strategy, use_pooler=use_pooler) for s in input]


class MolformerWrapper(BaseModelWrapper):
    """
    Wrapper for MoLFormer molecular language models (via HuggingFace Transformers).

    Loads a MolFormer checkpoint and exposes:
      - `embed(...)` for a single SMILES
      - `embed_batch(...)` for a list of SMILES

    Pooling strategies:
      • cls  → model.pooler_output
      • mean → average over token embeddings
      • max  → max over token embeddings
    """

    model_type = "molecule"
    available_pooling_strategies = ["cls", "mean", "max"]

    def __init__(self, model_path_or_name: str = "ibm/MoLFormer-XL-both-10pct", **kwargs):
        """
        Initialize the MolformerWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Hugging Face model identifier or local path for the MolFormer weights.
            Defaults to `"ibm/MoLFormer-XL-both-10pct"`.
        **kwargs
            Additional keyword arguments passed to `BaseModelWrapper`.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None

    def load(self, device: torch.device) -> None:
        """
        Load the MolFormer model and tokenizer onto `device`.

        Raises
        ------
        RuntimeError
            If the model or tokenizer fail to load.
        """
        if self.model is not None:
            logging.debug("MolFormer already loaded.")
            return

        logging.info(f"Loading MolFormer '{self.model_name}' (trust_remote_code)…")
        try:
            # need trust_remote_code to pick up custom classes in the repo
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                deterministic_eval=True,
                trust_remote_code=True,
            )
            self.model.to(device).eval()
            self.device = device
            logging.info("MolFormer loaded successfully.")
        except Exception as e:
            logging.error(f"MolFormer load error: {e}")
            raise RuntimeError(f"Could not load MolFormer '{self.model_name}'") from e

    def _preprocess_smiles(self, smiles: str) -> BatchEncoding:
        """
        Tokenize one or more SMILES strings into model-ready input tensors.

        This wraps the HuggingFace tokenizer to convert raw SMILES into
        `input_ids` and `attention_mask`, handling batching, padding, and
        truncation automatically.

        Parameters
        ----------
        smiles : str or list of str
            A single SMILES string (e.g. `"CCO"`) or a list of them (e.g.
            `["CCO", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]`).

        Returns
        -------
        transformers.BatchEncoding
            A BatchEncoding containing at least:

            - `input_ids` (`torch.LongTensor` of shape `(batch_size, seq_len)`)
            - `attention_mask` (`torch.LongTensor` of shape `(batch_size, seq_len)`)

        Raises
        ------
        RuntimeError
            If called before `load()`, i.e. when `self.tokenizer` is still `None`.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        seqs = smiles if isinstance(smiles, (list | tuple)) else [smiles]
        return self.tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Embed a single SMILES string.

        Parameters
        ----------
        input : str
            The SMILES string.
        pooling_strategy : str, default "cls"
            One of "cls", "mean", or "max".

        Returns
        -------
        np.ndarray
            1-D embedding vector.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("MolFormer model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}")

        tokens = self._preprocess_smiles(input)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        # out.pooler_output: (B, hidden_dim)
        # out.last_hidden_state: (B, seq_len, hidden_dim)
        if pooling_strategy == "cls":
            vec = out.pooler_output[0]
        else:
            seq_emb = out.last_hidden_state[0]  # (seq_len, hidden_dim)
            if pooling_strategy == "mean":
                vec = seq_emb.mean(dim=0)
            else:  # max
                vec = seq_emb.max(dim=0).values

        return vec.cpu().numpy()

    def embed_batch(
        self,
        input: str,
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Embed a batch of SMILES strings.

        Parameters
        ----------
        inputs : Sequence[str]
            List of SMILES strings.
        pooling_strategy : str, default "cls"
            One of "cls", "mean", or "max".

        Returns
        -------
        List[np.ndarray]
            A list of 1-D embedding vectors.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("MolFormer model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}")

        tokens = self._preprocess_smiles(input)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        embeddings: list[np.ndarray] = []
        if pooling_strategy == "cls":
            for vec in out.pooler_output:
                embeddings.append(vec.cpu().numpy())
        else:
            # (B, seq_len, hidden_dim)
            seqs = out.last_hidden_state
            if pooling_strategy == "mean":
                pooled = seqs.mean(dim=1)
            else:
                pooled = seqs.max(dim=1).values
            for vec in pooled:
                embeddings.append(vec.cpu().numpy())

        return embeddings


class RDKitWrapper(BaseModelWrapper):
    """
    Wrapper for RDKit molecular fingerprints.

    This generates fixed-length bit vectors (fingerprints) often used as
    baselines for molecular embedding tasks.

    Implements:
      - embed(smiles) → np.ndarray (size: n_bits)
      - embed_batch([...]) → list[np.ndarray]

    Pooling strategies:
      • "flat" (default) → Returns the static fingerprint vector.
        (Note: 'mean'/'max' are ignored as RDKit does not produce token-level embeddings).
    """

    model_type = "molecule"
    available_pooling_strategies = ["flat"]

    def __init__(
        self,
        model_path_or_name: str = "morgan_fingerprint",
        fingerprint_type: Literal["morgan", "rdkit"] = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
        **kwargs,
    ):
        """
        Initialize the RDKit wrapper.

        Args:
            model_path_or_name: Arbitrary identifier for logging (default: "morgan_fingerprint").
            fingerprint_type: "morgan" (ECFP-like) or "rdkit" (topological).
            radius: Radius for Morgan fingerprints (default: 2, approx ECFP4).
            n_bits: Length of the bit vector (default: 2048).
        """
        super().__init__(model_path_or_name, **kwargs)
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        self._loaded = False

    def load(self, device: torch.device):
        """
        Marks the wrapper as loaded.

        Note: RDKit runs entirely on the CPU. The `device` argument is
        stored for consistency but ignored during computation.
        """
        if self._loaded:
            return

        logging.info(
            f"Loading RDKit Wrapper (Type={self.fingerprint_type}, Radius={self.radius}, Bits={self.n_bits})..."
        )
        self.device = device
        self._loaded = True

    def embed(
        self,
        input: str,
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the fingerprint for a single SMILES string.

        Args:
            input: SMILES string.
            pooling_strategy: Ignored (kept for compatibility).

        Returns
        -------
            np.ndarray: A 1D float32 array of 0s and 1s.

        Raises
        ------
            ValueError: If the SMILES string is invalid.
        """
        if not self._loaded:
            raise RuntimeError("RDKit model not loaded. Call load() first.")

        # 1. Parse Molecule
        mol = Chem.MolFromSmiles(input)
        if mol is None:
            # You might prefer to return a zero-vector here depending on your pipeline
            raise ValueError(f"RDKit failed to parse SMILES: {input}")

        # 2. Generate Fingerprint (Bit Vector)
        if self.fingerprint_type == "morgan":
            # ECFP-like fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        elif self.fingerprint_type == "rdkit":
            # RDKit topological fingerprint
            fp = Chem.RDKitFingerprint(mol, fpSize=self.n_bits)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

        # 3. Convert to Numpy
        # We use float32 to be compatible with typical PyTorch/Transformer embeddings
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Computes fingerprints for a batch of SMILES.
        """
        # RDKit is CPU-bound and releases the GIL well for heavy ops,
        # but a simple loop is standard unless you need multiprocessing.
        return [self.embed(s, pooling_strategy=pooling_strategy, **kwargs) for s in inputs]
