# Placeholder for small molecule models (e.g., ChemBERTa, MolFormer)
import logging
from collections.abc import Sequence
from typing import Any

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
        self.tokenizer: Any = None
        self.model: Any = None
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
        if self.tokenizer is None:
            raise RuntimeError("Call load() first.")
        toks: BatchEncoding = self.tokenizer(  # type: ignore[operator]
            smiles,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids: list[int] = toks["input_ids"]  # type: ignore[assignment]
        return len(input_ids)

    def _preprocess_smiles(self, smiles: str) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Call load() first.")
        # we’ll still pass max_length for safety even though we pre-check length
        return self.tokenizer(  # type: ignore[operator]
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
        if self.max_len is not None and L > self.max_len:
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
    ) -> list[np.ndarray | None]:
        """
        Embed a batch of SMILES strings.

        Returns a list of 1D numpy arrays (or None for skipped SMILES).
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
            model = AutoModel.from_pretrained(
                self.model_name,
                deterministic_eval=True,
                trust_remote_code=True,
            )
            self.model = model.to(device).eval()  # type: ignore[union-attr]
            self.device = device
            logging.info("MolFormer loaded successfully.")
        except Exception as e:
            logging.error(f"MolFormer load error: {e}")
            raise RuntimeError(f"Could not load MolFormer '{self.model_name}'") from e

    def _preprocess_smiles(self, smiles: str | list[str]) -> BatchEncoding:
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
        inputs: Sequence[str],
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

        tokens = self._preprocess_smiles(list(inputs))
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
    """RDKit molecular fingerprint wrapper.

    Generates fixed-length fingerprint vectors from SMILES strings.
    Supports both **binary** (0/1 bit vectors) and **count-based**
    (continuous integer) representations.

    Fingerprint types
    -----------------
    * ``"rdkit"``  -- RDKit topological (Daylight-like), binary
    * ``"morgan"`` -- Morgan / ECFP bit vector, binary
    * ``"morgan_count"`` -- Morgan / ECFP **count** vector, continuous
    * ``"maccs"``  -- MACCS keys (166 public keys), binary
    * ``"atom_pair"`` -- Atom-pair bit fingerprint, binary
    * ``"atom_pair_count"`` -- Atom-pair count fingerprint, continuous
    * ``"topological_torsion"`` -- Topological torsion bits, binary
    * ``"topological_torsion_count"`` -- Topological torsion counts, continuous
    """

    model_type = "molecule"
    available_pooling_strategies = ["flat"]

    _VALID_FP_TYPES = (
        "morgan",
        "morgan_count",
        "rdkit",
        "maccs",
        "atom_pair",
        "atom_pair_count",
        "topological_torsion",
        "topological_torsion_count",
    )

    def __init__(
        self,
        model_path_or_name: str = "rdkit_fingerprint",
        fingerprint_type: str | None = None,
        radius: int = 2,
        n_bits: int = 2048,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        # Auto-detect: when model_path_or_name is itself a valid fingerprint
        # type (e.g. loaded from MODEL_REGISTRY), use it as fingerprint_type.
        if fingerprint_type is None:
            if model_path_or_name in self._VALID_FP_TYPES:
                fingerprint_type = model_path_or_name
            else:
                fingerprint_type = "rdkit"
        if fingerprint_type not in self._VALID_FP_TYPES:
            raise ValueError(f"Unknown fingerprint_type '{fingerprint_type}'. Choose from {self._VALID_FP_TYPES}")
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        # MACCS keys always have a fixed size (167 bits)
        self.n_bits = 167 if fingerprint_type == "maccs" else n_bits
        self._loaded = False

    @property
    def is_count_fingerprint(self) -> bool:
        """``True`` when the fingerprint produces continuous counts."""
        return self.fingerprint_type.endswith("_count")

    @property
    def is_binary_fingerprint(self) -> bool:
        """``True`` when the fingerprint produces binary 0/1 vectors."""
        return not self.is_count_fingerprint

    def load(self, device: torch.device) -> None:
        """Mark the wrapper as loaded (RDKit is CPU-only)."""
        if self._loaded:
            return
        logging.info(
            "Loading RDKit wrapper (type=%s, radius=%d, n_bits=%d)",
            self.fingerprint_type,
            self.radius,
            self.n_bits,
        )
        self.device = device
        self._loaded = True

    def embed(
        self,
        input: str,
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the fingerprint for a single SMILES string.

        Returns
        -------
        np.ndarray
            1-D ``float32`` array of length :attr:`n_bits`.
            Binary fingerprints contain only 0.0 / 1.0.
            Count fingerprints contain non-negative integers (as float32).
        """
        if not self._loaded:
            raise RuntimeError("RDKit model not loaded. Call load() first.")

        mol = Chem.MolFromSmiles(input)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {input}")

        return self._compute_fingerprint(mol)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute fingerprints for a batch of SMILES strings."""
        return [self.embed(s, pooling_strategy=pooling_strategy, **kwargs) for s in inputs]

    # ------------------------------------------------------------------
    # Internal fingerprint computation
    # ------------------------------------------------------------------

    def _compute_fingerprint(self, mol: Any) -> np.ndarray:
        """Dispatch to the correct RDKit fingerprint generator."""
        fp_type = self.fingerprint_type

        # --- Morgan (ECFP-like) ---
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
                mol,
                radius=self.radius,
                nBits=self.n_bits,
            )
            return self._bitvect_to_array(fp)

        if fp_type == "morgan_count":
            fp = AllChem.GetHashedMorganFingerprint(  # type: ignore[attr-defined]
                mol,
                radius=self.radius,
                nBits=self.n_bits,
            )
            return self._sparse_intvect_to_array(fp)

        # --- RDKit topological ---
        if fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            return self._bitvect_to_array(fp)

        # --- MACCS keys (167 bits) ---
        if fp_type == "maccs":
            from rdkit.Chem import MACCSkeys

            fp = MACCSkeys.GenMACCSKeys(mol)  # type: ignore[attr-defined]
            return self._bitvect_to_array(fp)

        # --- Atom pair ---
        if fp_type == "atom_pair":
            from rdkit.Chem import rdMolDescriptors

            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=self.n_bits,
            )
            return self._bitvect_to_array(fp)

        if fp_type == "atom_pair_count":
            from rdkit.Chem import rdMolDescriptors

            return self._sparse_intvect_to_array(rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=self.n_bits))

        # --- Topological torsion ---
        if fp_type == "topological_torsion":
            from rdkit.Chem import rdMolDescriptors

            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol,
                nBits=self.n_bits,
            )
            return self._bitvect_to_array(fp)

        if fp_type == "topological_torsion_count":
            from rdkit.Chem import rdMolDescriptors

            return self._sparse_intvect_to_array(
                rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(
                    mol,
                    nBits=self.n_bits,
                )
            )

        raise ValueError(f"Unknown fingerprint type: {fp_type}")

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _bitvect_to_array(self, fp: Any) -> np.ndarray:
        """Convert an RDKit ``ExplicitBitVect`` to a float32 numpy array."""
        arr = np.zeros(self.n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _sparse_intvect_to_array(self, fp: Any) -> np.ndarray:
        """Convert an RDKit ``(U)IntSparseIntVect`` to a dense float32 array."""
        arr = np.zeros(self.n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % self.n_bits] += count
        return arr


# ============================================================
# MiniMol wrapper (Message-Passing GNN – GINE)
# ============================================================


class MiniMolWrapper(BaseModelWrapper):
    """MiniMol pre-trained GINE molecular fingerprint wrapper.

    MiniMol is a 10M-parameter message-passing GNN (GIN with edge features)
    pre-trained on 6M molecules across 3,300+ bio/quantum tasks from the
    Graphium LargeMix dataset.  It produces fixed **512-dimensional**
    molecular fingerprints from SMILES strings.

    The model and weights ship together in the ``minimol`` pip package,
    so no separate download step is required.

    Requires
    --------
    ``pip install minimol``
    (also installs ``graphium`` and ``torch-geometric`` automatically)

    References
    ----------
    * Repository: https://github.com/graphcore-research/minimol
    * MiniMol outperforms models 10x its size on 17/22 TDC ADMET tasks.

    Notes
    -----
    MiniMol manages its own device selection (CUDA if available, else CPU).
    The ``device`` parameter in :meth:`load` is stored for interface
    consistency but does not override MiniMol's internal device selection.
    """

    model_type: str = "molecule"  # type: ignore[assignment]
    available_pooling_strategies: list[str] = ["flat"]
    EMBEDDING_DIM: int = 512

    def __init__(
        self,
        model_path_or_name: str = "minimol",
        batch_size: int = 100,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self._batch_size = batch_size
        self._minimol: Any = None
        self._loaded = False

    def load(self, device: torch.device) -> None:
        """Load the MiniMol model.

        Weights are bundled with the ``minimol`` package.  The first call
        initialises the underlying Graphium predictor and loads the
        pre-trained checkpoint.
        """
        if self._loaded:
            return

        try:
            from minimol import Minimol  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "MiniMol is not installed.  Install with:\n"
                "  pip install minimol\n"
                "See: https://github.com/graphcore-research/minimol"
            ) from e

        logging.info("Loading MiniMol (GINE, 512-dim fingerprints)…")
        self._minimol = Minimol(batch_size=self._batch_size)
        self.device = device
        self._loaded = True
        logging.info("MiniMol loaded successfully.")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute a 512-dim fingerprint for a single SMILES string."""
        if not self._loaded or self._minimol is None:
            raise RuntimeError("MiniMol not loaded. Call load() first.")

        results = self._minimol([input])
        if not results:
            raise ValueError(f"MiniMol returned no results for SMILES: {input}")
        return results[0].detach().cpu().numpy().astype(np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute 512-dim fingerprints for a batch of SMILES strings."""
        if not self._loaded or self._minimol is None:
            raise RuntimeError("MiniMol not loaded. Call load() first.")

        smiles_list = list(inputs)
        results = self._minimol(smiles_list)
        return [r.detach().cpu().numpy().astype(np.float32) for r in results]


# ============================================================
# MHG-GNN wrapper (Molecular Hypergraph Grammar + GNN)
# ============================================================


class MHGGNNWrapper(BaseModelWrapper):
    """MHG-GNN molecular hypergraph autoencoder wrapper.

    MHG-GNN combines a GIN-based graph encoder with a sequential decoder
    based on Molecular Hypergraph Grammar (MHG).  It was pre-trained on
    ~1.34M molecules from PubChem.

    The **encoder** produces latent-space embeddings from SMILES strings.
    The **decoder** can reconstruct structurally valid SMILES from those
    latent vectors, which is unique among molecular GNN models.

    Requires
    --------
    * ``mhg-gnn`` model package (from IBM Research / GT4SD)
    * ``torch-geometric``
    * ``huggingface-hub``

    Install::

        pip install torch-geometric huggingface-hub
        pip install git+https://github.com/GT4SD/mhg-gnn.git

    HuggingFace model: ``ibm-research/materials.mhg-ged``

    References
    ----------
    * Paper: https://arxiv.org/abs/2309.16374
    * HF: https://huggingface.co/ibm-research/materials.mhg-ged
    """

    model_type: str = "molecule"  # type: ignore[assignment]
    available_pooling_strategies: list[str] = ["flat"]

    def __init__(
        self,
        model_path_or_name: str = "ibm-research/materials.mhg-ged",
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self._wrapper: Any = None
        self._loaded = False

    def load(self, device: torch.device) -> None:
        """Load MHG-GNN pretrained weights from HuggingFace Hub.

        The model package's built-in ``load()`` function is used, which
        downloads the pretrained pickle from HuggingFace Hub and
        constructs the ``GrammarGINVAE`` model automatically.
        """
        if self._loaded:
            return

        import importlib

        _load_fn: Any = None
        for mod_path in ("mhg_model.load", "mhg_gnn.load"):
            try:
                mod = importlib.import_module(mod_path)
                _load_fn = getattr(mod, "load", None)
                if _load_fn is not None:
                    break
            except ImportError:
                continue

        if _load_fn is None:
            raise ImportError(
                "MHG-GNN model code not found.  Install one of:\n"
                "  pip install git+https://github.com/GT4SD/mhg-gnn.git\n"
                "  (or from the IBM internal mhg-gnn repository)\n"
                "Also requires: pip install torch-geometric huggingface-hub\n"
                "See: https://huggingface.co/ibm-research/materials.mhg-ged"
            )

        logging.info(f"Loading MHG-GNN from HuggingFace: {self.model_name}…")
        self._wrapper = _load_fn()
        self._wrapper.to(device)
        self.device = device
        self._loaded = True
        logging.info("MHG-GNN loaded successfully.")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a single SMILES string into its latent representation."""
        if not self._loaded or self._wrapper is None:
            raise RuntimeError("MHG-GNN not loaded. Call load() first.")

        with torch.no_grad():
            embeddings = self._wrapper.encode([input])
        if not embeddings:
            raise ValueError(f"MHG-GNN returned no embeddings for SMILES: {input}")
        return embeddings[0].detach().cpu().numpy().astype(np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "flat",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Encode a batch of SMILES strings into latent representations."""
        if not self._loaded or self._wrapper is None:
            raise RuntimeError("MHG-GNN not loaded. Call load() first.")

        with torch.no_grad():
            embeddings = self._wrapper.encode(list(inputs))
        return [e.detach().cpu().numpy().astype(np.float32) for e in embeddings]

    def decode(self, embeddings: list[np.ndarray]) -> list[str | None]:
        """Decode latent vectors back to SMILES strings.

        This leverages MHG-GNN's decoder, which is guaranteed to produce
        structurally valid molecules.

        Parameters
        ----------
        embeddings : list[np.ndarray]
            Latent vectors (e.g. from :meth:`embed` or :meth:`embed_batch`).

        Returns
        -------
        list[str | None]
            Reconstructed SMILES strings (``None`` for failed decodings).
        """
        if not self._loaded or self._wrapper is None:
            raise RuntimeError("MHG-GNN not loaded. Call load() first.")

        tensors = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        dev = self.device if self.device is not None else torch.device("cpu")
        tensors = [t.to(dev) for t in tensors]

        with torch.no_grad():
            decoded: list[str | None] = self._wrapper.decode(tensors)
        return decoded


# ============================================================
# MolE wrapper (Graph Transformer – DeBERTa-based)
# ============================================================


class MolEWrapper(BaseModelWrapper):
    """MolE (Molecular Embeddings) graph transformer wrapper.

    MolE is a DeBERTa-based graph transformer with disentangled attention,
    pre-trained on ~842M molecules via a two-step strategy:

    1. **Self-supervised** pre-training on molecular graph representations.
    2. **Multi-task supervised** training to assimilate biological information.

    It produces CLS-pooled embeddings from molecular graphs and achieves
    SOTA on 9/22 ADMET tasks in the TDC benchmark.

    .. note::

       Pretrained weights are **not** yet publicly released.  You must
       provide a ``checkpoint_path`` obtained through your own training
       or directly from the authors.

    Requires
    --------
    ``pip install git+https://github.com/recursionpharma/mole_public.git``

    References
    ----------
    * Repository: https://github.com/recursionpharma/mole_public
    * Published in *Nature Communications* (2024).
    """

    model_type: str = "molecule"  # type: ignore[assignment]
    available_pooling_strategies: list[str] = ["cls"]

    def __init__(
        self,
        model_path_or_name: str = "mole",
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self._checkpoint_path = checkpoint_path
        self._mole_predict: Any = None
        self._loaded = False

    def load(self, device: torch.device) -> None:
        """Load the MolE model.

        Uses the ``mole.mole_predict`` high-level API internally.  Requires
        a valid ``checkpoint_path`` pointing to a pre-trained ``.ckpt`` file.
        """
        if self._loaded:
            return

        _mole_predict: Any = None
        for mod_path in ("mole.mole_predict", "mole.cli.mole_predict"):
            try:
                import importlib

                _mole_predict = importlib.import_module(mod_path)
                break
            except ImportError:
                continue

        if _mole_predict is None:
            raise ImportError(
                "MolE is not installed.  Install with:\n"
                "  pip install git+https://github.com/recursionpharma/mole_public.git\n"
                "See: https://github.com/recursionpharma/mole_public"
            )

        if self._checkpoint_path is None:
            raise ValueError(
                "MolE requires a pretrained checkpoint path.  Pass "
                "checkpoint_path='path/to/checkpoint.ckpt' to the constructor.\n"
                "Note: pretrained weights are not yet publicly released."
            )

        self._mole_predict = _mole_predict
        self.device = device
        self._loaded = True
        logging.info(f"MolE wrapper ready (checkpoint: {self._checkpoint_path}).")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the CLS-pooled embedding for a single SMILES string."""
        if not self._loaded or self._mole_predict is None:
            raise RuntimeError("MolE not loaded. Call load() first.")

        batch_size: int = kwargs.get("batch_size", 32)  # type: ignore[assignment]
        num_workers: int = kwargs.get("num_workers", 0)  # type: ignore[assignment]

        embeddings = self._mole_predict.encode(
            smiles=[input],
            pretrained_model=self._checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return np.asarray(embeddings[0], dtype=np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute CLS-pooled embeddings for a batch of SMILES strings."""
        if not self._loaded or self._mole_predict is None:
            raise RuntimeError("MolE not loaded. Call load() first.")

        batch_size: int = kwargs.get("batch_size", 32)  # type: ignore[assignment]
        num_workers: int = kwargs.get("num_workers", 4)  # type: ignore[assignment]

        embeddings = self._mole_predict.encode(
            smiles=list(inputs),
            pretrained_model=self._checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return [np.asarray(e, dtype=np.float32) for e in embeddings]
