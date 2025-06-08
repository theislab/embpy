# Placeholder for small molecule models (e.g., ChemBERTa, MolFormer)
import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .base import BaseModelWrapper


class ChembertaWrapper(BaseModelWrapper):  # Example name
    """Wrapper for ChemBERTa-like models for SMILES strings."""

    model_type = "molecule"
    available_pooling_strategies = ["mean", "max", "cls"]  # Depending on model

    def __init__(self, model_path_or_name: str = "seyonec/ChemBERTa-zinc-base-v1", **kwargs):
        """
        Initializes the SMILES model wrapper.

        Args:
            model_path_or_name (str): Hugging Face model name for SMILES embedding.
            **kwargs: Additional config.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None

    def load(self, device: torch.device):
        """Loads the SMILES model and tokenizer."""
        if self.model is not None:
            logging.debug("SMILES model already loaded.")
            return

        if self.model_name is None:
            raise ValueError("model_path_or_name must be set for molecule model wrapper.")

        logging.info(f"Loading SMILES model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device
            # logging.warning("SMILES model loading is commented out. Need 'transformers' dependency.")
            # Placeholder for now:
            # logging.info("SMILES model loaded (placeholder).")
        except NameError as err:
            logging.error("Failed to load SMILES model: 'transformers' library not found.")
            raise RuntimeError("Please install 'transformers' to use SMILES models: pip install transformers") from err
        except Exception as e:
            logging.error(f"Failed to load SMILES model: {e}")
            raise RuntimeError(f"Could not load SMILES model '{self.model_name}'") from e

    def _preprocess_smiles(self, smiles: str) -> Any:
        """Tokenizes a SMILES string."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        return self.tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        # logging.warning("SMILES preprocessing is commented out.")
        # return {"input_ids": torch.randint(0, 100, (1, len(smiles)))}  # Dummy tokenization

    def embed(
        self,
        input: str,  # Renamed from input_text
        pooling_strategy: str = "mean",
        target_layer: int | None = None,  # Optional layer extraction for HF models
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the embedding for a single SMILES string.

        Args:
            input (str): The input SMILES string.
            pooling_strategy (str): Pooling strategy ('mean', 'max', 'cls').
            target_layer (Optional[int]): Extract embeddings from this hidden layer index.
            **kwargs: Additional arguments.

        Returns
        -------
            np.ndarray: The resulting embedding vector.
        """
        smiles = input  # Assign to a more specific variable name internally if desired
        if self.model is None or self.device is None:
            raise RuntimeError("SMILES model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        # Preprocess
        tokenized_input = self._preprocess_smiles(smiles)
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            print(f"Attention mask shape: {attention_mask}")
        # logging.warning("SMILES embedding uses placeholder preprocessing and inference.")
        # Placeholder inference
        # input_ids = torch.randint(0, 100, (1, min(len(smiles), 128)), device=self.device)  # Dummy input
        # attention_mask = torch.ones_like(input_ids)

        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=(target_layer is not None)
            )
            if target_layer is not None:
                embeddings_tensor = outputs.hidden_states[target_layer]
            else:
                embeddings_tensor = outputs.last_hidden_state

            # Placeholder output
            # hidden_dim = 768  # Example for base models
            # embeddings_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], hidden_dim, device=self.device)

        # Squeeze batch dim
        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[0] == 1:
            embeddings_tensor = embeddings_tensor.squeeze(0)  # (L, D)

        # Apply pooling
        if pooling_strategy == "cls":
            pooled_embedding = embeddings_tensor[0, :].cpu().numpy()
        else:
            pooled_embedding = self._apply_pooling(embeddings_tensor, pooling_strategy)

        # Cleanup
        del input_ids, attention_mask, embeddings_tensor  # , outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pooled_embedding


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
        inputs: str,
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

        tokens = self._preprocess_smiles(inputs)
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

    # Override embed_batch for efficiency
    # def embed_batch(...) -> list[np.ndarray]: ...
