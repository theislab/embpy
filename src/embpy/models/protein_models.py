# embpy/models/esm2_wrapper.py
import logging
from typing import Any

import numpy as np
import torch

# ESMC SDK import
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer

from .base import BaseModelWrapper


class ESM2Wrapper(BaseModelWrapper):
    """
    Wrapper for ESM2 protein language models using Hugging Face Transformers.

    Implements the ProteinEmbedder interface for ESM2 models. Supports tokenization,
    model inference, and pooling strategies to produce protein embeddings.
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(self, model_path_or_name: str = "facebook/esm2_t6_8M_UR50D", **kwargs):
        """
        Initialize the ESM2Wrapper.

        Args:
            model_path_or_name (str): Model identifier or path for the ESM2 model.
            **kwargs: Additional configuration parameters.
        """
        self.model_name = model_path_or_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, device: torch.device) -> None:
        """
        Load the ESM2 model and tokenizer onto the specified device.

        Args:
            device (torch.device): Device to load the model (e.g., cpu or cuda).

        Raises
        ------
            RuntimeError: If model or tokenizer loading fails.
        """
        if self.model is not None:
            logging.debug("ESM2 model already loaded.")
            return

        if not self.model_name:
            raise ValueError("model_path_or_name must be provided for ESM2Wrapper.")

        logging.info(f"Loading ESM2 model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            self.device = device
            logging.info("ESM2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ESM2 model: {e}")
            raise RuntimeError(f"Could not load ESM2 model '{self.model_name}'") from e

    def _preprocess_sequence(self, sequence: str) -> Any:
        """
        Tokenize the protein sequence using the associated tokenizer.

        Args:
            sequence (str): The input protein sequence.

        Returns
        -------
            dict: Tokenized sequence inputs suitable for model inference.

        Raises
        ------
            RuntimeError: If the tokenizer is not loaded.
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not loaded.")
        return self.tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)

    def embed(
        self, input: str, pooling_strategy: str = "mean", target_layer: int | None = None, **kwargs
    ) -> np.ndarray:
        """
        Generate an embedding for a protein sequence using the ESM2 model.

        Args:
            sequence (str): The protein sequence.
            pooling_strategy (str): Pooling strategy to aggregate token-level embeddings.
                                    Options: 'mean', 'max', 'cls'.
            target_layer (Optional[int]): Specific layer from which to extract embeddings.
            **kwargs: Additional parameters.

        Returns
        -------
            np.ndarray: A pooled vector representing the protein embedding.

        Raises
        ------
            RuntimeError: If the model is not loaded.
            ValueError: If an invalid pooling strategy is provided.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("ESM2 model not loaded. Please call load() first.")

        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available options: {self.available_pooling_strategies}"
            )

        # Preprocess the sequence and move inputs to the correct device:
        tokenized_input = self._preprocess_sequence(input)
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Model inference:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=(target_layer is not None)
            )

            if target_layer is not None:
                hidden_states = outputs.hidden_states
                if not hidden_states:
                    raise ValueError("Model did not return hidden states.")
                if not (-len(hidden_states) <= target_layer < len(hidden_states)):
                    raise ValueError(f"Invalid target_layer index {target_layer}.")
                embeddings_tensor = hidden_states[target_layer]
            else:
                embeddings_tensor = outputs.last_hidden_state

        # If batch dimension is 1, squeeze it:
        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[0] == 1:
            embeddings_tensor = embeddings_tensor.squeeze(0)

        # Apply pooling strategy:
        if pooling_strategy == "cls":
            pooled_embedding = embeddings_tensor[0].cpu().numpy()
        elif pooling_strategy == "max":
            pooled_embedding = torch.max(embeddings_tensor, dim=0)[0].cpu().numpy()
        else:  # default to mean pooling:
            pooled_embedding = torch.mean(embeddings_tensor, dim=0).cpu().numpy()

        return pooled_embedding


class ESMCWrapper(BaseModelWrapper):
    """
    Wrapper for the ESMC‐SDK protein language model 'esmc_300m'.

    This class handles:
      1. Loading the ESMC client onto the target device.
      2. Encoding a protein sequence via ESMC.encode.
      3. Retrieving logits and embeddings via ESMC.logits.
      4. Pooling over the sequence dimension: mean, max or CLS-token.

    Attributes
    ----------
    model_type : str
        Always "protein" for this wrapper.
    available_pooling_strategies : list[str]
        Supported pooling modes: ["mean", "max", "cls"].
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(self):
        """
        ESMC Embeddings.

        Initialize the ESMC wrapper. No model_path needed—ESMC is hard‐coded
        to use the 'esmc_300m' checkpoint.
        """
        self.client: ESMC | None = None
        self.device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        """
        Load the ESMC client for 'esmc_300m' onto the specified device.

        Parameters
        ----------
        device : torch.device
            The target device (CPU, CUDA, or MPS).

        Raises
        ------
        RuntimeError
            If loading fails for any reason.
        """
        self.device = device
        logging.info("Loading ESMC client 'esmc_300m' …")
        try:
            self.client = ESMC.from_pretrained("esmc_300m").to(device)
            logging.info(f"ESMC client loaded on {device}")
        except Exception as e:
            logging.error(f"Failed to load ESMC client: {e}")
            raise RuntimeError("Could not load ESMC client 'esmc_300m'.") from e

    def embed(self, sequence: str, pooling_strategy: str = "mean", **kwargs: Any) -> np.ndarray:
        """
        Generate a protein embedding.

        Parameters
        ----------
        sequence : str
            The amino‐acid string to embed (e.g. "MEEPQSDPSV").
        pooling_strategy : str, optional
            One of "mean", "max", or "cls" (the first token).

        Returns
        -------
        np.ndarray
            A 1D vector of size hidden_dim with the pooled representation.

        Raises
        ------
        RuntimeError
            If the client isn't loaded yet.
        ValueError
            If an unsupported pooling_strategy is requested.
        """
        if self.client is None or self.device is None:
            raise RuntimeError("ESMC client not loaded; call load() first.")

        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'; choose from {self.available_pooling_strategies}.")

        # 1) Wrap sequence
        prot = ESMProtein(sequence=sequence)
        # 2) Encode → tensor
        tensor = self.client.encode(prot)
        # 3) Get logits & embeddings
        out = self.client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
        emb = out.embeddings  # shape: (1, L, H) or (L, H)
        # squeeze batch if needed
        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        # 4) Pool over length dimension
        if pooling_strategy == "cls":
            pooled = emb[0]
        elif pooling_strategy == "max":
            pooled = torch.max(emb, dim=0)[0]
        else:  # mean
            pooled = torch.mean(emb, dim=0)

        return pooled.cpu().numpy()

    def embed_batch(self, sequences: list[str], pooling_strategy: str = "mean", **kwargs: Any) -> list[np.ndarray]:
        """
        Batch‐mode embedding: simply calls `embed` on each sequence in turn.

        Parameters
        ----------
        sequences : list[str]
            A list of amino‐acid sequences.
        pooling_strategy : str, optional
            Pooling strategy to apply in each `embed` call.

        Returns
        -------
        list[np.ndarray]
            A list of pooled embeddings, one per input sequence.
        """
        return [self.embed(seq, pooling_strategy=pooling_strategy, **kwargs) for seq in sequences]
