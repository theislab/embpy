# embpy/models/protein_models.py
import io
import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

from .base import BaseModelWrapper

# ESMC SDK import
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    _HAVE_ESMC = True
except ImportError:
    _HAVE_ESMC = False
from transformers import AutoModel, AutoTokenizer


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

    def embed_batch(
        self,
        inputs: list[str],
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute embeddings for multiple protein sequences.

        This feeds each sequence through the full ESM2 pipeline:
          1. Tokenize via the loaded HuggingFace tokenizer
          2. Move tokens to `self.device`
          3. Run the model to produce hidden states or last hidden layer
          4. Pool across sequence positions using `pooling_strategy`

        Parameters
        ----------
        inputs
            A list of protein sequence strings (e.g. ["MTEYKLVVVG", "ACDEFGHIK..."]).
        pooling_strategy
            One of {"mean", "max", "cls"}.  “cls” returns the embedding of the first token,
            “mean” averages across positions, and “max” takes the maximum across positions.
        target_layer
            If specified, extract embeddings from a particular hidden state layer
            instead of the default `last_hidden_state`.  Must be in range
            `[-num_layers, num_layers)`.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays, one per input sequence, each of length
            `self.TRUNK_OUTPUT_DIM`.

        Raises
        ------
        RuntimeError
            If `load()` has not been called (i.e., model or device is None).
        ValueError
            If `pooling_strategy` is not one of the supported strategies.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("ESM2 model not loaded. Please call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Choose from {self.available_pooling_strategies}"
            )

        embeddings: list[np.ndarray] = []
        for seq in inputs:
            emb = self.embed(
                input=seq,
                pooling_strategy=pooling_strategy,
                target_layer=target_layer,
                **kwargs,
            )
            embeddings.append(emb)
        return embeddings


class ESMCWrapper(BaseModelWrapper):
    """
    Wrapper for the ESMC‑SDK protein language models (e.g., 'esmc_300m', 'esmc_600m').

    Uses the external ESMC client to generate embeddings.
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self,
        model_path_or_name: str = "esmc_300m",
        **kwargs: Any,
    ):
        """
        Initialize the ESMC wrapper.

        Args:
            model_path_or_name (str): ESMC checkpoint name, e.g. 'esmc_300m' or 'esmc_600m'.
        """
        self.model_name = model_path_or_name
        self.client: ESMC | None = None
        self.device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        """
        Load the ESMC client onto the specified device.

        Raises
        ------
            RuntimeError: If SDK not installed or loading fails.
        """
        if not _HAVE_ESMC:
            raise RuntimeError("ESMC SDK not installed.")
        if self.client is not None:
            return
        self.device = device
        logging.info(f"Loading ESMC client '{self.model_name}'...")
        try:
            self.client = ESMC.from_pretrained(self.model_name).to(device).eval()
            logging.info("ESMC client loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ESMC client: {e}")
            raise RuntimeError(f"Could not load ESMC client '{self.model_name}'") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate an embedding via ESMC SDK.

        Args:
            sequence (str): Amino‑acid string to embed.
            pooling_strategy (str): One of 'mean', 'max', or 'cls'.

        Returns
        -------
            np.ndarray: Pooled 1D embedding vector.

        Raises
        ------
            RuntimeError: If client isn't loaded.
            ValueError: On invalid pooling strategy.
        """
        if self.client is None or self.device is None:
            raise RuntimeError("ESMC client not loaded; call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        prot = ESMProtein(sequence=input)
        tensor = self.client.encode(prot)
        out = self.client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
        emb = out.embeddings
        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        if pooling_strategy == "cls":
            pooled = emb[0]
        elif pooling_strategy == "max":
            pooled = torch.max(emb, dim=0)[0]
        else:
            pooled = torch.mean(emb, dim=0)

        return pooled.cpu().numpy()

    def embed_batch(
        self,
        sequences: list[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Batch‑mode embedding for multiple sequences.

        Args:
            sequences (Sequence[str]): A list of amino‑acid sequences to embed.
            pooling_strategy (str): Pooling strategy to apply per sequence.

        Returns
        -------
            List[np.ndarray]: Pooled embeddings, one per input.
        """
        return [self.embed(seq, pooling_strategy=pooling_strategy, **kwargs) for seq in sequences]


class STRINGWrapper(BaseModelWrapper):
    """
    STRINGWrapper API.

    Wrapper for the STRING database API to retrieve protein interaction data
    and convert it into simple embeddings (e.g., pooled interaction scores).

    Methods
    -------
        - load: No-op loader (stores device).
        - embed: Fetches the interaction network for a single protein,
          pools the combined scores, and returns a 1D numpy array.
        - embed_batch: Inherited batch looping over embed.
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max"]
    BASE_URL = "https://version-12-0.string-db.org/api"

    def __init__(
        self,
        model_path_or_name: str | None = None,
        caller_identity: str = "my_app",
        **kwargs: Any,
    ):
        """
        Initialize the STRINGWrapper.

        Args:
            model_path_or_name (Optional[str]): Ignored for STRING API.
            caller_identity (str): Identifier for your application. Used by STRING for usage tracking.
            **kwargs: Additional configuration (ignored).
        """
        super().__init__(model_path_or_name, **kwargs)
        self.caller = caller_identity

    def load(self, device: torch.device) -> None:
        """
        'Load' the wrapper by storing the device (no actual model to load).

        Args:
            device (torch.device): Device placeholder (not used).
        """
        self.device = device

    def _post(self, output_format: str, method: str, params: dict[str, Any]) -> requests.Response:
        """
        Internal helper to perform a POST to STRING API.

        Args:
            output_format (str): 'tsv', 'json', 'xml', or 'image'.
            method (str): API endpoint (e.g., 'get_string_ids', 'network').
            params (dict[str, Any]): API-specific parameters.

        Returns
        -------
            requests.Response: The HTTP response.

        Raises
        ------
            requests.HTTPError: On bad status codes.
        """
        url = f"{self.BASE_URL}/{output_format}/{method}"
        response = requests.post(
            url,
            data={**params, "caller_identity": self.caller},
        )
        # Raise on HTTP error
        response.raise_for_status()
        # Respect rate limits
        time.sleep(1)
        return response

    def get_string_ids(
        self,
        identifiers: Sequence[str],
        species: int | None = None,
        echo_query: bool = False,
    ) -> pd.DataFrame:
        """
        Map external IDs (gene symbols, UniProt IDs) to STRING internal IDs.

        Args:
            identifiers (Sequence[str]): Input IDs.
            species (Optional[int]): NCBI taxonomy ID filter.
            echo_query (bool): Include original query in output.

        Returns
        -------
            pd.DataFrame: Columns include 'queryItem', 'stringId', etc.
        """
        params: dict[str, Any] = {
            "identifiers": "\r".join(identifiers),
            "echo_query": int(echo_query),
        }
        if species is not None:
            params["species"] = species

        resp = self._post("tsv", "get_string_ids", params)
        return pd.read_csv(io.StringIO(resp.text), sep="\t", header=0)

    def get_network(
        self,
        identifiers: Sequence[str],
        species: int,
        required_score: int = 400,
        network_type: str = "functional",
    ) -> pd.DataFrame:
        """
        Retrieve interaction edges for given STRING IDs.

        Args:
            identifiers (Sequence[str]): STRING or external IDs.
            species (int): NCBI taxonomy ID.
            required_score (int): Minimum combined score (0–1000).
            network_type (str): 'functional', 'physical', etc.

        Returns
        -------
            pd.DataFrame: Columns include 'protein1', 'protein2', 'combined_score'.
        """
        params: dict[str, Any] = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
            "network_type": network_type,
        }
        resp = self._post("tsv", "network", params)
        return pd.read_csv(io.StringIO(resp.text), sep="\t", header=0)

    def embed(
        self,
        input: str,
        species: int = 9606,
        required_score: int = 400,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Retrieve the interaction network for a single protein as a DataFrame.

        This method maps the input identifier to a STRING ID and retrieves
        the full interaction network DataFrame (with columns like 'protein1',
        'protein2', 'combined_score').

        Args:
            input (str): Gene symbol or STRING ID.
            species (int): NCBI taxonomy ID.
            required_score (int): Minimum combined score (0–1000).
            **kwargs: Ignored.

        Returns
        -------
            pd.DataFrame: DataFrame of interaction edges for the given protein.
        """
        # Map to STRING ID
        ids_df = self.get_string_ids([input], species=species, echo_query=False)
        string_id = ids_df["stringId"].iloc[0]

        # Retrieve and return the full network DataFrame
        net_df = self.get_network([string_id], species=species, required_score=required_score)
        return net_df

    def embed_batch(
        self,
        inputs: Sequence[str],
        species: int = 9606,
        required_score: int = 400,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """
        Retrieve interaction networks for multiple proteins.

        Args:
            inputs (Sequence[str]): List of gene symbols or STRING IDs.
            species (int): NCBI taxonomy ID.
            required_score (int): Minimum combined score (0–1000).
            **kwargs: Ignored.

        Returns
        -------
            Dict[str, pd.DataFrame]: Mapping from input identifier to its network DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for inp in inputs:
            results[inp] = self.embed(inp, species=species, required_score=required_score)
        return results
