# embpy/models/protein_models.py
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .base import BaseModelWrapper

# EvolutionaryScale ESM SDK imports
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput

    _HAVE_ESMC = True
except ImportError:
    _HAVE_ESMC = False

try:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient

    _HAVE_ESM3 = True
except ImportError:
    _HAVE_ESM3 = False
    ESM3 = None  # type: ignore
    ESM3InferenceClient = None  # type: ignore
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer


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
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None

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
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("ESM2 model not loaded. Please call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Choose from {self.available_pooling_strategies}"
            )

        batch_size = kwargs.pop("batch_size", 32)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            tokenized = self.tokenizer(
                chunk, return_tensors="pt", truncation=True, padding=True,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=(target_layer is not None),
                )
                if target_layer is not None:
                    emb_tensor = outputs.hidden_states[target_layer]
                else:
                    emb_tensor = outputs.last_hidden_state

            for i in range(emb_tensor.shape[0]):
                seq_emb = emb_tensor[i]
                if attention_mask is not None:
                    mask = attention_mask[i].unsqueeze(-1).bool()
                else:
                    mask = None

                if pooling_strategy == "cls":
                    pooled = seq_emb[0]
                elif pooling_strategy == "max":
                    if mask is not None:
                        seq_emb = seq_emb.masked_fill(~mask, float("-inf"))
                    pooled = torch.max(seq_emb, dim=0)[0]
                else:
                    if mask is not None:
                        pooled = (seq_emb * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                    else:
                        pooled = seq_emb.mean(dim=0)

                all_embeddings.append(pooled.cpu().numpy())

        return all_embeddings


class ESMCWrapper(BaseModelWrapper):
    """
    Wrapper for the ESMC-SDK protein language models (e.g., 'esmc_300m', 'esmc_600m').

    This class hides the details of the ESMC client and provides a
    simple interface for:
      - loading the model to a device,
      - embedding a single sequence with mean/max/cls pooling,
      - optionally returning hidden‐states from all layers,
      - embedding batches of sequences.

    Attributes
    ----------
    model_name : str
        The checkpoint identifier passed to ESMC.from_pretrained().
    client : ESMC | None
        The underlying ESMC inference client (None until loaded).
    device : torch.device | None
        Device to which the client is moved (None until loaded).
    available_pooling_strategies : List[str]
        Allowed values for pooling_strategy: ['mean', 'max', 'cls'].
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(self, model_path_or_name: str = "esmc_300m", **kwargs: Any):
        """
        Initialize the ESMCWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            The ESMC checkpoint name, e.g. 'esmc_300m' or 'esmc_600m'.
            Defaults to 'esmc_300m'.
        """
        super().__init__(model_path_or_name=model_path_or_name, **kwargs)
        self.client: ESMC | None = None
        self.device: torch.device | None = None

    def load(self, device: torch.device) -> None:
        """
        Load the ESMC client onto the specified device and switch to eval mode.

        If already loaded, this is a no-op.

        Parameters
        ----------
        device : torch.device
            Target device for model inference (CPU, CUDA, or MPS).

        Raises
        ------
        RuntimeError
            If the ESMC SDK is not installed or loading the client fails.
        """
        if self.client is not None:
            return
        if not hasattr(ESMC, "from_pretrained"):
            raise RuntimeError("ESMC SDK not installed.")
        self.device = device
        logging.info(f"Loading ESMC client '{self.model_name}' onto {device} …")
        try:
            self.client = ESMC.from_pretrained(self.model_name).to(device).eval()
            logging.info("ESMC client loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ESMC client '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load ESMC client '{self.model_name}'") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        max_length: int = 10000,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Embed a single protein sequence.

        Parameters
        ----------
        input : str
            Amino acid sequence string.
        pooling_strategy : str
            One of 'mean', 'max', 'cls'.
        max_length : int
            Safety truncation limit to prevent OOM on very large proteins.
        **kwargs : Any
            Additional arguments (unused, for interface consistency).

        Returns
        -------
        np.ndarray
            1D embedding vector.
        """
        if self.client is None or self.device is None:
            raise RuntimeError("ESMC client not loaded; call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        sequence = input
        if len(sequence) > max_length:
            logging.warning(f"Sequence too long ({len(sequence)}). Truncating to {max_length}.")
            sequence = sequence[:max_length]

        prot = ESMProtein(sequence=sequence)
        tensor = self.client.encode(prot)

        cfg = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=False)
        out: LogitsOutput = self.client.logits(tensor, cfg)

        embs = out.embeddings
        if embs.dim() == 3 and embs.shape[0] == 1:
            embs = embs.squeeze(0)

        if pooling_strategy == "cls":
            pooled = embs[0]
        elif pooling_strategy == "max":
            pooled = torch.max(embs, dim=0)[0]
        else:
            pooled = torch.mean(embs, dim=0)

        return pooled.cpu().numpy()

    def embed_with_hidden_states(
        self,
        input: str,
        pooling_strategy: str = "mean",
        hidden_layers: Sequence[int] | None = None,
        max_length: int = 10000,
    ) -> dict[str, np.ndarray]:
        """
        Embed a single protein sequence and return hidden states.

        Parameters
        ----------
        input : str
            Amino acid sequence string.
        pooling_strategy : str
            One of 'mean', 'max', 'cls'.
        hidden_layers : Sequence[int], optional
            Specific layer indices to return. If None, returns all.
        max_length : int
            Safety truncation limit.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with 'embedding' (1D) and 'hidden_states' (3D) arrays.
        """
        if self.client is None or self.device is None:
            raise RuntimeError("ESMC client not loaded; call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        sequence = input
        if len(sequence) > max_length:
            logging.warning(f"Sequence too long ({len(sequence)}). Truncating to {max_length}.")
            sequence = sequence[:max_length]

        prot = ESMProtein(sequence=sequence)
        tensor = self.client.encode(prot)

        cfg = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        out: LogitsOutput = self.client.logits(tensor, cfg)

        embs = out.embeddings
        if embs.dim() == 3 and embs.shape[0] == 1:
            embs = embs.squeeze(0)

        if pooling_strategy == "cls":
            pooled = embs[0]
        elif pooling_strategy == "max":
            pooled = torch.max(embs, dim=0)[0]
        else:
            pooled = torch.mean(embs, dim=0)

        result: dict[str, np.ndarray] = {"embedding": pooled.cpu().numpy()}

        hidden = out.hidden_states
        if isinstance(hidden, tuple):
            hs = torch.stack(hidden, dim=0)
        else:
            hs = hidden
        while hs.dim() > 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)
        if hs.dim() == 4 and hs.shape[1] == 1:
            hs = hs.squeeze(1)

        if hidden_layers is not None:
            hs = hs[list(hidden_layers), :, :]
        result["hidden_states"] = hs.to(torch.float32).cpu().numpy()

        return result

    def embed_batch(
        self,
        inputs: list[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Embed a batch of protein sequences.

        Parameters
        ----------
        inputs : list[str]
            A list of amino-acid sequences.
        pooling_strategy : {'mean', 'max', 'cls'}, default='mean'
            Pooling to apply per sequence.
        **kwargs : Any
            Additional arguments forwarded to embed().

        Returns
        -------
        list[np.ndarray]
            A list of 1D embedding arrays, one per sequence.
        """
        return [
            self.embed(
                input=seq,
                pooling_strategy=pooling_strategy,
                **kwargs,
            )
            for seq in inputs
        ]


class ProtT5Wrapper(BaseModelWrapper):
    """Wrapper for ProtTrans ProtT5-XL-UniRef50 protein language model.

    Produces **1024-dimensional** per-protein embeddings by extracting
    residue-level representations from the T5 *encoder* and mean-pooling
    over the sequence length.

    This is the same embedding used as initial node features in the
    STRING-SPACE (sequence) and STRING-GNN frameworks (Hu et al., 2024).

    Supported model identifiers (Hugging Face):

    * ``Rostlab/prot_t5_xl_uniref50``  – full-precision (float32)
    * ``Rostlab/prot_t5_xl_half_uniref50-enc``  – half-precision encoder-only

    Parameters
    ----------
    model_path_or_name
        Hugging Face model name or local path.
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    # ProtT5-XL hidden dimension
    EMBEDDING_DIM = 1024

    def __init__(
        self,
        model_path_or_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: T5Tokenizer | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, device: torch.device) -> None:
        """Load the ProtT5 encoder and tokenizer onto *device*."""
        if self.model is not None:
            logging.debug("ProtT5 model already loaded.")
            return

        if not self.model_name:
            raise ValueError("model_path_or_name must be provided for ProtT5Wrapper.")

        logging.info("Loading ProtT5 model '%s' …", self.model_name)
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            # Use T5EncoderModel to avoid decoder overhead
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            self.model.to(device).eval()  # type: ignore[union-attr]
            self.device = device
            logging.info("ProtT5 model loaded successfully (dim=%d).", self.EMBEDDING_DIM)
        except Exception as e:
            logging.error("Failed to load ProtT5 model: %s", e)
            raise RuntimeError(f"Could not load ProtT5 model '{self.model_name}'") from e

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_sequence(sequence: str) -> str:
        """Prepare a protein sequence for ProtT5 tokenization.

        ProtT5 expects **spaces between each amino acid** and all rare /
        ambiguous residues (B, J, O, U, X, Z) mapped to ``X``.
        """
        # Remove whitespace
        sequence = sequence.strip().upper()
        # Replace rare amino acids with X
        sequence = re.sub(r"[UZOB]", "X", sequence)
        # Insert spaces between every amino acid
        return " ".join(list(sequence))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed a single protein sequence.

        Parameters
        ----------
        input
            Amino-acid sequence (e.g. ``"MTEYKLVVVG..."``).
        pooling_strategy
            One of ``"mean"`` (default), ``"max"``, or ``"cls"``.
        target_layer
            If given, extract hidden states from a specific encoder layer
            instead of the last one.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(1024,)`` (or model hidden dim).
        """
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("ProtT5 model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}")

        prepared = self._prepare_sequence(input)
        ids = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(target_layer is not None),
            )

            if target_layer is not None:
                hidden_states = outputs.hidden_states
                if not hidden_states:
                    raise ValueError("Model did not return hidden states.")
                if not (-len(hidden_states) <= target_layer < len(hidden_states)):
                    raise ValueError(f"Invalid target_layer {target_layer}.")
                emb = hidden_states[target_layer]
            else:
                emb = outputs.last_hidden_state  # (1, seq_len, 1024)

        # Squeeze batch dim
        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # (seq_len, 1024)

        # Mask out padding tokens for mean/max pooling
        if attention_mask is not None and pooling_strategy != "cls":
            mask = attention_mask.squeeze(0).unsqueeze(-1).bool()  # (seq_len, 1)
            emb = emb.masked_fill(~mask, 0.0)

        if pooling_strategy == "cls":
            pooled = emb[0]
        elif pooling_strategy == "max":
            if attention_mask is not None:
                emb = emb.masked_fill(~mask, float("-inf"))
            pooled = torch.max(emb, dim=0)[0]
        else:  # mean
            if attention_mask is not None:
                pooled = emb.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
            else:
                pooled = emb.mean(dim=0)

        return pooled.float().cpu().numpy()

    def embed_batch(
        self,
        inputs: list[str],
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Embed multiple protein sequences.

        Parameters
        ----------
        inputs
            List of amino-acid sequence strings.
        pooling_strategy
            One of ``"mean"`` (default), ``"max"``, ``"cls"``.
        target_layer
            Optional encoder layer index.

        Returns
        -------
        list[np.ndarray]
            One 1-D array of shape ``(1024,)`` per input sequence.
        """
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("ProtT5 model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}"
            )

        batch_size = kwargs.pop("batch_size", 16)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            prepared = [self._prepare_sequence(seq) for seq in chunk]
            tokenized = self.tokenizer(
                prepared,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=(target_layer is not None),
                )
                if target_layer is not None:
                    emb_tensor = outputs.hidden_states[target_layer]
                else:
                    emb_tensor = outputs.last_hidden_state

            for i in range(emb_tensor.shape[0]):
                seq_emb = emb_tensor[i]
                if attention_mask is not None:
                    mask = attention_mask[i].unsqueeze(-1).bool()
                else:
                    mask = None

                if pooling_strategy == "cls":
                    pooled = seq_emb[0]
                elif pooling_strategy == "max":
                    if mask is not None:
                        seq_emb = seq_emb.masked_fill(~mask, float("-inf"))
                    pooled = torch.max(seq_emb, dim=0)[0]
                else:
                    if mask is not None:
                        pooled = (seq_emb * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                    else:
                        pooled = seq_emb.mean(dim=0)

                all_embeddings.append(pooled.float().cpu().numpy())

        return all_embeddings


class ESM3Wrapper(BaseModelWrapper):
    """Wrapper for ESM3 protein language models (EvolutionaryScale SDK).

    ESM3 is a multimodal generative model for proteins. This wrapper
    uses the sequence track to produce per-residue embeddings and pools
    them into a single vector.

    Supported checkpoints:

    * ``esm3-small-2024-08`` (1.4B, open weights)
    * ``esm3-medium-2024-08`` (7B, Forge API)
    * ``esm3-large-2024-03`` (98B, Forge API)

    For open weights, the model runs locally. For larger models, pass
    a Forge API token via ``forge_token``.

    Requires the ``esm`` package: ``pip install esm``
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self,
        model_path_or_name: str = "esm3-small-2024-08",
        forge_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.forge_token = forge_token
        self._client: Any = None

    def load(self, device: torch.device) -> None:
        """Load the ESM3 model locally or connect to Forge API."""
        if self._client is not None:
            logging.debug("ESM3 already loaded.")
            return

        if not _HAVE_ESM3:
            raise ImportError(
                "The 'esm' package is required for ESM3. "
                "Install with: pip install esm"
            )

        name = self.model_name or "esm3-small-2024-08"

        if self.forge_token:
            import esm.sdk.client as esm_client
            self._client = esm_client(name, token=self.forge_token)
            logging.info("Connected to ESM3 Forge API: %s", name)
        else:
            self._client = ESM3.from_pretrained(name).to(device).eval()
            logging.info("Loaded ESM3 '%s' locally on %s", name, device)

        self.device = device
        self.model = self._client

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed a single protein sequence using ESM3.

        Uses the ESM3 encoder to get per-residue embeddings from the
        sequence track, then pools them.
        """
        if self._client is None:
            raise RuntimeError("ESM3 not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling '{pooling_strategy}'. "
                f"Choose from {self.available_pooling_strategies}"
            )

        protein = ESMProtein(sequence=input)
        protein_tensor = self._client.encode(protein)
        cfg = LogitsConfig(sequence=True, return_embeddings=True)
        out = self._client.logits(protein_tensor, cfg)

        embs = out.embeddings
        if embs.dim() == 3 and embs.shape[0] == 1:
            embs = embs.squeeze(0)

        if pooling_strategy == "cls":
            pooled = embs[0]
        elif pooling_strategy == "max":
            pooled = torch.max(embs, dim=0)[0]
        else:
            pooled = torch.mean(embs, dim=0)

        return pooled.float().cpu().numpy()

    def embed_batch(
        self,
        inputs: list[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Embed multiple protein sequences using ESM3."""
        if self._client is None:
            raise RuntimeError("ESM3 not loaded. Call load() first.")
        return [
            self.embed(seq, pooling_strategy=pooling_strategy, **kwargs)
            for seq in inputs
        ]
