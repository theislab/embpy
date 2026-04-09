import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    Defines the common interface for loading models, computing embeddings,
    and extracting hidden states from specific layers.

    Layer Extraction
    ----------------
    Models differ in how hidden states are accessed:

    * **HuggingFace models** (ESM2, BERT, etc.): pass ``output_hidden_states=True``
      to the forward call and index into the returned tuple.
    * **Non-HF models** (StripedHyena, custom architectures): use PyTorch forward
      hooks on the desired layer module.

    The :meth:`extract_hidden_states` method auto-detects the strategy.
    """

    model_type: Literal["dna", "protein", "molecule", "text", "ppi", "unknown"] = "unknown"
    available_pooling_strategies: list[str] = ["mean", "max", "median", "none"]  # Common defaults

    def __init__(self, model_path_or_name: str | None = None, **kwargs: Any):
        """
        Initialize the wrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Identifier for the specific model weights/config
            (e.g., Hugging Face name, local path).
        **kwargs : Any
            Additional configuration for the specific model.
        """
        self.model_name = model_path_or_name
        self.model: torch.nn.Module | None = None
        self.device: torch.device | None = None
        self.config = kwargs

    # =================================================================
    # Abstract interface
    # =================================================================

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load the model weights and move to the specified device."""

    @abstractmethod
    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the embedding for a single input string.

        Parameters
        ----------
        input : str
            The input data (DNA sequence, protein sequence, SMILES, or text).
        pooling_strategy : str
            The pooling strategy to use.
        **kwargs : Any
            Model-specific arguments (e.g., ``target_layer``).

        Returns
        -------
        np.ndarray
            The resulting embedding vector.
        """

    @abstractmethod
    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute embeddings for a batch of input strings.

        Parameters
        ----------
        inputs : Sequence[str]
            A list/tuple of input strings.
        pooling_strategy : str
            The pooling strategy to use.
        **kwargs : Any
            Model-specific arguments.

        Returns
        -------
        list[np.ndarray]
            A list of resulting embedding vectors.
        """

    # =================================================================
    # Pooling
    # =================================================================

    def _apply_pooling(self, embeddings: torch.Tensor, strategy: str) -> np.ndarray:
        """
        Apply the specified pooling strategy to token/residue embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor of shape ``(batch, seq_len, hidden_dim)`` or ``(seq_len, hidden_dim)``.
        strategy : str
            Pooling strategy (``'mean'``, ``'max'``, ``'cls'``, ``'median'``,
            ``'none'``).  ``'none'`` returns the raw tensor as-is.

        Returns
        -------
        np.ndarray
            Pooled embedding of shape ``(hidden_dim,)`` or ``(batch, hidden_dim)``,
            or raw ``(seq_len, hidden_dim)`` when ``strategy='none'``.
        """
        if strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling strategy '{strategy}'. Available: {self.available_pooling_strategies}")

        if strategy == "none":
            return embeddings.cpu().numpy()

        if embeddings.dim() == 3:  # Batch dimension present
            if strategy == "mean":
                pooled = embeddings.mean(dim=1)
            elif strategy == "max":
                pooled = embeddings.max(dim=1).values
            elif strategy == "cls":
                pooled = embeddings[:, 0, :]
            elif strategy == "median":
                pooled = embeddings[0, :]
            else:
                raise ValueError(f"Pooling strategy '{strategy}' not implemented for batched tensors.")
        elif embeddings.dim() == 2:  # No batch dimension
            if strategy == "mean":
                pooled = embeddings.mean(dim=0)
            elif strategy == "max":
                pooled = embeddings.max(dim=0).values
            elif strategy == "cls":
                pooled = embeddings[0, :]
            elif strategy == "median":
                pooled = embeddings[0, :]
            else:
                raise ValueError(f"Pooling strategy '{strategy}' not implemented for single tensors.")
        else:
            raise ValueError(f"Unsupported embedding tensor dimension: {embeddings.dim()}")

        return pooled.cpu().numpy()

    # =================================================================
    # Layer introspection
    # =================================================================

    def get_num_layers(self) -> int:
        """
        Return the number of hidden layers in the loaded model.

        Detection order:

        1. HuggingFace ``config.num_hidden_layers`` (or ``n_layer`` / ``num_layers``).
        2. Top-level ``model.blocks`` / ``model.layers`` ``ModuleList``.
        3. Encoder container ``model.encoder.layer``.

        Returns
        -------
        int
            The number of layers.

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        NotImplementedError
            If the layer count cannot be auto-detected. Override in subclass.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1) HuggingFace config
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                n = getattr(cfg, attr, None)
                if isinstance(n, int):
                    return n

        # 2) Direct layer containers
        for attr_name in ("blocks", "layers"):
            container = getattr(self.model, attr_name, None)
            if isinstance(container, torch.nn.ModuleList):
                return len(container)

        # 3) Encoder-based (BERT, ESM2, etc.)
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            for attr_name in ("layer", "layers"):
                container = getattr(encoder, attr_name, None)
                if isinstance(container, torch.nn.ModuleList):
                    return len(container)

        raise NotImplementedError(
            f"Cannot auto-detect number of layers for {type(self.model).__name__}. "
            "Override get_num_layers() in your wrapper subclass."
        )

    def _get_layer_modules(self) -> torch.nn.ModuleList:
        """
        Return the sequential layer modules of the model.

        Used by the hook-based hidden-state extractor. The detection mirrors
        :meth:`get_num_layers`.

        Returns
        -------
        torch.nn.ModuleList
            The iterable of layer modules.

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        NotImplementedError
            If the layers cannot be found. Override in subclass.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for attr_name in ("blocks", "layers"):
            container = getattr(self.model, attr_name, None)
            if isinstance(container, torch.nn.ModuleList):
                return container

        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            for attr_name in ("layer", "layers"):
                container = getattr(encoder, attr_name, None)
                if isinstance(container, torch.nn.ModuleList):
                    return container

        raise NotImplementedError(
            f"Cannot auto-detect layer modules for {type(self.model).__name__}. "
            "Override _get_layer_modules() in your wrapper subclass."
        )

    def _is_huggingface_model(self) -> bool:
        """
        Check whether the loaded model looks like a HuggingFace Transformers model.

        Returns ``True`` when the model has a ``.config`` attribute with a
        ``num_hidden_layers`` (or similar) field, which is the standard marker
        for HF ``PreTrainedModel`` instances.
        """
        if self.model is None:
            return False
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        return any(hasattr(cfg, a) for a in ("num_hidden_layers", "n_layer", "num_layers"))

    # =================================================================
    # Hidden-state extraction
    # =================================================================

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract hidden-state tensors from specific model layers.

        Automatically selects the extraction strategy:

        * **HuggingFace models**: calls the model with
          ``output_hidden_states=True`` and indexes into the returned tuple.
          Layer 0 is the embedding output; layers 1..N correspond to
          the N transformer blocks.
        * **Non-HF models**: registers PyTorch forward hooks on the layer
          modules returned by :meth:`_get_layer_modules`.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenised input, typically of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor, optional
            Attention mask (only used for HF models).
        layers : list[int], optional
            Layer indices to extract. Negative values count from the end
            (``-1`` = last layer).  If ``None``, all layers are returned.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from normalised layer index to a hidden-state tensor,
            each of shape ``(batch, seq_len, hidden_dim)``.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        IndexError
            If a requested layer index is out of range.

        Examples
        --------
        >>> wrapper.load(torch.device("cpu"))
        >>> ids = torch.tensor([[101, 2003, 102]])
        >>> states = wrapper.extract_hidden_states(ids, layers=[0, -1])
        >>> states[0].shape   # embedding layer output
        torch.Size([1, 3, 768])
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._is_huggingface_model():
            return self._extract_hidden_states_hf(input_ids, attention_mask, layers)
        return self._extract_hidden_states_hook(input_ids, layers)

    def _extract_hidden_states_hf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layers: list[int] | None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract hidden states from a HuggingFace model.

        Uses ``output_hidden_states=True``, which returns a tuple of
        ``(N+1)`` tensors (embedding output + N layer outputs).
        """
        model: Any = self.model
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        all_hidden: tuple[torch.Tensor, ...] = outputs.hidden_states
        n = len(all_hidden)

        if layers is None:
            target_layers = list(range(n))
        else:
            target_layers = list(layers)

        result: dict[int, torch.Tensor] = {}
        for idx in target_layers:
            norm_idx = idx if idx >= 0 else n + idx
            if not (0 <= norm_idx < n):
                raise IndexError(
                    f"Layer index {idx} out of range. "
                    f"Model has {n} hidden states (indices 0 to {n - 1}, or -{n} to -1)."
                )
            result[norm_idx] = all_hidden[norm_idx]

        return result

    def _extract_hidden_states_hook(
        self,
        input_tensor: torch.Tensor,
        layers: list[int] | None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract hidden states from non-HF models using forward hooks.

        Registers a hook on each requested layer module, runs the forward
        pass, captures the first output element (handling ``(tensor, cache)``
        tuples), and removes hooks afterwards.
        """
        layer_modules = self._get_layer_modules()
        n = len(layer_modules)

        if layers is None:
            target_layers = list(range(n))
        else:
            target_layers = list(layers)

        # Normalise negative indices
        norm_layers: list[int] = []
        for idx in target_layers:
            norm_idx = idx if idx >= 0 else n + idx
            if not (0 <= norm_idx < n):
                raise IndexError(
                    f"Layer index {idx} out of range. "
                    f"Model has {n} layers (indices 0 to {n - 1}, or -{n} to -1)."
                )
            norm_layers.append(norm_idx)

        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        for layer_idx in norm_layers:

            def _make_hook(li: int) -> Any:
                def hook_fn(module: Any, inp: Any, output: Any) -> None:
                    if isinstance(output, tuple):
                        captured[li] = output[0]
                    else:
                        captured[li] = output

                return hook_fn

            handle = layer_modules[layer_idx].register_forward_hook(_make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                self.model(input_tensor)  # type: ignore[misc]
        finally:
            for h in handles:
                h.remove()

        return captured

    # =================================================================
    # Convenience: embed from a specific layer
    # =================================================================

    def embed_from_layer(
        self,
        input: str,
        layer: int,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute a pooled embedding from a specific model layer.

        This is a convenience wrapper that calls :meth:`embed` with the
        ``target_layer`` keyword argument. Wrappers whose ``embed()``
        method already accepts ``target_layer`` (e.g. ESM2, TextLLM) work
        automatically. Other wrappers should override this method.

        Parameters
        ----------
        input : str
            The input string (sequence, SMILES, text).
        layer : int
            The layer index from which to extract the embedding.
            Negative values count from the end.
        pooling_strategy : str
            Pooling strategy to aggregate token-level embeddings.
        **kwargs : Any
            Forwarded to :meth:`embed`.

        Returns
        -------
        np.ndarray
            A 1D embedding vector from the requested layer.
        """
        return self.embed(input, pooling_strategy=pooling_strategy, target_layer=layer, **kwargs)

    def embed_all_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pooling_strategy: str = "mean",
    ) -> dict[int, np.ndarray]:
        """
        Extract pooled embeddings from every layer of the model.

        Useful for layer-wise analysis (e.g. probing, CKA similarity).

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenised input of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor, optional
            Attention mask (HF models only).
        pooling_strategy : str
            Pooling strategy applied per layer.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from layer index to a pooled 1D embedding (numpy array).
        """
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. "
                f"Available: {self.available_pooling_strategies}"
            )

        hidden_states = self.extract_hidden_states(
            input_ids, attention_mask=attention_mask, layers=None
        )

        result: dict[int, np.ndarray] = {}
        for layer_idx, tensor in hidden_states.items():
            result[layer_idx] = self._apply_pooling(tensor, pooling_strategy)

        return result
