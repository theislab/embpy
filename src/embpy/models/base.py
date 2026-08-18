from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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

    #: Whether the wrapped architecture computes attention weights that can be
    #: extracted via :meth:`extract_attention`. Set to ``False`` for
    #: attention-free architectures (state-space models such as Caduceus,
    #: implicit long-convolution models such as HyenaDNA, message-passing GNNs
    #: such as MiniMol), where attention weights do not exist and a request must
    #: fail loudly rather than return something meaningless.
    has_attention: bool = True

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
            Pooling strategy. Must also be listed in this wrapper's
            ``available_pooling_strategies``, otherwise a ``ValueError`` is
            raised — subclasses expose different subsets. Supported here:

            * ``'mean'`` — arithmetic mean over the token axis.
            * ``'max'`` — element-wise maximum over the token axis.
            * ``'median'`` — element-wise median over the token axis, using
              :func:`torch.median` semantics: for an *even* number of tokens
              this is the **lower** of the two middle values rather than
              their average, so it can differ from :func:`numpy.median`.
              (This matches the median pooling in the Enformer and Borzoi
              wrappers.)
            * ``'cls'`` — the first token's embedding, no aggregation.
            * ``'none'`` — the raw tensor, unpooled.

        Returns
        -------
        np.ndarray
            Pooled embedding of shape ``(hidden_dim,)`` for a 2D input, or
            ``(batch, hidden_dim)`` for a 3D input. When ``strategy='none'``
            the tensor is returned with its original shape, unpooled.
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
                pooled = embeddings.median(dim=1).values
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
                pooled = embeddings.median(dim=0).values
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

    def _model_device(self) -> torch.device:
        """Device the model's parameters actually live on.

        Preferred over ``self.device``, which records what was *requested* and can
        disagree with reality (e.g. a wrapper that falls back to CPU). Extraction
        inputs are usually built by a tokenizer, which returns CPU tensors, so they
        must be moved here or the forward pass fails on MPS/CUDA.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        try:
            device = next(self.model.parameters()).device
        except (StopIteration, AttributeError, TypeError):  # paramless or non-Module
            device = None
        # Guard the isinstance: mocks and stand-ins return something that is not a
        # real torch.device, and passing that to .to() fails confusingly.
        if isinstance(device, torch.device):
            return device
        return self.device if isinstance(self.device, torch.device) else torch.device("cpu")

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
        device = self._model_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
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
            input_tensor = input_tensor.to(self._model_device())
            with torch.no_grad():
                self.model(input_tensor)  # type: ignore[misc]
        finally:
            for h in handles:
                h.remove()

        return captured

    @staticmethod
    def _looks_like_attention(tensor: Any) -> bool:
        """Whether ``tensor`` has the shape and normalisation of attention weights.

        Attention is ``(batch, heads, seq, seq)`` or ``(batch, seq, seq)`` and each
        query row sums to 1. Checking the row sums matters: a hook sees every tensor a
        module emits, and a square activation would otherwise be mistaken for
        attention.
        """
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (3, 4):
            return False
        if tensor.shape[-1] != tensor.shape[-2]:
            return False
        with torch.no_grad():
            sums = tensor.float().sum(dim=-1)
            return bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))

    def _extract_attention_hook(
        self,
        input_tensor: torch.Tensor,
        layers: list[int] | None,
    ) -> dict[int, torch.Tensor]:
        """Extract attention from non-HuggingFace models using forward hooks.

        The counterpart to :meth:`_extract_hidden_states_hook`, using the same
        :meth:`_get_layer_modules` discovery. Two things make attention harder than
        hidden states:

        * Some modules only return weights when asked. ``torch.nn.MultiheadAttention``
          takes ``need_weights``, and ``nn.TransformerEncoderLayer`` hardcodes it to
          ``False`` internally, so a plain forward hook sees ``None``. A forward
          *pre*-hook flips that kwarg back on without touching the model definition.
        * Fused kernels never materialise the matrix at all. If a layer routes through
          ``F.scaled_dot_product_attention``, FlashAttention or a Triton kernel, the
          weights exist only inside the kernel and **no hook can recover them** -- such
          layers are simply absent from the returned dict.

        Returns
        -------
        dict[int, torch.Tensor]
            Layer index -> attention tensor, for whichever layers yielded one. Layers
            whose attention could not be observed are omitted rather than faked.
        """
        layer_modules = self._get_layer_modules()
        n = len(layer_modules)

        target_layers = list(range(n)) if layers is None else list(layers)
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

        def _find_attention(obj: Any, depth: int = 0) -> torch.Tensor | None:
            if depth > 2:
                return None
            if BaseModelWrapper._looks_like_attention(obj):
                return obj  # type: ignore[return-value]
            if isinstance(obj, tuple | list):
                for item in obj:
                    found = _find_attention(item, depth + 1)
                    if found is not None:
                        return found
            return None

        def _weight_flags(module: Any) -> dict[str, Any]:
            """Kwargs that make ``module`` return its attention weights.

            Attention modules that *can* hand back weights default to not doing so,
            and they do not agree on the spelling: ``torch.nn.MultiheadAttention``
            takes ``need_weights``, while LLM-Foundry-derived blocks (Tahoe's
            ``GroupedQueryAttention``) take ``needs_weights``. Both compute the
            matrix either way and simply drop it, so flipping the flag recovers it
            without touching the model.
            """
            import inspect

            try:
                params = inspect.signature(module.forward).parameters
            except (TypeError, ValueError):  # pragma: no cover - C-implemented forward
                params = {}

            flags: dict[str, Any] = {}
            if "need_weights" in params or isinstance(module, torch.nn.MultiheadAttention):
                flags["need_weights"] = True
                if "average_attn_weights" in params or isinstance(
                    module, torch.nn.MultiheadAttention
                ):
                    # Keep per-head resolution; averaged weights lose the head axis.
                    flags["average_attn_weights"] = False
            if "needs_weights" in params:
                flags["needs_weights"] = True
            return flags

        def _make_pre_hook(flags: dict[str, Any]) -> Any:
            # Ask modules that can return weights to actually do so.
            def pre_hook(module: Any, args: Any, kwargs: Any) -> Any:
                return args, {**kwargs, **flags}

            return pre_hook

        def _make_hook(li: int) -> Any:
            def hook_fn(module: Any, inp: Any, output: Any) -> None:
                found = _find_attention(output)
                if found is not None:
                    captured[li] = found.detach()

            return hook_fn

        for layer_idx in norm_layers:
            module = layer_modules[layer_idx]
            # module.modules() yields the module itself first, then its descendants.
            # Both are candidates: the weights are usually emitted by an inner
            # attention module, but a layer can also be the attention module itself.
            for sub in module.modules():
                handles.append(sub.register_forward_hook(_make_hook(layer_idx)))
                flags = _weight_flags(sub)
                if flags:
                    handles.append(
                        sub.register_forward_pre_hook(_make_pre_hook(flags), with_kwargs=True)
                    )

        try:
            input_tensor = input_tensor.to(self._model_device())
            with torch.no_grad():
                self.model(input_tensor)  # type: ignore[misc]
        finally:
            for h in handles:
                h.remove()

        return captured

    # =================================================================
    # Attention extraction
    # =================================================================

    @staticmethod
    @contextmanager
    def _eager_attention(model: Any) -> Iterator[None]:
        """Force eager attention for the duration of a forward pass.

        Yields with ``model`` configured to materialise attention weights, then
        restores whatever implementation it was using before. Older transformers
        releases have no ``set_attn_implementation``; there the config attribute
        is the supported switch, and releases predating both already default to
        eager, so the fallback is a no-op.
        """
        config = getattr(model, "config", None)
        previous = getattr(config, "_attn_implementation", None) if config is not None else None
        if previous is None or previous == "eager":
            yield
            return

        setter = getattr(model, "set_attn_implementation", None)
        try:
            if callable(setter):
                setter("eager")
            else:
                config._attn_implementation = "eager"
            yield
        finally:
            if callable(setter):
                try:
                    setter(previous)
                except Exception:  # noqa: BLE001 - restoring is best-effort
                    config._attn_implementation = previous
            else:
                config._attn_implementation = previous

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract per-layer attention weight matrices.

        Only HuggingFace models expose attention weights in a uniform way, via
        ``output_attentions=True``. The returned tuple has exactly one entry per
        transformer layer (there is **no** embedding-layer entry, unlike
        :meth:`extract_hidden_states`), so layer index ``i`` maps directly to
        the ``i``-th transformer block's attention.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenised input of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor, optional
            Attention mask of shape ``(batch, seq_len)``. Pass it for padded
            batches so masked positions do not leak attention mass.
        layers : list[int], optional
            Layer indices to return. Negative values count from the end
            (``-1`` = last layer). If ``None``, every layer is returned.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from normalised layer index to an attention tensor of shape
            ``(batch, n_heads, seq_len, seq_len)``. Each query row sums to 1.

        Raises
        ------
        RuntimeError
            If the model is not loaded, or if the model produced no attention
            weights (typically because it is using a fused attention kernel
            such as SDPA or FlashAttention that does not expose them).
        NotImplementedError
            If the wrapped architecture has no attention
            (:attr:`has_attention` is ``False``) or is not a HuggingFace model.
        IndexError
            If a requested layer index is out of range.

        Examples
        --------
        >>> wrapper.load(torch.device("cpu"))
        >>> ids = tok("MKT", return_tensors="pt")["input_ids"]
        >>> attn = wrapper.extract_attention(ids, layers=[-1])
        >>> attn[next(iter(attn))].shape  # (batch, heads, seq, seq)
        torch.Size([1, 20, 5, 5])
        """
        # has_attention is a class attribute, so it is knowable without weights.
        # Check it before the load guard: an architecture that cannot produce
        # attention at all should say so, rather than telling the caller to load
        # a model that would not help. This also gives the right answer for
        # wrappers that keep their module somewhere other than .model.
        if not self.has_attention:
            raise NotImplementedError(
                f"{type(self).__name__} wraps an architecture whose attention "
                "weights are not observable -- either it is attention-free "
                "(state-space, implicit convolution, graph message passing) or it "
                "computes attention with a fused kernel that never materialises "
                "the matrix. Use extract_hidden_states() for per-layer "
                "activations instead."
            )

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self._is_huggingface_model():
            # Non-HF models have no uniform output_attentions flag, so fall back to
            # forward hooks (mirroring extract_hidden_states). This succeeds only for
            # architectures that actually materialise the matrix; fused-kernel
            # attention (SDPA / FlashAttention / Triton) cannot be observed by any
            # hook, and yields nothing.
            captured = self._extract_attention_hook(input_ids, layers)
            if not captured:
                raise NotImplementedError(
                    f"{type(self).__name__} is not a HuggingFace model and no attention "
                    "weights could be captured by forward hooks. This normally means the "
                    "model uses a fused attention kernel (torch SDPA, FlashAttention or "
                    "Triton), which never materialises the attention matrix, so it cannot "
                    "be recovered without changing the model. Use extract_hidden_states() "
                    "for per-layer activations instead."
                )
            return captured

        model: Any = self.model
        device = self._model_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        # transformers >= 4.48 defaults most architectures to SDPA, and that fused
        # kernel returns no per-head weights: output_attentions=True then yields
        # None, which is indistinguishable from an architecture that has no
        # attention at all. Eager exists for every HF architecture that does have
        # attention, so run the forward pass under eager and restore the model's
        # own implementation afterwards.
        with self._eager_attention(model), torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
            )

        attns: tuple[torch.Tensor, ...] | None = getattr(outputs, "attentions", None)
        if attns is None or any(a is None for a in attns):
            raise RuntimeError(
                "The model returned no attention weights even under eager "
                "attention. This means the architecture hard-codes a fused "
                "attention kernel (FlashAttention or a Triton kernel) that never "
                "materialises the per-head matrix, so it cannot be recovered "
                "without changing the model. Use extract_hidden_states() for "
                "per-layer activations instead."
            )

        n = len(attns)  # one entry per transformer layer; NO embedding offset
        target_layers = list(range(n)) if layers is None else list(layers)

        result: dict[int, torch.Tensor] = {}
        for idx in target_layers:
            norm_idx = idx if idx >= 0 else n + idx
            if not (0 <= norm_idx < n):
                raise IndexError(
                    f"Layer index {idx} out of range. "
                    f"Model has {n} attention layers (indices 0 to {n - 1}, "
                    f"or -{n} to -1)."
                )
            result[norm_idx] = attns[norm_idx]

        return result

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
