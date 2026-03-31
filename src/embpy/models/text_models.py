import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseModelWrapper


class TextLLMWrapper(BaseModelWrapper):
    """Wrapper for general-purpose text embedding models"""

    model_type = "text"
    available_pooling_strategies = ["mean", "max", "cls", "last_token"]

    def __init__(
        self,
        model_path_or_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int | None = None,
        cls_token_position: int = 0,
        **kwargs,
    ):
        """
        Initializes the Text LLM wrapper.

        Args:
            model_path_or_name (str): Hugging Face model name or Sentence-Transformer name.
            max_length (Optional[int]): Maximum sequence length. If None, will be determined
                                      from model config or default to 512.
            cls_token_position (int): Position of CLS token (default: 0 for BERT-style models).
            **kwargs: Additional config.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None
        self.max_length = max_length
        self.cls_token_position = cls_token_position

    def load(self, device: torch.device):
        """Loads the text model and potentially tokenizer."""
        if self.model is not None:
            logging.debug("Text model already loaded.")
            return

        if self.model_name is None:
            raise ValueError("model_path_or_name must be set for TextLLMWrapper.")

        logging.info(f"Loading text model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device

            # Determine max_length if not provided
            if self.max_length is None:
                model_max = getattr(self.model.config, "max_position_embeddings", None)
                if model_max is None:
                    model_max = getattr(self.tokenizer, "model_max_length", None)

                if model_max is None or model_max > 1024:
                    model_max = 512

                self.max_length = int(model_max)

            # Ensure tokenizer respects our max_length
            self.tokenizer.model_max_length = self.max_length
            logging.info(f"Using max_length={self.max_length}, cls_token_position={self.cls_token_position}")
        except Exception as e:
            logging.error(f"Failed to load text model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load text model '{self.model_name}'") from e

    def _get_cls_embedding(self, embeddings_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts CLS token embedding from the sequence.

        Args:
            embeddings_tensor: Tensor of shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)

        Returns
        -------
            torch.Tensor: CLS token embedding
        """
        if embeddings_tensor.dim() == 3:  # Batch dimension present
            return embeddings_tensor[:, self.cls_token_position, :]
        else:  # Single sequence
            return embeddings_tensor[self.cls_token_position, :]

    def _preprocess_text_hf(self, text: str) -> Any:
        """Tokenizes text using Hugging Face tokenizer."""
        if not self.tokenizer:
            raise RuntimeError("Hugging Face Tokenizer not loaded.")
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,  # For HF models if needed
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the embedding for a single text string.

        Args:
            input (str): The input text string.
            pooling_strategy (str): Pooling strategy ('mean', 'max', 'cls'). Sentence-Transformers typically uses 'mean'.
            target_layer (Optional[int]): For HF models, extract embeddings from this hidden layer index.
            **kwargs: Additional arguments (e.g., `normalize_embeddings` for SentenceTransformer).

        Returns
        -------
            np.ndarray: The resulting embedding vector.
        """
        text = input
        if self.model is None or self.device is None:
            raise RuntimeError("Text model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        # Preprocess
        tokenized_input = self._preprocess_text_hf(text)
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=(target_layer is not None)
            )
            if target_layer is not None:
                embeddings_tensor = outputs.hidden_states[target_layer]  # (B, L, D)
            else:
                embeddings_tensor = outputs.last_hidden_state  # (B, L, D)

        # Squeeze batch dim
        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[0] == 1:
            embeddings_tensor = embeddings_tensor.squeeze(0)  # (L, D)

        # Apply pooling (Needs attention mask awareness for HF mean pooling)
        if pooling_strategy == "cls":
            pooled_embedding = self._get_cls_embedding(embeddings_tensor).cpu().numpy()
        elif pooling_strategy == "last_token":
            pooled_embedding = self._last_token_pool(embeddings_tensor, attention_mask).cpu().numpy()
        else:
            # Apply proper pooling with attention mask
            if pooling_strategy == "mean" and attention_mask is not None:
                # Mask-aware mean pooling
                mask = attention_mask.squeeze(0).unsqueeze(-1)  # (L, 1)
                sum_embeddings = torch.sum(embeddings_tensor * mask, dim=0)
                sum_mask = torch.sum(mask, dim=0)
                pooled_embedding = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                # Simple mean/max over sequence length
                pooled_embedding = self._apply_pooling(embeddings_tensor, pooling_strategy)

        # Cleanup
        del input_ids, attention_mask, embeddings_tensor, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pooled_embedding

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Computes embeddings for a batch of text strings efficiently.

        Args:
            inputs (Sequence[str]): A list/tuple of text strings.
            pooling_strategy (str): Pooling strategy ('mean', 'max', 'cls').
            target_layer (Optional[int]): For HF models, extract embeddings from this hidden layer index.
            batch_size (Optional[int]): Maximum batch size for processing. If None, processes all inputs at once.
                                   Use smaller values (e.g., 16-64) to avoid OOM errors.
            **kwargs: Additional arguments for the model.

        Returns
        -------
            list[np.ndarray]: A list of resulting embedding vectors.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Text model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )
        if not inputs:
            return []

        # If batch_size is specified and smaller than input size, process in chunks
        if batch_size is not None and len(inputs) > batch_size:
            logging.info(f"Processing {len(inputs)} inputs in chunks of {batch_size}")
            results = []
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                chunk_results = self._process_batch_chunk(chunk, pooling_strategy, target_layer, **kwargs)
                results.extend(chunk_results)
            return results
        else:
            # Process all inputs at once (original behavior)
            return self._process_batch_chunk(inputs, pooling_strategy, target_layer, **kwargs)

    def _process_batch_chunk(
        self,
        inputs: Sequence[str],
        pooling_strategy: str,
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Process a single batch chunk."""
        logging.debug(f"Processing batch chunk of {len(inputs)} text inputs")

        # Tokenize all inputs in the chunk
        tokenized_batch = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized_batch["input_ids"].to(self.device)
        attention_mask = tokenized_batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=(target_layer is not None)
            )
            if target_layer is not None:
                embeddings_tensor = outputs.hidden_states[target_layer]  # (B, L, D)
            else:
                embeddings_tensor = outputs.last_hidden_state  # (B, L, D)

        # Apply pooling for each item in batch
        results = []
        batch_size = embeddings_tensor.shape[0]

        for i in range(batch_size):
            seq_embeddings = embeddings_tensor[i]
            seq_mask = None if attention_mask is None else attention_mask[i]

            if pooling_strategy == "cls":
                pooled_embedding = seq_embeddings[self.cls_token_position, :].cpu().numpy()
            elif pooling_strategy == "last_token":
                # For batch processing, we need to handle single sequence
                single_hidden = seq_embeddings.unsqueeze(0)  # Add batch dim
                single_mask = seq_mask.unsqueeze(0) if seq_mask is not None else None
                pooled_embedding = self._last_token_pool(single_hidden, single_mask).cpu().numpy()
            else:
                if pooling_strategy == "mean" and seq_mask is not None:
                    mask = seq_mask.unsqueeze(-1)
                    sum_embeddings = torch.sum(seq_embeddings * mask, dim=0)
                    sum_mask = torch.sum(mask, dim=0)
                    pooled_embedding = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    pooled_embedding = self._apply_pooling(seq_embeddings, pooling_strategy)

            results.append(pooled_embedding)

        # Cleanup
        del input_ids, attention_mask, embeddings_tensor, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return results

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool by taking the last token's embedding, handling both left and right padding.

        Args:
            last_hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
            attention_mask: Tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns
        -------
            torch.Tensor: Last token embeddings of shape (batch_size, hidden_dim) or (hidden_dim,)
        """
        # Handle single sequence case
        if last_hidden_states.dim() == 2:
            last_hidden_states = last_hidden_states.unsqueeze(0)  # Add batch dim
            attention_mask = attention_mask.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            result = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            result = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        if squeeze_batch:
            result = result.squeeze(0)

        return result


class LlamaEmbeddingWrapper(TextLLMWrapper):
    """Extract embeddings from LLaMA decoder-only models.

    LLaMA models are autoregressive (decoder-only), so the natural
    embedding is the **last non-padding token** representation. This
    wrapper loads the model via ``AutoModelForCausalLM`` in bfloat16
    for memory efficiency and defaults to ``last_token`` pooling.

    Requires a HuggingFace token for gated models -- set ``HF_TOKEN``
    environment variable or run ``huggingface-cli login``.

    Parameters
    ----------
    model_path_or_name
        HuggingFace model ID (e.g. ``"meta-llama/Llama-3.2-3B"``).
    max_length
        Maximum sequence length. Defaults to 4096.
    """

    available_pooling_strategies = ["last_token", "mean", "max"]

    def __init__(
        self,
        model_path_or_name: str = "meta-llama/Llama-3.2-3B",
        max_length: int | None = 4096,
        **kwargs,
    ):
        super().__init__(model_path_or_name, max_length=max_length, **kwargs)

    def load(self, device: torch.device):
        """Load a LLaMA model in bfloat16 for embedding extraction."""
        if self.model is not None:
            return

        if self.model_name is None:
            raise ValueError("model_path_or_name must be set.")

        import os

        logging.info("Loading LLaMA '%s' in bfloat16...", self.model_name)
        try:
            from transformers import AutoModelForCausalLM

            token = os.environ.get("HF_TOKEN")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=token,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device} if device.type == "cuda" else None,
                token=token,
            )
            if device.type != "cuda":
                self.model = self.model.to(device)
            self.model.eval()
            self.device = device

            if self.max_length is None:
                self.max_length = 4096
            self.tokenizer.model_max_length = self.max_length

            logging.info(
                "LLaMA loaded: %s, dtype=bfloat16, max_length=%d",
                self.model_name, self.max_length,
            )
        except Exception as e:
            logging.error("Failed to load LLaMA '%s': %s", self.model_name, e)
            raise RuntimeError(f"Could not load LLaMA '{self.model_name}'") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "last_token",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute embedding from a LLaMA model.

        Default pooling is ``last_token`` (last non-padding token),
        which is the natural choice for decoder-only models.
        """
        text = input
        if self.model is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling '{pooling_strategy}'. "
                f"Available: {self.available_pooling_strategies}"
            )

        tokenized = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if target_layer is not None:
                hidden = outputs.hidden_states[target_layer]
            else:
                hidden = outputs.hidden_states[-1]

        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)

        if pooling_strategy == "last_token":
            h = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
            pooled = self._last_token_pool(h, attention_mask).cpu().float().numpy()
            if pooled.ndim > 1:
                pooled = pooled.squeeze(0)
        elif pooling_strategy == "mean":
            mask = attention_mask.squeeze(0).unsqueeze(-1).float()
            pooled = (torch.sum(hidden * mask, dim=0) / torch.sum(mask, dim=0)).cpu().float().numpy()
        elif pooling_strategy == "max":
            pooled = hidden.max(dim=0).values.cpu().float().numpy()
        else:
            pooled = self._apply_pooling(hidden, pooling_strategy)

        del input_ids, attention_mask, hidden, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pooled
