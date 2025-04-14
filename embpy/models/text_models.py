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
    # Common pooling strategies for text models
    available_pooling_strategies = ["mean", "max", "cls"]  # Add 'cls'

    def __init__(self, model_path_or_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        """
        Initializes the Text LLM wrapper.

        Args:
            model_path_or_name (str): Hugging Face model name or Sentence-Transformer name.
                                      Defaults to a common Sentence-Transformer model.
            **kwargs: Additional config.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None  # For standard transformers
        # SentenceTransformer handles tokenization internally

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
            logging.info("Hugging Face Transformers model loaded successfully.")

        except ImportError as e:
            logging.error(f"Failed to load text model: Dependency missing ({e.name}).")
            raise RuntimeError(f"Please install '{e.name}' to use this text model.") from e
        except Exception as e:
            logging.error(f"Failed to load text model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load text model '{self.model_name}'") from e

    def _preprocess_text_hf(self, text: str) -> Any:
        """Tokenizes text using Hugging Face tokenizer."""
        if not self.tokenizer:
            raise RuntimeError("Hugging Face Tokenizer not loaded.")
        # Adjust truncation and padding
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

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
            # Assuming CLS token is at index 0
            pooled_embedding = embeddings_tensor[0, :].cpu().numpy()
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
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Computes embeddings for a batch of text strings efficiently.

        Args:
            inputs (Sequence[str]): A list/tuple of text strings.
            pooling_strategy (str): Pooling strategy ('mean', 'max', 'cls').
            target_layer (Optional[int]): For HF models, extract embeddings from this hidden layer index.
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

        logging.debug(f"Processing batch of {len(inputs)} text inputs")

        # Tokenize all inputs in a batch
        tokenized_batch = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
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
            # Extract single sequence embeddings and attention mask
            seq_embeddings = embeddings_tensor[i]  # (L, D)
            seq_mask = None if attention_mask is None else attention_mask[i]  # (L)

            # Apply pooling based on strategy
            if pooling_strategy == "cls":
                # CLS token is at index 0
                pooled_embedding = seq_embeddings[0, :].cpu().numpy()
            else:
                # Apply proper pooling with attention mask
                if pooling_strategy == "mean" and seq_mask is not None:
                    # Mask-aware mean pooling
                    mask = seq_mask.unsqueeze(-1)  # (L, 1)
                    sum_embeddings = torch.sum(seq_embeddings * mask, dim=0)
                    sum_mask = torch.sum(mask, dim=0)
                    pooled_embedding = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    # Use the existing _apply_pooling method
                    pooled_embedding = self._apply_pooling(seq_embeddings, pooling_strategy)

            results.append(pooled_embedding)

        # Cleanup
        del input_ids, attention_mask, embeddings_tensor, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return results
