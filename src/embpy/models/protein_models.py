# Placeholder for protein models like ESM
import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseModelWrapper


class ESMWrapper(BaseModelWrapper):
    """
    Wrapper for ESM protein language models (using Hugging Face Transformers).
    """

    model_type = "protein"
    # ESM often uses CLS token or mean pooling
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self, model_path_or_name: str = "facebook/esm2_t6_8M_UR50D", **kwargs
    ):  # TODO: I think maybe better use ESM3 official implementation...
        """
        Initializes the ESM wrapper.

        Args:
            model_path_or_name (str): Hugging Face model name
                                      (e.g., "facebook/esm2_t6_8M_UR50D", "facebook/esm2_t33_650M_UR50D").
            **kwargs: Additional config (unused for now).
        """
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer = None  # Store tokenizer

    def load(self, device: torch.device):
        """Loads the ESM model and tokenizer."""
        if self.model is not None:
            logging.debug("ESM model already loaded.")
            return

        if self.model_name is None:
            raise ValueError("model_path_or_name must be set for ESMWrapper.")

        logging.info(f"Loading ESM model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)  # change to ESM3 github
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device
            logging.warning("ESM loading is commented out. Need 'transformers' dependency.")
            # Placeholder for now:
            self.model = torch.nn.Module()  # Dummy model
            self.device = device
            logging.info("ESM model loaded (placeholder).")
        except NameError:
            logging.error("Failed to load ESM model: 'transformers' library not found.")
            raise RuntimeError("Please install 'transformers' to use ESM models: pip install transformers")
        except Exception as e:
            logging.error(f"Failed to load ESM model: {e}")
            raise RuntimeError(f"Could not load ESM model '{self.model_name}'") from e

    def _preprocess_sequence(self, sequence: str) -> Any:
        """Tokenizes a protein sequence."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        # Add truncation/padding as needed by the specific ESM model
        # return self.tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
        logging.warning("ESM preprocessing is commented out.")
        return {"input_ids": torch.randint(0, 33, (1, len(sequence)))}  # Dummy tokenization

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the ESM embedding for a single protein sequence.

        Args:
            input (str): The input protein sequence string.
            pooling_strategy (str): Pooling strategy ('mean', 'max', 'cls').
            target_layer (Optional[int]): Extract embeddings from this hidden layer index (e.g., -1 for last).
                                          If None, uses the default last_hidden_state.
            **kwargs: Additional arguments (unused).

        Returns
        -------
            np.ndarray: The resulting embedding vector.
        """
        protein_sequence = input
        if self.model is None or self.device is None:
            raise RuntimeError("ESM model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )
        if pooling_strategy == "cls" and (not self.tokenizer or not hasattr(self.tokenizer, "cls_token_id")):
            # Check if model/tokenizer actually supports CLS token
            logging.warning("CLS pooling requested, but CLS token might not be standard for this ESM model/tokenizer.")

        # Preprocess
        # tokenized_input = self._preprocess_sequence(protein_sequence)
        # input_ids = tokenized_input["input_ids"].to(self.device)
        # attention_mask = tokenized_input.get("attention_mask", None)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(self.device)
        logging.warning("ESM embedding uses placeholder preprocessing and inference.")
        # Placeholder inference
        input_ids = torch.randint(0, 33, (1, min(len(protein_sequence), 512)), device=self.device)  # Dummy input
        attention_mask = torch.ones_like(input_ids)

        # Inference
        with torch.no_grad():
            # outputs = self.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     output_hidden_states= (target_layer is not None) # Request hidden states only if needed
            # )

            # if target_layer is not None:
            #     if not outputs.hidden_states:
            #          raise ValueError("Requested target_layer, but model did not return hidden_states.")
            #     # Ensure layer index is valid
            #     if not (-len(outputs.hidden_states) <= target_layer < len(outputs.hidden_states)):
            #          raise ValueError(f"Invalid target_layer index {target_layer}. Model has {len(outputs.hidden_states)} layers.")
            #     embeddings_tensor = outputs.hidden_states[target_layer] # (B, L, D)
            # else:
            #     embeddings_tensor = outputs.last_hidden_state # (B, L, D)

            # Placeholder output
            hidden_dim = 320  # Example for 8M model
            embeddings_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], hidden_dim, device=self.device)

        # Squeeze batch dim
        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[0] == 1:
            embeddings_tensor = embeddings_tensor.squeeze(0)  # (L, D)

        # Apply pooling (handle attention mask for mean pooling if applicable)
        # Need to adjust pooling logic if using attention mask
        if pooling_strategy == "cls":
            # Assuming CLS token is at index 0
            pooled_embedding = embeddings_tensor[0, :].cpu().numpy()
        else:
            # Simple mean/max over sequence length for now
            pooled_embedding = self._apply_pooling(embeddings_tensor, pooling_strategy)

        # Cleanup
        del input_ids, attention_mask, embeddings_tensor  # , outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pooled_embedding

    # Override embed_batch for efficiency using tokenizer batch encoding and model batch inference
    # def embed_batch(...) -> list[np.ndarray]: ...
