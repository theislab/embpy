# Placeholder for small molecule models (e.g., ChemBERTa, MolFormer)
import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseModelWrapper


class ChembertaWrapper(BaseModelWrapper):  # Example name
    """
    Wrapper for ChemBERTa-like models for SMILES strings.
    """

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
            logging.warning("SMILES model loading is commented out. Need 'transformers' dependency.")
            # Placeholder for now:
            self.model = torch.nn.Module()  # Dummy model
            self.device = device
            logging.info("SMILES model loaded (placeholder).")
        except NameError:
            logging.error("Failed to load SMILES model: 'transformers' library not found.")
            raise RuntimeError("Please install 'transformers' to use SMILES models: pip install transformers")
        except Exception as e:
            logging.error(f"Failed to load SMILES model: {e}")
            raise RuntimeError(f"Could not load SMILES model '{self.model_name}'") from e

    def _preprocess_smiles(self, smiles: str) -> Any:
        """Tokenizes a SMILES string."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        # return self.tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        logging.warning("SMILES preprocessing is commented out.")
        return {"input_ids": torch.randint(0, 100, (1, len(smiles)))}  # Dummy tokenization

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
        # tokenized_input = self._preprocess_smiles(smiles)
        # input_ids = tokenized_input["input_ids"].to(self.device)
        # attention_mask = tokenized_input.get("attention_mask", None)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(self.device)
        logging.warning("SMILES embedding uses placeholder preprocessing and inference.")
        # Placeholder inference
        input_ids = torch.randint(0, 100, (1, min(len(smiles), 128)), device=self.device)  # Dummy input
        attention_mask = torch.ones_like(input_ids)

        # Inference
        with torch.no_grad():
            # outputs = self.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     output_hidden_states= (target_layer is not None)
            # )
            # if target_layer is not None:
            #     embeddings_tensor = outputs.hidden_states[target_layer]
            # else:
            #     embeddings_tensor = outputs.last_hidden_state

            # Placeholder output
            hidden_dim = 768  # Example for base models
            embeddings_tensor = torch.randn(input_ids.shape[0], input_ids.shape[1], hidden_dim, device=self.device)

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

    # Override embed_batch for efficiency
    # def embed_batch(...) -> list[np.ndarray]: ...
