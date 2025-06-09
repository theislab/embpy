from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    Defines the common interface for loading models and computing embeddings.
    """

    model_type: Literal["dna", "protein", "molecule", "text", "unknown"] = "unknown"
    available_pooling_strategies: list[str] = ["mean", "max"]  # Common defaults

    def __init__(self, model_path_or_name: str | None = None, **kwargs):
        """
        Initializes the wrapper.

        Args:
            model_path_or_name (Optional[str]): Identifier for the specific model weights/config
                                                (e.g., Hugging Face name, local path).
            **kwargs: Additional configuration for the specific model.
        """
        self.model_name = model_path_or_name
        self.model: torch.nn.Module | None = None
        self.device: torch.device | None = None
        self.config = kwargs

    @abstractmethod
    def load(self, device: torch.device):
        """
        Loads the model weights and moves the model to the specified device.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def embed(
        self,
        input: str,  # Input sequence (DNA, protein), SMILES, or general text
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the embedding for a single input string.

        Must be implemented by subclasses.

        Args:
            input (str): The input data (DNA sequence, protein sequence, SMILES string, or general text).
            pooling_strategy (str): The pooling strategy to use.
            **kwargs: Model-specific arguments (e.g., layer selection).

        Returns
        -------
            np.ndarray: The resulting embedding vector.

        Raises
        ------
            ValueError: If pooling_strategy is invalid or model not loaded.
        """
        pass

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Computes embeddings for a batch of input strings.

        This is a basic loop implementation. Subclasses should override this
        for efficient batch processing if possible.

        Args:
            inputs (Sequence[str]): A list/tuple of input strings (sequences, SMILES, text).
            pooling_strategy (str): The pooling strategy to use.
            **kwargs: Model-specific arguments.

        Returns
        -------
            list[np.ndarray]: A list of resulting embedding vectors.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        results = []
        # Basic loop - override in subclasses for efficiency!
        for single_input in inputs:
            # Note: Error handling for individual sequences in batch might be needed
            results.append(self.embed(single_input, pooling_strategy=pooling_strategy, **kwargs))
        return results

    def _apply_pooling(self, embeddings: torch.Tensor, strategy: str) -> np.ndarray:
        """
        Applies the specified pooling strategy to token/residue embeddings.

        Args:
            embeddings (torch.Tensor): Tensor of embeddings (batch, seq_len, hidden_dim)
                                       or (seq_len, hidden_dim).
            strategy (str): Pooling strategy ('mean', 'max', 'cls', etc.).

        Returns
        -------
            np.ndarray: Pooled embedding (batch, hidden_dim) or (hidden_dim,).
        """
        if strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling strategy '{strategy}'. Available: {self.available_pooling_strategies}")

        if embeddings.dim() == 3:  # Batch dimension present
            if strategy == "mean":
                pooled = embeddings.mean(dim=1)
            elif strategy == "max":
                pooled = embeddings.max(dim=1).values
            # Add other strategies like 'cls' if applicable (e.g., index 0)
            elif strategy == "cls":
                pooled = embeddings[:, 0, :]
            else:
                # Should be caught by the check above, but as fallback
                raise ValueError(f"Pooling strategy '{strategy}' not implemented for batched tensors.")
        elif embeddings.dim() == 2:  # No batch dimension
            if strategy == "mean":
                pooled = embeddings.mean(dim=0)
            elif strategy == "max":
                pooled = embeddings.max(dim=0).values
            elif strategy == "cls":
                pooled = embeddings[0, :]
            else:
                raise ValueError(f"Pooling strategy '{strategy}' not implemented for single tensors.")
        else:
            raise ValueError(f"Unsupported embedding tensor dimension: {embeddings.dim()}")

        return pooled.cpu().numpy()
