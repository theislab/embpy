# embpy/models/base.py
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class ProteinEmbedder(ABC):
    """
    Abstract base class for protein embedding models.

    This class defines the common interface for all protein embedders,
    including methods for loading models, preprocessing sequences, and
    generating embeddings.
    """

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """
        Load the protein embedding model and all necessary resources.

        Args:
            device (torch.device): The device on which to load the model.
        """
        pass

    @abstractmethod
    def _preprocess_sequence(self, sequence: str) -> Any:
        """
        Preprocess a protein sequence for model inference.

        Args:
            sequence (str): The input protein sequence.

        Returns
        -------
            Any: Preprocessed data (e.g., tokenized inputs) ready for the model.
        """
        pass

    @abstractmethod
    def embed(self, sequence: str, **kwargs) -> np.ndarray:
        """
        Compute an embedding for a given protein sequence.

        Args:
            sequence (str): The input protein sequence.
            **kwargs: Additional parameters for the embedding process.

        Returns
        -------
            np.ndarray: A vector representing the protein embedding.
        """
        pass

    def embed_batch(self, sequences: list[str], **kwargs) -> list[np.ndarray]:
        """
        Compute embeddings for a batch of protein sequences.

        This default implementation calls `embed` on each individual sequence.
        Override this method for more efficient batch processing if supported.

        Args:
            sequences (List[str]): List of protein sequence strings.
            **kwargs: Additional parameters for the embedding process.

        Returns
        -------
            List[np.ndarray]: A list containing the embedding for each sequence.
        """
        return [self.embed(seq, **kwargs) for seq in sequences]
