import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Use the specific enformer-pytorch package you have as dependency
try:
    from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
except ImportError:
    logging.warning("enformer-pytorch not found. EnformerWrapper will not be functional.")
    from_pretrained = None  # type: ignore
    seq_indices_to_one_hot = None  # type: ignore

from .base import BaseModelWrapper


class EnformerWrapper(BaseModelWrapper):
    """
    Wrapper for the Enformer model (using enformer-pytorch).

    Embeds DNA sequences of a fixed length required by the model.
    """

    model_type = "dna"
    # Enformer trunk output is (batch, num_bins, hidden_dim). Pooling happens over bins.
    available_pooling_strategies = ["mean", "max"]  # Median is less common for high-dim embeddings

    # Enformer specific constants
    SEQUENCE_LENGTH = 196608  # Standard Enformer input length
    TRUNK_OUTPUT_DIM = 3072  # Dimension of the trunk embeddings

    def __init__(self, model_path_or_name: str = "EleutherAI/enformer-official-rough", **kwargs):
        """
        Initializes the Enformer wrapper.

        Args:
            model_path_or_name (str): Hugging Face model name or path for Enformer.
            **kwargs: Additional config (e.g., use_tf_gamma).
        """
        super().__init__(model_path_or_name, **kwargs)
        # Store Enformer specific config if needed
        self.use_tf_gamma = kwargs.get("use_tf_gamma", False)  # Example from gene_embeddings

    def load(self, device: torch.device):
        """Loads the Enformer model."""
        if self.model is not None:
            logging.warning(f"Model {self.model_name} already loaded.")
            return
        if from_pretrained is None or seq_indices_to_one_hot is None:  # Check both imports
            raise ImportError("Cannot load Enformer model: enformer-pytorch package or required functions not found.")

        logging.info(f"Loading Enformer model '{self.model_name}'...")
        try:
            # Load the model using the identifier stored during init
            self.model = from_pretrained(self.model_name, use_tf_gamma=self.use_tf_gamma)
            self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
            self.device = device
            logging.info(f"Enformer model '{self.model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"Failed to load Enformer model '{self.model_name}': {e}")
            self.model = None
            self.device = None
            raise RuntimeError(f"Failed to load Enformer model '{self.model_name}'.") from e

    def _preprocess_sequence(self, sequence: str, print_debug: bool = False) -> torch.Tensor:
        """

        Converts DNA sequence string to a one-hot encoded tensor and pads/truncates to the required Enformer input length (SEQUENCE_LENGTH).

        Aligns with enformer-pytorch tutorial input format.

        """
        if seq_indices_to_one_hot is None:
            raise RuntimeError("seq_indices_to_one_hot function not available from enformer-pytorch.")

        sequence = sequence.upper()
        # Mapping: A=0, C=1, G=2, T=3, N=4
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        # Default to 4 ('N') for any unknown characters
        seq_numeric = np.array([mapping.get(base, 4) for base in sequence])

        if print_debug:
            print(seq_numeric)
        # Create index tensor (dtype must be compatible with seq_indices_to_one_hot, usually long)
        seq_tensor_indices = torch.from_numpy(seq_numeric).long().unsqueeze(0)  # (1, L)

        # --- Padding/Truncation to SEQUENCE_LENGTH ---
        current_len = seq_tensor_indices.shape[1]
        target_len = self.SEQUENCE_LENGTH
        processed_indices = seq_tensor_indices  # Start with the original indices

        if current_len != target_len:
            if current_len < target_len:
                # Pad symmetrically with 4 ('N') index
                pad_total = target_len - current_len
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                # Pad with 4 (index for N) before one-hot encoding
                processed_indices = F.pad(seq_tensor_indices, pad=(pad_left, pad_right), mode="constant", value=4)
                logging.debug(f"Padded sequence indices from {current_len} to {target_len} using 'N' (4)")
            else:  # current_len > target_len
                # Truncate from center
                trim_total = current_len - target_len
                trim_left = trim_total // 2
                trim_end = trim_left + target_len
                processed_indices = seq_tensor_indices[:, trim_left:trim_end]
                logging.warning(f"Truncated sequence indices from {current_len} to {target_len} from center.")

        if processed_indices.shape[1] != target_len:
            raise ValueError(
                f"Preprocessing failed: final index tensor length is {processed_indices.shape[1]}, expected {target_len}"
            )

        # --- Convert final index tensor to one-hot ---
        # seq_indices_to_one_hot expects indices 0-4 (ACGTN)
        # Output shape: (batch, seq_len, alphabet_size=5)
        try:
            seq_tensor_one_hot = seq_indices_to_one_hot(processed_indices)
            if print_debug:
                print(seq_tensor_one_hot)
                one_hot = F.one_hot(processed_indices, num_classes=5)
                print(one_hot)
        except Exception as e:
            logging.error(f"Failed to convert sequence indices to one-hot: {e}")
            # Log the problematic tensor shape and dtype for debugging
            logging.error(
                f"Problematic index tensor shape: {processed_indices.shape}, dtype: {processed_indices.dtype}"
            )
            # Check for out-of-bounds indices (should be 0-4 after mapping/padding)
            if torch.any(processed_indices < 0) or torch.any(processed_indices > 4):
                logging.error("Index tensor contains values outside the expected 0-4 range.")
            raise RuntimeError("One-hot encoding failed.") from e

        # Expected shape: (1, SEQUENCE_LENGTH, 5)
        logging.debug(f"Preprocessing complete. Output one-hot tensor shape: {seq_tensor_one_hot.shape}")
        # The model expects float input for convolutions etc.
        return seq_tensor_one_hot.float()

    def embed(
        self,
        input: str,  # DNA sequence
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the Enformer trunk embedding for a single DNA sequence.

        Input sequence is converted to one-hot encoding before inference.

        Args:
            input (str): The input DNA sequence string.
            pooling_strategy (str): How to pool the trunk output bins ('mean', 'max').
            **kwargs: Additional arguments (currently unused).

        Returns
        -------
            np.ndarray: The resulting pooled embedding vector (shape: TRUNK_OUTPUT_DIM).
        """
        dna_sequence = input
        if self.model is None or self.device is None:
            raise RuntimeError("Enformer model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        # Preprocess (now returns one-hot tensor)
        seq_tensor_one_hot = self._preprocess_sequence(dna_sequence)
        seq_tensor_one_hot = seq_tensor_one_hot.to(self.device)

        # Inference (pass the one-hot tensor)
        with torch.no_grad():
            try:
                # Pass one-hot tensor to the model
                output = self.model(seq_tensor_one_hot, return_embeddings=True)

                # --- Output extraction logic remains the same ---
                if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                    embeddings_tensor = output[1]
                    logging.debug(f"Extracted trunk embeddings with shape: {embeddings_tensor.shape}")
                elif isinstance(output, torch.Tensor):
                    embeddings_tensor = output
                    logging.debug(
                        f"Received single tensor output (assumed trunk) with shape: {embeddings_tensor.shape}"
                    )
                # --- Fallback logic might need re-evaluation if default call expects different input ---
                # For now, assume return_embeddings=True path is primary
                else:
                    raise TypeError(
                        f"Could not extract trunk embeddings from model output (using one-hot input). Output type: {type(output)}"
                    )
                # --- End output extraction ---

            except Exception as e:
                logging.error(f"Error during Enformer model inference (with one-hot input): {e}")
                raise RuntimeError("Enformer inference failed.") from e

        # --- Pooling logic remains the same ---
        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[0] == 1:
            embeddings_tensor = embeddings_tensor.squeeze(0)
        elif embeddings_tensor.dim() != 2:
            raise ValueError(f"Unexpected embedding tensor dimensions: {embeddings_tensor.shape}")

        pooled_embedding = self._apply_pooling(embeddings_tensor, pooling_strategy)

        # Cleanup
        del seq_tensor_one_hot, embeddings_tensor, output
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pooled_embedding

    def embed_batch(
        self,
        inputs: Sequence[str],  # List of DNA sequences
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Computes Enformer trunk embeddings for a batch of DNA sequences efficiently.

        Input sequences are converted to one-hot encoding before inference.

        Args:
            inputs (Sequence[str]): A list/tuple of input DNA sequence strings.
            pooling_strategy (str): How to pool the trunk output bins ('mean', 'max').
            **kwargs: Additional arguments (currently unused).

        Returns
        -------
            list[np.ndarray]: A list of resulting pooled embedding vectors.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Enformer model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        logging.debug(f"Processing batch of {len(inputs)} DNA sequences (using one-hot).")

        # 1. Preprocess all sequences and collect one-hot tensors
        preprocessed_tensors = []
        for dna_sequence in inputs:
            try:
                # _preprocess_sequence now returns shape (1, SEQUENCE_LENGTH, 5)
                seq_tensor_one_hot = self._preprocess_sequence(dna_sequence)
                preprocessed_tensors.append(seq_tensor_one_hot)
            except Exception as e:
                logging.error(f"Failed to preprocess sequence: {dna_sequence[:50]}... Error: {e}")
                # Handle error - e.g., skip sequence, return None placeholder?
                # For simplicity now, let's assume preprocessing works or raises fully.
                # If partial failure is needed, return list[np.ndarray | None]
                raise  # Re-raise for now

        # 2. Stack into a batch tensor
        try:
            # Shape will be (B, SEQUENCE_LENGTH, 5)
            batch_tensor_one_hot = torch.cat(preprocessed_tensors, dim=0).to(self.device)
        except Exception as e:
            logging.error(f"Failed to concatenate preprocessed tensors into a batch: {e}")
            raise RuntimeError("Batch creation failed.") from e

        # 3. Run model inference (pass one-hot batch)
        embeddings_batch = None
        try:
            with torch.no_grad():
                output = self.model(batch_tensor_one_hot, return_embeddings=True)
                # --- Output extraction logic remains the same ---
                if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                    embeddings_batch = output[1]  # Shape: (B, num_bins, hidden_dim)
                elif isinstance(output, torch.Tensor):
                    embeddings_batch = output  # Shape: (B, num_bins, hidden_dim)
                else:
                    raise TypeError(
                        f"Unexpected output format from batched Enformer inference (one-hot): {type(output)}"
                    )
                # --- End adaptation ---
                logging.debug(f"Batched inference successful. Output shape: {embeddings_batch.shape}")
        except Exception as e:
            logging.error(f"Error during batched Enformer model inference: {e}")
            raise RuntimeError("Batched Enformer inference failed.") from e

        # 4. Apply pooling (logic remains the same, operates on embeddings_batch)
        results = []
        try:
            if pooling_strategy == "mean":
                pooled_batch = embeddings_batch.mean(dim=1)  # (B, hidden_dim)
            elif pooling_strategy == "max":
                pooled_batch = embeddings_batch.max(dim=1).values  # (B, hidden_dim)
            else:
                # Should be caught by initial check, but as safeguard
                raise ValueError(f"Pooling strategy '{pooling_strategy}' not implemented for batch.")

            results = list(pooled_batch.cpu().numpy())

        except Exception as e:
            logging.error(f"Error during batch pooling: {e}")
            # Handle pooling error - maybe return partial results or raise
            raise RuntimeError("Batch pooling failed.") from e

        # Cleanup
        del batch_tensor_one_hot, embeddings_batch, output, preprocessed_tensors
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if len(results) != len(inputs):
            # This indicates an issue, maybe during error handling or pooling
            logging.error(f"Batch processing mismatch: {len(inputs)} inputs, {len(results)} outputs.")
            # Decide how to handle: raise error, return None list?
            # For now, raise error
            raise RuntimeError("Mismatch between input batch size and output results count.")

        return results


# Add BorzoiWrapper here later if needed
# class BorzoiWrapper(BaseModelWrapper): ...
