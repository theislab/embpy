import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ——————————————————————————————————————————————————————————————————————————
#                       EVO2 WRAPPER
# ——————————————————————————————————————————————————————————————————————————

try:
    from evo2 import Evo2

    _HAVE_EVO2 = True
except ImportError:
    _HAVE_EVO2 = False
    Evo2 = None  # type: ignore

# ——————————————————————————————————————————————————————————————————————————
#                       ENFORMER WRAPPER
# ——————————————————————————————————————————————————————————————————————————

# TODO: This 2 wrappers can be merged with some condition logic

# Use the specific enformer-pytorch package you have as a dependency
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

    This class handles:
      1. Padding or truncating an arbitrary-length DNA string to exactly 196,608 base pairs.
      2. Converting the padded/truncated index tensor to a 5-channel one-hot tensor.
      3. Running the Enformer forward pass with `return_embeddings=True` to obtain trunk embeddings.
      4. Pooling over the genomic bins dimension (mean or max) to yield a final embedding of size 3072.

    Attributes
    ----------
    SEQUENCE_LENGTH : int
        Required fixed input length (196,608 bp) for Enformer.
    TRUNK_OUTPUT_DIM : int
        Hidden dimension size of Enformer trunk embeddings (3072).
    model_type : Literal["dna"]
        Indicates that this wrapper expects DNA sequence inputs.
    available_pooling_strategies : list[str]
        Supported pooling strategies ("mean", "max").
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "median"]

    SEQUENCE_LENGTH = 196_608
    TRUNK_OUTPUT_DIM = 3072

    def __init__(self, model_path_or_name: str = "EleutherAI/enformer-official-rough", **kwargs):
        """
        Initialize the EnformerWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Hugging Face model identifier or local path for Enformer weights.
            Defaults to "EleutherAI/enformer-official-rough".
        **kwargs : Any
            Additional configuration options (e.g., use_tf_gamma: bool).
        """
        super().__init__(model_path_or_name, **kwargs)
        self.use_tf_gamma = kwargs.get("use_tf_gamma", False)

    def load(self, device: torch.device):
        """
        Load the Enformer model onto the specified device.

        This method uses `enformer_pytorch.from_pretrained` to instantiate the model,
        moves it to the given device, and sets it to evaluation mode.

        Parameters
        ----------
        device : torch.device
            The target device for model inference (e.g., torch.device("cuda") or torch.device("cpu")).

        Raises
        ------
        ImportError
            If `enformer_pytorch` or `seq_indices_to_one_hot` is not available.
        RuntimeError
            If the model fails to load for any other reason.
        """
        if self.model is not None:
            logging.warning(f"Enformer '{self.model_name}' already loaded.")
            return
        if from_pretrained is None or seq_indices_to_one_hot is None:
            raise ImportError("Cannot load Enformer: enformer-pytorch or seq_indices_to_one_hot missing.")

        logging.info(f"Loading Enformer model '{self.model_name}' …")
        try:
            self.model = from_pretrained(self.model_name, use_tf_gamma=self.use_tf_gamma)
            self.model = self.model.to(device).eval()
            self.device = device
            logging.info(f"Enformer '{self.model_name}' loaded on {device}.")
        except Exception as e:
            logging.error(f"Failed to load Enformer '{self.model_name}': {e}")
            self.model = None
            self.device = None
            raise RuntimeError(f"Could not load Enformer '{self.model_name}'.") from e

    def _preprocess_sequence(self, sequence: str) -> torch.Tensor:
        """
        Convert an arbitrary-length DNA string into a one-hot tensor of shape (1, 5, 196608).

        Steps:
          1. Uppercase the input string and map characters A/C/G/T/N → 0/1/2/3/4.
          2. Pad (with index 4) or center-truncate the index tensor to length 196,608.
          3. Use `seq_indices_to_one_hot` to obtain a (1, 196608, 5) tensor, then permute to (1, 5, 196608).

        Parameters
        ----------
        sequence : str
            Raw DNA sequence (e.g., "ACGTN...").

        Returns
        -------
        torch.Tensor
            A float tensor of shape (1, 5, 196608), suitable for Enformer input.

        Raises
        ------
        RuntimeError
            If `seq_indices_to_one_hot` is not available.
        ValueError
            If, after padding/truncation, the length is not exactly 196,608.
        RuntimeError
            If one-hot conversion fails for any reason.
        """
        if seq_indices_to_one_hot is None:
            raise RuntimeError("seq_indices_to_one_hot not available from enformer-pytorch.")

        seq = sequence.upper()
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        idx_list = [mapping.get(b, 4) for b in seq]
        idx_tensor = torch.tensor(idx_list, dtype=torch.long).unsqueeze(0)  # (1, L_in)

        current_len = idx_tensor.shape[1]
        target_len = self.SEQUENCE_LENGTH

        if current_len < target_len:
            pad_total = target_len - current_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            idx_tensor = F.pad(idx_tensor, (pad_left, pad_right), mode="constant", value=4)
            logging.debug(f"Padded Enformer indices from {current_len}→{target_len} with 'N' (4).")
        elif current_len > target_len:
            trim_total = current_len - target_len
            trim_left = trim_total // 2
            idx_tensor = idx_tensor[:, trim_left : trim_left + target_len]
            logging.warning(f"Truncated Enformer indices from {current_len}→{target_len} (center‐crop).")

        if idx_tensor.shape[1] != target_len:
            raise ValueError(f"Enformer preprocessing error: final length {idx_tensor.shape[1]} != {target_len}")

        try:
            one_hot = seq_indices_to_one_hot(idx_tensor)  # (1, 196608, 5)
            # one_hot = oh.permute(0, 2, 1).float()  # (1, 5, 196608)
        except Exception as e:
            logging.error(f"One-hot encoding failed: {e}")
            raise RuntimeError("Failed to one-hot encode Enformer input.") from e

        return one_hot

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the Enformer trunk embedding for a single DNA sequence.

        1. Preprocess the raw DNA string → one-hot (1, 5, 196608).
        2. Run model(one_hot, return_embeddings=True) → trunk tensor of shape (1, num_bins, 3072).
        3. Squeeze batch dim → (num_bins, 3072), then pool over num_bins (mean or max).
        4. Return a NumPy array of shape (3072,).

        Parameters
        ----------
        input : str
            The DNA sequence string.
        pooling_strategy : str, default "mean"
            “mean” or “max” pooling over genomic bins.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length 3072 representing the pooled Enformer embedding.

        Raises
        ------
        RuntimeError
            If the model hasn’t been loaded (`load()` not called) or pooling strategy is invalid.
        TypeError
            If the model output format is unexpected.
        ValueError
            If trunk tensor has unexpected dimensions.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Enformer model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling strategy:'{pooling_strategy}'")

        one_hot = self._preprocess_sequence(input).to(self.device)  # (1, 5, 196608)

        with torch.no_grad():
            out = self.model(one_hot, return_embeddings=True)
            if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                trunk = out[1]  # (1, num_bins, 3072)
            elif isinstance(out, torch.Tensor):
                trunk = out  # (1, num_bins, 3072)
            else:
                raise TypeError(f"Unexpected Enformer output type: {type(out)}")

        if trunk.dim() == 3 and trunk.shape[0] == 1:
            trunk = trunk.squeeze(0)  # (num_bins, 3072)
        else:
            raise RuntimeError(f"Unexpected trunk shape: {trunk.shape}")

        if pooling_strategy == "mean":
            pooled = trunk.mean(dim=0)  # (3072,)
        elif pooling_strategy == "median":
            pooled = trunk.median(dim=0).values
        else:
            pooled = trunk.max(dim=0).values  # (3072,)

        return pooled.cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Enformer embeddings for a batch of DNA sequences.

        1. Preprocess each string → one-hot (1, 5, 196608).
        2. Concatenate → batch tensor (B, 5, 196608).
        3. Run model(batch_tensor, return_embeddings=True) → (B, num_bins, 3072).
        4. Pool each sample over num_bins (mean or max) → (B, 3072).
        5. Return a list of NumPy arrays of length 3072, one per input.

        Parameters
        ----------
        inputs : Sequence[str]
            List of DNA sequence strings.
        pooling_strategy : str, default "mean"
            “mean” or “max” pooling over genomic bins.
        **kwargs : Any
            Currently unused.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays (length 3072), one for each input sequence.

        Raises
        ------
        RuntimeError
            If the model isn’t loaded or pooling strategy is invalid.
        TypeError
            If the model output format is unexpected.
        RuntimeError
            If batch pooling fails or output count mismatches input count.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Enformer model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling strategy:'{pooling_strategy}'")

        preproc_list = []
        for seq in inputs:
            preproc_list.append(self._preprocess_sequence(seq))  # each → (1, 5, 196608)

        batch_tensor = torch.cat(preproc_list, dim=0).to(self.device)  # (B, 5, 196608)
        with torch.no_grad():
            out = self.model(batch_tensor, return_embeddings=True)
            if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                emb = out[1]  # (B, num_bins, 3072)
            elif isinstance(out, torch.Tensor):
                emb = out  # (B, num_bins, 3072)
            else:
                raise TypeError(f"Unexpected Enformer output type: {type(out)}")

        if pooling_strategy == "mean":
            pooled = emb.mean(dim=1)  # (B, 3072)
        elif pooling_strategy == "median":
            pooled = emb.median(dim=1).values
        else:
            pooled = emb.max(dim=1).values  # (B, 3072)

        return [arr.cpu().numpy() for arr in pooled]


# ——————————————————————————————————————————————————————————————————————————
#                       BORZOI WRAPPER
# ——————————————————————————————————————————————————————————————————————————

try:
    from borzoi_pytorch import Borzoi
except ImportError:
    logging.warning("borzoi_pytorch not installed; BorzoiWrapper will be nonfunctional.")
    Borzoi = None  # type: ignore


class BorzoiWrapper(BaseModelWrapper):
    """
    Wrapper for the Borzoi model (via borzoi_pytorch).

    This class handles:
      1. Padding or center-cropping an arbitrary-length DNA string to exactly 524,288 base pairs.
      2. Converting the padded/truncated index tensor to a 4-channel one-hot tensor.
      3. Running `get_embs_after_crop` on the Borzoi model to obtain trunk embeddings.
      4. Pooling over the genomic bins dimension (mean or max) to yield a final embedding of size hidden_dim.

    Attributes
    ----------
    SEQUENCE_LENGTH : int
        Required fixed input length (524,288 bp) for Borzoi.
    NUM_CHANNELS : int
        Number of one-hot channels (4 for A/C/G/T).
    ALPHABET_MAP : dict[str, int]
        Mapping from nucleotide characters to indices (A=0, C=1, G=2, T=3). Others default to 0.
    model_type : Literal["dna"]
        Indicates that this wrapper expects DNA sequence inputs.
    available_pooling_strategies : list[str]
        Supported pooling strategies ("mean", "max").
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "median"]

    SEQUENCE_LENGTH = 524_288
    NUM_CHANNELS = 4
    ALPHABET_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(self, model_path_or_name: str = "johahi/borzoi-replicate-0", **kwargs):
        """
        Initialize the BorzoiWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Hugging Face model identifier or local path for Borzoi weights.
            Defaults to "johahi/borzoi-replicate-0".
        **kwargs : Any
            Reserved for future use (none currently).
        """
        super().__init__(model_path_or_name, **kwargs)

    def load(self, device: torch.device):
        """
        Load the Borzoi model onto the specified device.

        This method uses `borzoi_pytorch.Borzoi.from_pretrained` to instantiate the model,
        moves it to the given device, and sets it to evaluation mode. It also attempts to
        read `config.dim` for hidden dimension; defaults to 512 if missing.

        Parameters
        ----------
        device : torch.device
            The target device for model inference (e.g., torch.device("cuda") or torch.device("cpu")).

        Raises
        ------
        ImportError
            If `borzoi_pytorch` is not available.
        RuntimeError
            If the model fails to load for any other reason.
        """
        if self.model is not None:
            logging.warning(f"Borzoi '{self.model_name}' already loaded.")
            return
        if Borzoi is None:
            raise ImportError("borzoi_pytorch not installed; cannot load BorzoiWrapper.")

        logging.info(f"Loading Borzoi '{self.model_name}' …")
        try:
            self.model = Borzoi.from_pretrained(self.model_name).to(device).eval()
            self.device = device
            hidden_dim = getattr(self.model.config, "dim", None)
            if hidden_dim is None:
                hidden_dim = 512
                logging.warning("Could not detect Borzoi config.dim; defaulting to 512.")
            self.TRUNK_OUTPUT_DIM = hidden_dim
            logging.info(f"Borzoi '{self.model_name}' loaded on {device} (trunk_dim={hidden_dim}).")
        except Exception as e:
            logging.error(f"Failed to load Borzoi '{self.model_name}': {e}")
            self.model = None
            self.device = None
            raise RuntimeError(f"Could not load Borzoi '{self.model_name}'.") from e

    def _preprocess_sequence(self, sequence: str) -> torch.Tensor:
        """
        Internal method to preprocess a DNA sequence for Borzoi.

        Convert an arbitrary-length DNA string into a one-hot tensor of shape
        (1, NUM_CHANNELS, SEQUENCE_LENGTH), padding with zero-vectors if necessary.

        Steps:
        1. Uppercase the input string and map characters A/C/G/T → 0/1/2/3; others → 0.
        2. Build an index tensor of shape (1, L_in).
        3. One-hot encode → (1, L_in, NUM_CHANNELS), then permute → (1, NUM_CHANNELS, L_in).
        4. If L_in < SEQUENCE_LENGTH, pad on the last axis with [0,0,0,0] columns.
            If L_in > SEQUENCE_LENGTH, center-crop the last axis to exactly SEQUENCE_LENGTH.
        5. Return the resulting float tensor of shape (1, NUM_CHANNELS, SEQUENCE_LENGTH).

        Parameters
        ----------
        sequence : str
            Raw DNA sequence (e.g., "ACGT...").

        Returns
        -------
        torch.Tensor
            Float tensor of shape (1, 4, SEQUENCE_LENGTH), zero-padded or cropped.

        Raises
        ------
        ValueError
            If after padding/cropping the final length is not exactly SEQUENCE_LENGTH.
        """
        seq = sequence.upper()
        # 1) Map to integer indices
        idx = torch.tensor([self.ALPHABET_MAP.get(b, 0) for b in seq], dtype=torch.long).unsqueeze(0)  # (1, L_in)

        # 2) One-hot → (1, L_in, 4) then permute → (1, 4, L_in)
        oh = F.one_hot(idx, num_classes=self.NUM_CHANNELS).permute(0, 2, 1).float()  # (1, 4, L_in)

        L_in = oh.shape[2]
        L_tar = self.SEQUENCE_LENGTH

        # 3) Pad or crop on the last dimension
        if L_in < L_tar:
            pad_total = L_tar - L_in
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # pad with zero-vectors → value=0.0
            oh = F.pad(oh, (pad_left, pad_right), mode="constant", value=0.0)
            logging.debug(f"Padded Borzoi indices from {L_in}→{L_tar} with zeros.")
        elif L_in > L_tar:
            trim = (L_in - L_tar) // 2
            oh = oh[:, :, trim : trim + L_tar]
            logging.warning(f"Truncated Borzoi indices from {L_in}→{L_tar} (center‐crop).")
        # 4) Sanity check
        if oh.shape[2] != L_tar:
            raise ValueError(f"Preprocessing error: final length {oh.shape[2]} != {L_tar}")

        return oh  # shape: (1, 4, SEQUENCE_LENGTH)

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute Borzoi embeddings for a single DNA sequence.

        1. Preprocess the raw DNA string → one-hot (1, 4, 524288).
        2. Run `get_embs_after_crop(one_hot)` → trunk tensor of shape (1, hidden_dim, num_bins).
        3. Squeeze batch dim → (hidden_dim, num_bins), then pool over num_bins (mean or max).
        4. Return a NumPy array of shape (hidden_dim,).

        Parameters
        ----------
        input : str
            The DNA sequence string.
        pooling_strategy : str, default "mean"
            “mean” or “max” pooling over genomic bins.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length hidden_dim representing the pooled Borzoi embedding.

        Raises
        ------
        RuntimeError
            If the model hasn’t been loaded (`load()` not called) or pooling strategy is invalid.
        RuntimeError
            If `get_embs_after_crop` returns unexpected output.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Borzoi model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling: '{pooling_strategy}'")

        one_hot = self._preprocess_sequence(input).to(self.device)  # (1, 4, 524288)

        with torch.no_grad():
            logging.debug("Running Borzoi model forward pass…")
            embs = self.model.get_embs_after_crop(one_hot)
            if not isinstance(embs, torch.Tensor) or embs.dim() != 3:
                raise RuntimeError(f"Unexpected Borzoi output: {type(embs)}, shape={getattr(embs, 'shape', None)}")

        trunk = embs.squeeze(0)  # (hidden_dim, num_bins)
        if pooling_strategy == "mean":
            pooled = trunk.mean(dim=1)  # (hidden_dim,)
        else:
            pooled = trunk.max(dim=1).values  # (hidden_dim,)

        return pooled.to(torch.float32).cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Borzoi embeddings for a batch of arbitrary-length DNA sequences.

        1. Preprocess each string → one-hot (1, 4, 524288).
        2. Concatenate → batch tensor of shape (B, 4, 524288).
        3. Run `get_embs_after_crop(batch_tensor)` → trunk tensor of shape (B, hidden_dim, num_bins).
        4. Pool each sample over the bins dimension (mean or max) → (B, hidden_dim).
        5. Return a Python list of NumPy arrays, one per input, each of length hidden_dim.

        Parameters
        ----------
        inputs : Sequence[str]
            List of raw DNA sequence strings of arbitrary length.
        pooling_strategy : str, default="mean"
            “mean” or “max” pooling over the bins dimension.
        **kwargs : Any
            Currently unused (accepted for interface consistency).

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays (length=hidden_dim), one for each input sequence. If `inputs` is empty,
            returns an empty list.

        Raises
        ------
        RuntimeError
            If the Borzoi model has not been loaded (i.e., `load()` not called) or if `pooling_strategy` is invalid.
        TypeError
            If `get_embs_after_crop(...)` returns something other than a Tensor of shape (B, hidden_dim, num_bins).
        RuntimeError
            If pooling fails or if the number of output embeddings does not match the number of inputs.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Borzoi model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}.")

        # 1. Preprocess each DNA string → one-hot tensor of shape (1, 4, 524288)
        preproc_list: list[torch.Tensor] = []
        for seq in inputs:
            try:
                # _preprocess_sequence returns (1, 4, 524288)
                one_hot = self._preprocess_sequence(seq)
                preproc_list.append(one_hot)
            except Exception as e:
                logging.error(f"Failed to preprocess sequence: {seq[:50]}… Error: {e}")
                # You can choose to skip or re-raise; here we re-raise so the user is aware
                raise

        # 2. Concatenate into a single batch tensor of shape (B, 4, 524288)
        try:
            batch_tensor = torch.cat(preproc_list, dim=0).to(self.device)
        except Exception as e:
            logging.error(f"Failed to concatenate one-hot tensors into a batch: {e}")
            raise RuntimeError("Batch creation failed.") from e

        # 3. Run Borzoi’s forward up to cropping: get_embs_after_crop → Tensor (B, hidden_dim, num_bins)
        with torch.no_grad():
            emb_tensor = self.model.get_embs_after_crop(batch_tensor)
            if not isinstance(emb_tensor, torch.Tensor) or emb_tensor.dim() != 3:
                raise TypeError(
                    f"Unexpected Borzoi output from get_embs_after_crop: type={type(emb_tensor)}, "
                    f"shape={getattr(emb_tensor, 'shape', None)}"
                )

        # 4. Pool over the bins dimension (dim=2) according to pooling_strategy
        try:
            if pooling_strategy == "mean":
                # (B, hidden_dim, num_bins) → (B, hidden_dim)
                pooled = emb_tensor.mean(dim=2)
            else:  # “max”
                pooled = emb_tensor.max(dim=2).values
        except Exception as e:
            logging.error(f"Error during batch pooling: {e}")
            raise RuntimeError("Batch pooling failed.") from e

        # 5. Convert each row to a NumPy array and return as a list
        pooled = pooled.to(torch.float32).cpu()
        result_list = [pooled[i].numpy() for i in range(pooled.size(0))]

        if len(result_list) != len(inputs):
            logging.error(f"Mismatch in batch size: expected {len(inputs)} outputs, saw {len(result_list)}")
            raise RuntimeError("Output count does not match input count in embed_batch().")

        return result_list


# ——————————————————————————————————————————————————————————————————————————
#                       EVO2 WRAPPER
# ——————————————————————————————————————————————————————————————————————————


class Evo2Wrapper(BaseModelWrapper):
    """
    Wrapper for the Evo2 DNA language model.

    Evo2 is a state-of-the-art DNA language model for long-context modeling
    and design, supporting up to 1M base pair context at single-nucleotide
    resolution using the StripedHyena 2 architecture.

    This wrapper handles:
      1. Loading Evo2 checkpoints (7B, 40B, or smaller base models).
      2. Tokenizing DNA sequences via Evo2's built-in tokenizer.
      3. Extracting intermediate-layer embeddings (recommended over final layer).
      4. Pooling over the sequence length dimension.

    Attributes
    ----------
    LAYER_DEFAULTS : dict[str, str]
        Default embedding layers per model size, following the paper's
        recommendation that intermediate embeddings work better.
    model_type : Literal["dna"]
        Indicates that this wrapper expects DNA sequence inputs.
    available_pooling_strategies : list[str]
        Supported pooling strategies ("mean", "max", "cls").

    Notes
    -----
    Evo2 requires specific hardware: CUDA 12.1+, Compute Capability 8.9+ (Ada/Hopper),
    Transformer Engine >= 2.0, and Flash Attention. The 40B model requires multiple GPUs.

    Install with: ``pip install embpy[evo2]`` or ``pip install evo2``
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]

    LAYER_DEFAULTS: dict[str, str] = {
        "evo2_7b": "blocks.28.mlp.l3",
        "evo2_7b_base": "blocks.28.mlp.l3",
        "evo2_7b_262k": "blocks.28.mlp.l3",
        "evo2_7b_microviridae": "blocks.28.mlp.l3",
        "evo2_40b": "blocks.56.mlp.l3",
        "evo2_40b_base": "blocks.56.mlp.l3",
        "evo2_1b_base": "blocks.12.mlp.l3",
    }

    def __init__(
        self,
        model_path_or_name: str = "evo2_7b",
        layer_name: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the Evo2Wrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Evo2 checkpoint name. One of 'evo2_7b', 'evo2_40b', 'evo2_7b_base',
            'evo2_40b_base', 'evo2_1b_base', 'evo2_7b_262k', 'evo2_7b_microviridae'.
            Defaults to 'evo2_7b'.
        layer_name : str, optional
            Name of the layer from which to extract embeddings. If None, uses a
            recommended default for the given model (intermediate layer).
            See Evo2 paper for guidance on layer selection.
        **kwargs : Any
            Additional configuration passed to BaseModelWrapper.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.layer_name = layer_name
        self._evo2_model = None

    def load(self, device: torch.device) -> None:
        """
        Load the Evo2 model.

        Evo2 uses Vortex for inference and handles device placement internally,
        automatically splitting across available GPUs for the 40B model.
        The ``device`` argument is stored for interface consistency.

        Parameters
        ----------
        device : torch.device
            Target device. Evo2 manages its own device placement via Vortex,
            but this is stored for consistency with the BaseModelWrapper interface.

        Raises
        ------
        ImportError
            If the ``evo2`` package is not installed.
        RuntimeError
            If the model fails to load.
        """
        if self._evo2_model is not None:
            logging.warning(f"Evo2 '{self.model_name}' already loaded.")
            return
        if not _HAVE_EVO2 or Evo2 is None:
            raise ImportError(
                "evo2 package is not installed. Install it with: pip install embpy[evo2] or pip install evo2"
            )

        self.device = device

        if self.layer_name is None:
            self.layer_name = self.LAYER_DEFAULTS.get(self.model_name, "blocks.28.mlp.l3")
            logging.info(f"Using default embedding layer '{self.layer_name}' for model '{self.model_name}'.")

        logging.info(f"Loading Evo2 model '{self.model_name}'...")
        try:
            self._evo2_model = Evo2(self.model_name)
            self.model = self._evo2_model
            logging.info(f"Evo2 '{self.model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Evo2 '{self.model_name}': {e}")
            self._evo2_model = None
            self.model = None
            raise RuntimeError(f"Could not load Evo2 '{self.model_name}'.") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        layer_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute an embedding for a single DNA sequence using Evo2.

        Steps:
          1. Tokenize the sequence using Evo2's built-in tokenizer.
          2. Run the forward pass with ``return_embeddings=True``.
          3. Extract embeddings from the specified intermediate layer.
          4. Pool over the sequence length dimension.

        Parameters
        ----------
        input : str
            The DNA sequence string (e.g., "ACGTACGT...").
        pooling_strategy : str, default "mean"
            Pooling strategy: "mean", "max", or "cls" (first token).
        layer_name : str, optional
            Override the default embedding layer for this call.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array representing the pooled Evo2 embedding.

        Raises
        ------
        RuntimeError
            If the model hasn't been loaded.
        ValueError
            If the pooling strategy is invalid or no embeddings are returned.
        """
        if self._evo2_model is None:
            raise RuntimeError("Evo2 model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        target_layer = layer_name or self.layer_name

        target_device = self.device if self.device is not None else torch.device("cuda:0")
        input_ids = torch.tensor(
            self._evo2_model.tokenizer.tokenize(input),
            dtype=torch.int,
        ).unsqueeze(0).to(target_device)

        with torch.no_grad():
            _, embeddings = self._evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=[target_layer],
            )

        if target_layer not in embeddings:
            available_layers = list(embeddings.keys())
            raise ValueError(
                f"Layer '{target_layer}' not found in model output. Available: {available_layers}"
            )

        emb_tensor = embeddings[target_layer]

        if emb_tensor.dim() == 3 and emb_tensor.shape[0] == 1:
            emb_tensor = emb_tensor.squeeze(0)

        if pooling_strategy == "cls":
            pooled = emb_tensor[0]
        elif pooling_strategy == "max":
            pooled = emb_tensor.max(dim=0).values
        else:
            pooled = emb_tensor.mean(dim=0)

        return pooled.float().cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        layer_name: str | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Evo2 embeddings for a batch of DNA sequences.

        Processes each sequence individually since Evo2's tokenizer and
        inference pipeline are oriented toward single sequences.

        Parameters
        ----------
        inputs : Sequence[str]
            List of DNA sequence strings.
        pooling_strategy : str, default "mean"
            Pooling strategy to apply per sequence.
        layer_name : str, optional
            Override the default embedding layer for this call.
        **kwargs : Any
            Currently unused.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays, one per input sequence.
        """
        if self._evo2_model is None:
            raise RuntimeError("Evo2 model not loaded. Call load() first.")
        if not inputs:
            return []

        results: list[np.ndarray] = []
        for i, seq in enumerate(inputs):
            if i > 0 and i % 10 == 0:
                logging.info(f"Evo2 batch: processed {i}/{len(inputs)} sequences...")
            emb = self.embed(seq, pooling_strategy=pooling_strategy, layer_name=layer_name, **kwargs)
            results.append(emb)

        return results
