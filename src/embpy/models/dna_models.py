# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
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
    from evo2 import Evo2  # pyright: ignore[reportMissingImports]

    _HAVE_EVO2 = True
except ImportError:
    _HAVE_EVO2 = False
    Evo2 = None  # type: ignore

# ——————————————————————————————————————————————————————————————————————————
#                       EVO (v1 / v1.5) WRAPPER
# ——————————————————————————————————————————————————————————————————————————

try:
    from evo import Evo as EvoModel  # type: ignore[import-untyped]

    _HAVE_EVO = True
except ImportError:
    _HAVE_EVO = False
    EvoModel = None  # type: ignore

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

try:
    from transformers import AutoModel, AutoTokenizer

    _HAVE_TRANSFORMERS = True
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
except ImportError:
    _HAVE_TRANSFORMERS = False
    AutoModel = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForMaskedLM = None


from .base import BaseModelWrapper


def _hf_batched_embed(
    model: Any,
    tokenizer: Any,
    device: Any,
    inputs: Sequence[str],
    pooling_strategy: str,
    batch_size: int = 16,
    target_layer: int | None = None,
    tokenizer_kwargs: dict | None = None,
    cast_float: bool = False,
) -> list[np.ndarray]:
    """Shared batched inference for HuggingFace-tokenizer DNA models.

    Tokenizes inputs in chunks, runs one forward pass per chunk, and
    pools per-sequence.
    """
    if not inputs:
        return []

    tok_kw: dict[str, Any] = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": True,
    }
    if tokenizer_kwargs:
        tok_kw.update(tokenizer_kwargs)

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(inputs), batch_size):
        chunk = list(inputs[start : start + batch_size])

        enc = tokenizer(chunk, **tok_kw)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if hasattr(model, "config") else None,
                output_hidden_states=True,
            )
            if target_layer is not None:
                emb = out.hidden_states[target_layer]
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                emb = out.last_hidden_state
            else:
                emb = out.hidden_states[-1]

        for i in range(emb.shape[0]):
            seq_emb = emb[i]
            if attention_mask is not None:
                mask = attention_mask[i].unsqueeze(-1).float()
            else:
                mask = None

            if pooling_strategy == "cls":
                pooled = seq_emb[0]
            elif pooling_strategy == "last":
                pooled = seq_emb[-1]
            elif pooling_strategy == "max":
                if mask is not None:
                    seq_emb = seq_emb.masked_fill(mask == 0, float("-inf"))
                pooled = seq_emb.max(dim=0).values
            else:
                if mask is not None:
                    pooled = (seq_emb * mask).sum(0).div(mask.sum(0).clamp(min=1))
                else:
                    pooled = seq_emb.mean(dim=0)

            if cast_float:
                pooled = pooled.float()
            all_embeddings.append(pooled.cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_embeddings


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
            enformer_model: Any = from_pretrained(self.model_name, use_tf_gamma=self.use_tf_gamma)
            self.model = enformer_model.to(device).eval()
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
            borzoi_model: Any = Borzoi.from_pretrained(self.model_name)
            try:
                self.model = borzoi_model.to(device).eval()
            except NotImplementedError:
                self.model = borzoi_model.to_empty(device=device).eval()
            self.device = device
            hidden_dim = getattr(borzoi_model.config, "dim", None)
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
            model: Any = self.model
            embs = model.get_embs_after_crop(one_hot)
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
            batch_model: Any = self.model
            emb_tensor = batch_model.get_embs_after_crop(batch_tensor)
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
#                       EVO (v1 / v1.5) WRAPPER
# ——————————————————————————————————————————————————————————————————————————


class EvoWrapper(BaseModelWrapper):
    """
    Wrapper for the Evo (v1 / v1.5) DNA language model.

    Evo is a biological foundation model based on the StripedHyena architecture,
    a hybrid of attention and gated convolutions. It supports long-context
    modeling at single-nucleotide, byte-level resolution with near-linear
    scaling of compute and memory.

    This wrapper handles:
      1. Loading Evo checkpoints (v1 8k, v1 131k, v1.5 8k, and fine-tuned variants).
      2. Tokenizing DNA sequences via Evo's CharLevelTokenizer.
      3. Extracting embeddings from an intermediate StripedHyena block using
         a forward hook (Evo's forward pass returns logits, not embeddings).
      4. Pooling over the sequence length dimension.

    Attributes
    ----------
    AVAILABLE_MODELS : list[str]
        Supported Evo checkpoint names.
    model_type : Literal["dna"]
        Indicates that this wrapper expects DNA sequence inputs.
    available_pooling_strategies : list[str]
        Supported pooling strategies ("mean", "max", "cls").

    Notes
    -----
    Evo requires FlashAttention-2 (≤ 2.7.4) and a compatible GPU.
    Install with: ``pip install evo-model``

    See Also
    --------
    Evo2Wrapper : Wrapper for the successor model (Evo 2).
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]

    AVAILABLE_MODELS: list[str] = [
        "evo-1-8k-base",
        "evo-1-131k-base",
        "evo-1.5-8k-base",
        "evo-1-8k-crispr",
        "evo-1-8k-transposon",
    ]

    def __init__(
        self,
        model_path_or_name: str = "evo-1-8k-base",
        embedding_layer: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the EvoWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            Evo checkpoint name. One of ``'evo-1-8k-base'``, ``'evo-1-131k-base'``,
            ``'evo-1.5-8k-base'``, ``'evo-1-8k-crispr'``, ``'evo-1-8k-transposon'``.
            Defaults to ``'evo-1-8k-base'``.
        embedding_layer : int, optional
            Index of the StripedHyena block from which to extract hidden states.
            If None, defaults to the middle block (``num_blocks // 2``),
            which typically produces the best general-purpose representations.
        **kwargs : Any
            Additional configuration passed to BaseModelWrapper.
        """
        super().__init__(model_path_or_name, **kwargs)
        self.embedding_layer = embedding_layer
        self._evo_model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self, device: torch.device) -> None:
        """
        Load the Evo model onto the specified device.

        Internally uses the ``evo`` package's ``Evo`` class to download and
        instantiate the StripedHyena model and character-level tokenizer.

        Parameters
        ----------
        device : torch.device
            The target device for model inference.

        Raises
        ------
        ImportError
            If the ``evo-model`` package is not installed.
        RuntimeError
            If the model fails to load.
        """
        if self._evo_model is not None:
            logging.warning(f"Evo '{self.model_name}' already loaded.")
            return
        if not _HAVE_EVO or EvoModel is None:
            raise ImportError("evo-model package is not installed. Install with: pip install evo-model")

        self.device = device
        logging.info(f"Loading Evo model '{self.model_name}'...")

        try:
            evo_instance = EvoModel(self.model_name, device=str(device))
            sh_model: Any = evo_instance.model
            self._evo_model = sh_model
            self._tokenizer = evo_instance.tokenizer
            self.model = sh_model
            sh_model.eval()

            num_blocks = len(sh_model.blocks)
            if self.embedding_layer is None:
                self.embedding_layer = num_blocks // 2
            elif self.embedding_layer < 0 or self.embedding_layer >= num_blocks:
                raise ValueError(
                    f"embedding_layer={self.embedding_layer} is out of range "
                    f"for a model with {num_blocks} blocks (valid: 0–{num_blocks - 1})."
                )

            logging.info(
                f"Evo '{self.model_name}' loaded on {device} "
                f"({num_blocks} blocks, extracting embeddings from block {self.embedding_layer})."
            )
        except Exception as e:
            logging.error(f"Failed to load Evo '{self.model_name}': {e}")
            self._evo_model = None
            self._tokenizer = None
            self.model = None
            raise RuntimeError(f"Could not load Evo '{self.model_name}'.") from e

    def _extract_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract the hidden state from the configured block via a forward hook.

        The StripedHyena blocks return ``(output_tensor, inference_params)``
        tuples; only the output tensor is captured.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenised input of shape ``(1, seq_len)``.

        Returns
        -------
        torch.Tensor
            Hidden state tensor of shape ``(1, seq_len, hidden_size)``.
        """
        captured: dict[str, torch.Tensor] = {}

        evo_model: Any = self._evo_model
        target_block = evo_model.blocks[self.embedding_layer]

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                captured["hidden"] = output[0]
            else:
                captured["hidden"] = output

        handle = target_block.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                evo_model(input_ids)
        finally:
            handle.remove()

        return captured["hidden"]

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        embedding_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute an embedding for a single DNA sequence using Evo.

        Steps:
          1. Tokenize the sequence with Evo's CharLevelTokenizer.
          2. Run the forward pass and capture the hidden state from an
             intermediate StripedHyena block via a forward hook.
          3. Pool over the sequence length dimension.

        Parameters
        ----------
        input : str
            The DNA sequence string (e.g., ``"ACGTACGT..."``).
        pooling_strategy : str, default "mean"
            ``"mean"``, ``"max"``, or ``"cls"`` (first token).
        embedding_layer : int, optional
            Override the default embedding layer for this call only.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array representing the pooled Evo embedding.

        Raises
        ------
        RuntimeError
            If the model hasn't been loaded.
        ValueError
            If the pooling strategy is invalid.
        """
        if self._evo_model is None or self._tokenizer is None:
            raise RuntimeError("Evo model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. Available: {self.available_pooling_strategies}"
            )

        target_device = self.device if self.device is not None else torch.device("cuda:0")
        input_ids = (
            torch.tensor(
                self._tokenizer.tokenize(input),
                dtype=torch.int,
            )
            .unsqueeze(0)
            .to(target_device)
        )

        orig_layer = self.embedding_layer
        if embedding_layer is not None:
            self.embedding_layer = embedding_layer

        try:
            hidden = self._extract_hidden_state(input_ids)
        finally:
            self.embedding_layer = orig_layer

        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)  # (seq_len, hidden_dim)

        if pooling_strategy == "cls":
            pooled = hidden[0]
        elif pooling_strategy == "max":
            pooled = hidden.max(dim=0).values
        else:
            pooled = hidden.mean(dim=0)

        return pooled.float().cpu().numpy()

    def embed_from_layer(
        self,
        input: str,
        layer: int,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Extract an Evo embedding from a specific StripedHyena block.

        Overrides the base class to use Evo's ``embedding_layer`` parameter
        instead of the HF ``target_layer`` convention.

        Parameters
        ----------
        input : str
            DNA sequence string.
        layer : int
            StripedHyena block index to extract from.
        pooling_strategy : str
            Pooling strategy.
        **kwargs : Any
            Forwarded to :meth:`embed`.

        Returns
        -------
        np.ndarray
            Pooled 1D embedding from the specified layer.
        """
        return self.embed(input, pooling_strategy=pooling_strategy, embedding_layer=layer, **kwargs)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        embedding_layer: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Evo embeddings for a batch of DNA sequences.

        Processes each sequence individually since Evo's tokenizer and the
        StripedHyena inference pipeline are oriented toward single sequences.

        Parameters
        ----------
        inputs : Sequence[str]
            List of DNA sequence strings.
        pooling_strategy : str, default "mean"
            Pooling strategy to apply per sequence.
        embedding_layer : int, optional
            Override the default embedding layer for this call.
        **kwargs : Any
            Currently unused.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays, one per input sequence.
        """
        if self._evo_model is None:
            raise RuntimeError("Evo model not loaded. Call load() first.")
        if not inputs:
            return []

        results: list[np.ndarray] = []
        for i, seq in enumerate(inputs):
            if i > 0 and i % 10 == 0:
                logging.info(f"Evo batch: processed {i}/{len(inputs)} sequences...")
            emb = self.embed(
                seq,
                pooling_strategy=pooling_strategy,
                embedding_layer=embedding_layer,
                **kwargs,
            )
            results.append(emb)
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        return results


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
            name = self.model_name or "evo2_7b"
            self.layer_name = self.LAYER_DEFAULTS.get(name, "blocks.28.mlp.l3")
            logging.info(f"Using default embedding layer '{self.layer_name}' for model '{name}'.")

        logging.info(f"Loading Evo2 model '{self.model_name}'...")
        try:
            self._evo2_model = Evo2(self.model_name)
            self.model = self._evo2_model
            if hasattr(self._evo2_model, "eval"):
                self._evo2_model.eval()
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
        input_ids = (
            torch.tensor(
                self._evo2_model.tokenizer.tokenize(input),
                dtype=torch.int,
            )
            .unsqueeze(0)
            .to(target_device)
        )

        with torch.no_grad():
            _, embeddings = self._evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=[target_layer],
            )

        if target_layer not in embeddings:
            available_layers = list(embeddings.keys())
            raise ValueError(f"Layer '{target_layer}' not found in model output. Available: {available_layers}")

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

    def embed_from_layer(
        self,
        input: str,
        layer: int | str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Extract an Evo2 embedding from a specific layer.

        Overrides the base class to accept either an integer block index
        (converted to the Evo2 layer-name convention ``blocks.<i>.mlp.l3``)
        or an explicit layer name string.

        Parameters
        ----------
        input : str
            DNA sequence string.
        layer : int or str
            If ``int``, converted to ``"blocks.<layer>.mlp.l3"``.
            If ``str``, used directly as the Evo2 layer name.
        pooling_strategy : str
            Pooling strategy.
        **kwargs : Any
            Forwarded to :meth:`embed`.

        Returns
        -------
        np.ndarray
            Pooled 1D embedding from the specified layer.
        """
        if isinstance(layer, int):
            layer_name = f"blocks.{layer}.mlp.l3"
        else:
            layer_name = layer
        return self.embed(input, pooling_strategy=pooling_strategy, layer_name=layer_name, **kwargs)

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
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        return results


class GENALMWrapper(BaseModelWrapper):
    """Wrapper for GENA-LM BERT-style DNA language models.

    GENA-LM models are masked-language-model transformers trained on the
    human T2T genome assembly with BPE tokenization.  They accept sequences
    up to ~4,500 bp (BERT-base) or ~36,000 bp (BigBird-base).

    Available model identifiers (``AIRI-Institute/<name>``):

    * ``gena-lm-bert-base-t2t``          - 110 M params, 4.5 kb context
    * ``gena-lm-bert-large-t2t``         - 336 M params, 4.5 kb context
    * ``gena-lm-bert-base-lastln-t2t``   - 110 M params, 4.5 kb context
    * ``gena-lm-bert-base-t2t-multi``    - 110 M params, multi-species
    * ``gena-lm-bigbird-base-t2t``       - 110 M params, 36 kb context
    * ``gena-lm-bigbird-base-sparse-t2t``- 110 M params, 36 kb (requires DeepSpeed)

    Install:  ``pip install transformers``

    Parameters
    ----------
    model_path_or_name
        Full HuggingFace identifier, e.g.
        ``"AIRI-Institute/gena-lm-bert-base-t2t"``.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self,
        model_path_or_name: str = "AIRI-Institute/gena-lm-bert-base-t2t",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: Any = None

    def load(self, device: torch.device) -> None:
        if self.model is not None:
            logging.warning(f"GENA-LM '{self.model_name}' already loaded.")
            return
        if not _HAVE_TRANSFORMERS:
            raise ImportError("transformers package required: pip install transformers")

        logging.info(f"Loading GENA-LM '{self.model_name}' …")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(device).eval()
            self.device = device
            logging.info(f"GENA-LM '{self.model_name}' loaded on {device}.")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Could not load GENA-LM '{self.model_name}'.") from e

    def _tokenize(self, sequence: str) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("GENA-LM not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        enc = self._tokenize(input)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if target_layer is not None:
                emb = out.hidden_states[target_layer]
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                emb = out.last_hidden_state
            else:
                emb = out.hidden_states[-1]  # (1, L, H)

        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # (L, H)

        if pooling_strategy == "cls":
            return emb[0].cpu().numpy()
        elif pooling_strategy == "max":
            return emb.max(dim=0).values.cpu().numpy()
        else:
            if attention_mask is not None:
                mask = attention_mask.squeeze(0).unsqueeze(-1).float()
                return (emb * mask).sum(0).div(mask.sum(0).clamp(min=1)).cpu().numpy()
            return emb.mean(dim=0).cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("GENA-LM not loaded. Call load() first.")
        return _hf_batched_embed(
            self.model, self.tokenizer, self.device, inputs,
            pooling_strategy, batch_size=batch_size,
            target_layer=kwargs.get("target_layer"),
        )


class NucleotideTransformerWrapper(BaseModelWrapper):
    """Wrapper for Nucleotide Transformer v1/v2 models (InstaDeep / NVIDIA / TUM).

    NT models are large transformer encoders pre-trained on human and multi-species
    DNA using 6-mer tokenization (NT-v1) or BPE-like tokenization with RoPE (NT-v2).

    Available model identifiers (``InstaDeepAI/<name>``):

    NT-v1 (6-mer, 6 kb context):
    * ``nucleotide-transformer-500m-human-ref``
    * ``nucleotide-transformer-500m-1000g``
    * ``nucleotide-transformer-2.5b-1000g``
    * ``nucleotide-transformer-2.5b-multi-species``

    NT-v2 (RoPE, 12 kb context):
    * ``nucleotide-transformer-v2-50m-multi-species``
    * ``nucleotide-transformer-v2-100m-multi-species``
    * ``nucleotide-transformer-v2-250m-multi-species``
    * ``nucleotide-transformer-v2-500m-multi-species``

    Install:  ``pip install transformers``

    Notes
    -----
    NT models use trust_remote_code because the tokenizer/model config is
    hosted on the HuggingFace Hub, not shipped with transformers.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self,
        model_path_or_name: str = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: Any = None

    def load(self, device: torch.device) -> None:
        if self.model is not None:
            logging.warning(f"NT '{self.model_name}' already loaded.")
            return
        if not _HAVE_TRANSFORMERS:
            raise ImportError("transformers package required: pip install transformers")

        logging.info(f"Loading Nucleotide Transformer '{self.model_name}' …")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            from transformers import AutoConfig, AutoModelForMaskedLM  # type: ignore

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = (
                AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                )
                .to(device)
                .eval()
            )
            self.device = device
            logging.info(f"NT '{self.model_name}' loaded via AutoModelForMaskedLM on {device}.")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Could not load NT '{self.model_name}'.") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("NT model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        enc = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if target_layer is not None:
                emb = out.hidden_states[target_layer]
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                emb = out.last_hidden_state
            else:
                emb = out.hidden_states[-1]  # (1, L, H)

        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        if pooling_strategy == "cls":
            return emb[0].cpu().numpy()
        elif pooling_strategy == "max":
            return emb.max(dim=0).values.cpu().numpy()
        else:
            if attention_mask is not None:
                mask = attention_mask.squeeze(0).unsqueeze(-1).float()
                return (emb * mask).sum(0).div(mask.sum(0).clamp(min=1)).cpu().numpy()
            return emb.mean(dim=0).cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("NT model not loaded. Call load() first.")
        return _hf_batched_embed(
            self.model, self.tokenizer, self.device, inputs,
            pooling_strategy, batch_size=batch_size,
            target_layer=kwargs.get("target_layer"),
        )


class NucleotideTransformerV3Wrapper(BaseModelWrapper):
    """Wrapper for Nucleotide Transformer v3 (NTv3) by InstaDeep.

    NTv3 is a U-Net–style genomic foundation model with single-base tokenisation
    supporting sequences up to 1 Mb.  It is pre-trained on ~9T bp from OpenGenome2
    across >128k species and post-trained on ~16k functional tracks.

    Unlike earlier JAX-based NTv3, the HuggingFace release uses standard PyTorch
    and loads via ``AutoModelForMaskedLM``.

    Available model identifiers (``InstaDeepAI/<n>``):

    * ``NTv3_8M_pre``   -   8 M params, pre-trained only
    * ``NTv3_100M_pre`` - 100 M params, pre-trained
    * ``NTv3_100M_pos`` - 100 M params, post-trained (tracks + annotation)
    * ``NTv3_650M_pre`` - 650 M params, pre-trained
    * ``NTv3_650M_pos`` - 650 M params, post-trained (best accuracy)

    Install:  ``pip install transformers``

    Notes
    -----
    * Input sequence length **must be a multiple of 128** (U-Net downsampling).
      Sequences are padded with ``N`` automatically.
    * Embeddings are extracted from ``output.hidden_states[-1]`` (final encoder
      layer before the MLM head).  Use ``target_layer`` to extract intermediate
      representations.
    * ``add_special_tokens=False`` and ``pad_to_multiple_of=128`` are set
      automatically to match the model's requirements.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]
    _PAD_MULTIPLE = 128

    def __init__(
        self,
        model_path_or_name: str = "InstaDeepAI/NTv3_100M_pre",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: Any = None

    def load(self, device: torch.device) -> None:
        if self.model is not None:
            logging.warning(f"NTv3 '{self.model_name}' already loaded.")
            return
        if not _HAVE_TRANSFORMERS:
            raise ImportError("transformers package required: pip install transformers")

        logging.info(f"Loading NTv3 '{self.model_name}' (PyTorch) …")
        try:
            from transformers import AutoModelForMaskedLM  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True).to(device).eval()
            self.device = device
            logging.info(f"NTv3 '{self.model_name}' loaded on {device}.")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Could not load NTv3 '{self.model_name}'.") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("NTv3 not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        enc = self.tokenizer(
            input.upper(),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=self._PAD_MULTIPLE,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if target_layer is not None:
                emb = out.hidden_states[target_layer]
            else:
                # hidden_states[-1] is the final encoder state before MLM head
                emb = out.hidden_states[-1]  # (1, L, H)

        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # (L, H)

        if pooling_strategy == "cls":
            return emb[0].float().cpu().numpy()
        elif pooling_strategy == "max":
            return emb.max(dim=0).values.float().cpu().numpy()
        else:
            if attention_mask is not None:
                mask = attention_mask.squeeze(0).unsqueeze(-1).float()
                return (emb * mask).sum(0).div(mask.sum(0).clamp(min=1)).float().cpu().numpy()
            return emb.mean(dim=0).float().cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("NTv3 not loaded. Call load() first.")
        uppercased = [s.upper() for s in inputs]
        return _hf_batched_embed(
            self.model, self.tokenizer, self.device, uppercased,
            pooling_strategy, batch_size=batch_size,
            target_layer=kwargs.get("target_layer"),
            tokenizer_kwargs={"add_special_tokens": False, "pad_to_multiple_of": self._PAD_MULTIPLE},
            cast_float=True,
        )


class HyenaDNAWrapper(BaseModelWrapper):
    """Wrapper for HyenaDNA long-range genomic foundation models (HazyResearch).

    HyenaDNA is a causal language model based on the Hyena (implicit long
    convolution) operator, supporting context lengths up to 1 M tokens at
    single-nucleotide resolution.  Weights are hosted on the HuggingFace Hub
    under ``LongSafari/hyenadna-*-seqlen-hf``.

    Available identifiers:

    * ``LongSafari/hyenadna-tiny-1k-seqlen-hf``
    * ``LongSafari/hyenadna-tiny-1k-d256-seqlen-hf``
    * ``LongSafari/hyenadna-tiny-16k-d128-seqlen-hf``
    * ``LongSafari/hyenadna-small-32k-seqlen-hf``
    * ``LongSafari/hyenadna-medium-160k-seqlen-hf``
    * ``LongSafari/hyenadna-medium-450k-seqlen-hf``
    * ``LongSafari/hyenadna-large-1m-seqlen-hf``

    Install:  ``pip install transformers``

    Notes
    -----
    Embeddings are extracted from the hidden states of the causal LM backbone
    (``output_hidden_states=True``).  By default the last hidden state is used.
    Set ``target_layer`` to extract intermediate representations.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls", "last"]

    _CHAR_TO_ID: dict[str, int] = {
        "A": 7,
        "C": 8,
        "G": 9,
        "T": 10,
        "N": 11,
        "a": 7,
        "c": 8,
        "g": 9,
        "t": 10,
        "n": 11,
    }
    _DEFAULT_TOKEN_ID = 11  # N

    def __init__(
        self,
        model_path_or_name: str = "LongSafari/hyenadna-small-32k-seqlen-hf",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: Any = None

    def load(self, device: torch.device) -> None:
        if self.model is not None:
            logging.warning(f"HyenaDNA '{self.model_name}' already loaded.")
            return
        if not _HAVE_TRANSFORMERS or AutoModelForCausalLM is None:
            raise ImportError("transformers package required: pip install transformers")

        logging.info(f"Loading HyenaDNA '{self.model_name}' …")
        try:
            # HyenaDNA uses character-level tokenisation; the AutoTokenizer from
            # the HF repo handles this via trust_remote_code
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(device).eval()
            self.device = device
            logging.info(f"HyenaDNA '{self.model_name}' loaded on {device}.")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Could not load HyenaDNA '{self.model_name}'.") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("HyenaDNA not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        enc = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
            hs = out.hidden_states
            if target_layer is not None:
                emb = hs[target_layer]
            else:
                emb = hs[-1]  # last hidden state

        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        if pooling_strategy == "cls":
            return emb[0].float().cpu().numpy()
        elif pooling_strategy == "last":
            return emb[-1].float().cpu().numpy()
        elif pooling_strategy == "max":
            return emb.max(dim=0).values.float().cpu().numpy()
        else:
            return emb.mean(dim=0).float().cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("HyenaDNA not loaded. Call load() first.")
        return _hf_batched_embed(
            self.model, self.tokenizer, self.device, inputs,
            pooling_strategy, batch_size=batch_size,
            target_layer=kwargs.get("target_layer"),
            cast_float=True,
        )


class CaduceusWrapper(BaseModelWrapper):
    """Wrapper for Caduceus bi-directional equivariant DNA language models.

    Caduceus extends the Mamba/SSM architecture with reverse-complement (RC)
    equivariance for long-range DNA modelling (up to 131 k bp context).

    Available model identifiers:

    * ``kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16``
      - RC-*augmented* (PhiH) variant; standard MLM pre-training.
    * ``kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16``
      - RC-*equivariant* (PhiS) variant; no RC augmentation needed.

    Install:  ``pip install transformers``

    Notes
    -----
    * Caduceus is an MLM model.  We extract hidden states from the backbone
      rather than the MLM head.
    * For the PS variant, embedding dimensions may be doubled (RC complement
      is modelled jointly); mean-pooling collapses this correctly.
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "cls"]

    def __init__(
        self,
        model_path_or_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name, **kwargs)
        self.tokenizer: Any = None

    def load(self, device: torch.device) -> None:
        if self.model is not None:
            logging.warning(f"Caduceus '{self.model_name}' already loaded.")
            return
        if not _HAVE_TRANSFORMERS:
            raise ImportError("transformers package required: pip install transformers")

        logging.info(f"Loading Caduceus '{self.model_name}' …")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            try:
                m = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                m = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = m.to(device).eval()
            self.device = device
            logging.info(f"Caduceus '{self.model_name}' loaded on {device}.")
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Could not load Caduceus '{self.model_name}'.") from e

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        target_layer: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("Caduceus not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'")

        enc = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            hs = out.hidden_states
            if target_layer is not None:
                emb = hs[target_layer]
            else:
                emb = hs[-1]

        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        if pooling_strategy == "cls":
            return emb[0].float().cpu().numpy()
        elif pooling_strategy == "max":
            return emb.max(dim=0).values.float().cpu().numpy()
        else:
            if attention_mask is not None:
                mask = attention_mask.squeeze(0).unsqueeze(-1).float()
                return (emb * mask).sum(0).div(mask.sum(0).clamp(min=1)).float().cpu().numpy()
            return emb.mean(dim=0).float().cpu().numpy()

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if self.model is None or self.device is None or self.tokenizer is None:
            raise RuntimeError("Caduceus not loaded. Call load() first.")
        return _hf_batched_embed(
            self.model, self.tokenizer, self.device, inputs,
            pooling_strategy, batch_size=batch_size,
            target_layer=kwargs.get("target_layer"),
            cast_float=True,
        )
