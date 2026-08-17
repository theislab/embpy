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


def _safe_model_forward(model: Any, **kwargs: Any) -> Any:
    """Forward pass that drops kwargs the model can't accept.

    HuggingFace models declare an explicit signature for ``forward(self, ...)``
    but the catalog mixes them with custom causal LMs (notably HyenaDNA's
    ``HyenaDNAForCausalLM``) that reject ``attention_mask`` outright. Rather
    than maintain a hand-curated allowlist, we attempt the call, catch the
    ``TypeError("got an unexpected keyword argument")`` once, cache the
    rejected kwarg name on the model instance, and retry. Subsequent calls
    pay zero retry overhead because we filter the kwargs up-front using the
    cached rejection set.
    """
    rejected: set[str] = getattr(model, "_embpy_rejected_kwargs", set())
    call_kwargs = {k: v for k, v in kwargs.items() if v is None or k not in rejected}
    # Always drop None values for kwargs that are in the rejected set --
    # passing ``attention_mask=None`` to HyenaDNA still triggers TypeError.
    call_kwargs = {k: v for k, v in call_kwargs.items() if k not in rejected}
    try:
        return model(**call_kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" not in msg:
            raise
        changed = False
        for k in list(call_kwargs.keys()):
            if k in msg:
                rejected.add(k)
                call_kwargs.pop(k, None)
                changed = True
        if not changed:
            raise
        # Cache the rejection set on the model so future calls skip them.
        try:
            model._embpy_rejected_kwargs = rejected
        except AttributeError:
            pass
        return model(**call_kwargs)


def _resolve_context_window(model: Any, tokenizer: Any) -> int:
    """Return the smallest context window the model+tokenizer can handle.

    DNA LMs differ wildly in their context capacity (GENA-LM-BERT: 512
    BPE tokens / ~4.5 kb; GENA-BigBird: 4096 / ~36 kb; NT-v2: 12k;
    HyenaDNA: 1M; Caduceus: 131k). Both pieces of information can be
    independently capped:

    * ``model.config.max_position_embeddings`` is the HARD limit -- the
      positional-embedding buffer has that many rows and a longer input
      crashes with a tensor-expansion RuntimeError.
    * ``tokenizer.model_max_length`` is the SOFT limit -- many HF
      tokenizers set this to ``1e30`` (an "unlimited" sentinel), which
      means ``truncation=True`` silently does nothing.

    We take the min of the two whenever they expose a real (< 1e6)
    value, and fall back to 512 (BERT-standard) only if neither does.
    """
    candidates: list[int] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        m = getattr(cfg, "max_position_embeddings", None)
        if isinstance(m, int) and 0 < m < 1_000_000:
            candidates.append(int(m))
    if tokenizer is not None:
        t = getattr(tokenizer, "model_max_length", None)
        if isinstance(t, int) and 0 < t < 1_000_000:
            candidates.append(int(t))
    return min(candidates) if candidates else 512


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
    """Chunked batched inference for HuggingFace-tokenizer DNA models.

    Each input is tokenized without truncation, then split into chunks
    no larger than the model's context window (``max_position_embeddings``).
    Every chunk is embedded once; per-input results are mean-pooled across
    chunks so each input always produces a single output vector regardless
    of sequence length. Special tokens (CLS / SEP / BOS / EOS) are re-added
    to each chunk so the model sees a self-contained span every time.

    Why chunking instead of plain truncation:
        Several DNA LMs in our catalog (GENA-LM BERT 4.5 kb context,
        GENA-LM BigBird 36 kb, Nucleotide Transformer ~12 kb) have
        context windows much smaller than a full gene locus
        (150 kb+). Plain truncation would silently throw away
        99% of the sequence and produce a near-degenerate embedding.
        Chunk-and-mean keeps full coverage at the cost of more
        forward passes for long inputs.

    Pooling semantics:
        ``mean / max / cls / last`` are applied PER CHUNK first; the
        resulting one-vector-per-chunk embeddings are then mean-pooled
        across chunks per input.

        ``none`` is not compatible with chunked inference (no canonical
        way to stitch per-token embeddings across overlapping spans).
        For short inputs that fit in one chunk we still honour it; for
        long inputs we raise NotImplementedError so the caller picks
        another strategy.
    """
    if not inputs:
        return []

    ctx = _resolve_context_window(model, tokenizer)

    # The encoding of an input includes special tokens (CLS, SEP, ...)
    # added by the tokenizer. We have to account for those when deciding
    # how many "content" tokens fit in one chunk -- otherwise we still
    # overflow ctx by 1-2 tokens.
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    head_id = cls_id if cls_id is not None else bos_id
    tail_id = sep_id if sep_id is not None else eos_id
    head_cost = 1 if head_id is not None else 0
    tail_cost = 1 if tail_id is not None else 0
    inner_max = ctx - head_cost - tail_cost
    if inner_max <= 0:
        raise RuntimeError(
            f"Context window {ctx} too small after reserving room for "
            f"special tokens ({head_cost} head + {tail_cost} tail)."
        )

    # --- Step 1: tokenize each input WITHOUT truncation, then chunk. -------
    # We carry chunk_to_input_idx so we can group chunks back per input
    # after the forward pass.
    chunk_ids: list[list[int]] = []
    chunk_attn: list[list[int]] = []
    chunk_to_input_idx: list[int] = []
    overflow_warned = False
    for i, text in enumerate(inputs):
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
            **(tokenizer_kwargs or {}),
        )
        ids = enc["input_ids"]
        # Some tokenizers return list[list[int]] for batch encoding;
        # here we passed a single string so we expect list[int]. Coerce.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        n_chunks = max(1, (len(ids) + inner_max - 1) // inner_max)
        if n_chunks > 1 and not overflow_warned:
            logging.info(
                "Sequence longer than context window (%d > %d) -- "
                "chunking into %d pieces and mean-pooling per input.",
                len(ids), ctx, n_chunks,
            )
            overflow_warned = True
        for j in range(0, max(len(ids), 1), inner_max):
            piece = ids[j : j + inner_max]
            full = (
                ([head_id] if head_id is not None else [])
                + piece
                + ([tail_id] if tail_id is not None else [])
            )
            chunk_ids.append(full)
            chunk_attn.append([1] * len(full))
            chunk_to_input_idx.append(i)

    # --- Step 2: embed chunks in `batch_size`-sized minibatches. -----------
    chunk_pooled: list[np.ndarray | None] = [None] * len(chunk_ids)
    needs_sequence_out = pooling_strategy == "none"
    if needs_sequence_out and any(
        sum(1 for src in chunk_to_input_idx if src == i) > 1
        for i in range(len(inputs))
    ):
        raise NotImplementedError(
            "pooling_strategy='none' is not supported when inputs exceed "
            "the model context window (would produce variable-length "
            "outputs across inputs)."
        )

    for start in range(0, len(chunk_ids), batch_size):
        end = min(start + batch_size, len(chunk_ids))
        batch_ids = chunk_ids[start:end]
        batch_attn = chunk_attn[start:end]
        max_len = max(len(x) for x in batch_ids)
        input_ids = torch.tensor(
            [x + [pad_id] * (max_len - len(x)) for x in batch_ids],
            dtype=torch.long, device=device,
        )
        attention_mask = torch.tensor(
            [x + [0] * (max_len - len(x)) for x in batch_attn],
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            out = _safe_model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if target_layer is not None:
                emb = out.hidden_states[target_layer]
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                emb = out.last_hidden_state
            else:
                emb = out.hidden_states[-1]

        for k in range(emb.shape[0]):
            seq_emb = emb[k]
            mask = attention_mask[k].unsqueeze(-1).float()
            if pooling_strategy == "none":
                if cast_float:
                    seq_emb = seq_emb.float()
                chunk_pooled[start + k] = seq_emb.cpu().numpy()
                continue
            elif pooling_strategy == "cls":
                pooled = seq_emb[0]
            elif pooling_strategy == "last":
                pooled = seq_emb[-1]
            elif pooling_strategy == "max":
                pooled = seq_emb.masked_fill(mask == 0, float("-inf")).max(dim=0).values
            else:
                pooled = (seq_emb * mask).sum(0).div(mask.sum(0).clamp(min=1))
            if cast_float:
                pooled = pooled.float()
            chunk_pooled[start + k] = pooled.cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 3: group chunks per input, mean-pool across chunks. ----------
    # Build an index from input -> list of its chunk indices first to
    # avoid an O(n_inputs * n_chunks) scan.
    chunks_per_input: list[list[int]] = [[] for _ in range(len(inputs))]
    for c, src in enumerate(chunk_to_input_idx):
        chunks_per_input[src].append(c)

    all_embeddings: list[np.ndarray] = []
    for i in range(len(inputs)):
        idxs = chunks_per_input[i]
        if not idxs:
            raise RuntimeError(f"No chunks produced for input {i}.")
        if len(idxs) == 1 or pooling_strategy == "none":
            # Single chunk: return its pooled (or per-token) embedding as-is.
            all_embeddings.append(chunk_pooled[idxs[0]])  # type: ignore[arg-type]
            continue
        stacked = np.stack([chunk_pooled[c] for c in idxs], axis=0)  # type: ignore[misc]
        all_embeddings.append(stacked.mean(axis=0).astype(stacked.dtype))

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
    available_pooling_strategies = ["mean", "max", "median", "none"]

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

        if pooling_strategy == "none":
            return trunk.cpu().numpy()
        elif pooling_strategy == "mean":
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
        batch_size: int = 4,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Enformer embeddings for a batch of DNA sequences.

        Processes inputs in mini-batches of ``batch_size`` (default 4). The
        previous implementation concatenated ALL inputs into a single giant
        tensor of shape ``(B, 5, 196608)`` and pushed it through the model
        in one forward pass; that crashes for any realistic perturbation
        catalog because Enformer's dilated conv trunk maintains an
        L=196608, ~1500-channel activation throughout most layers --
        roughly ``B * 196608 * 1500 * 4 bytes`` of activations per layer,
        which blows past 80 GB GPU memory once ``B`` exceeds a handful of
        sequences. The observed failure was a 1335 GiB allocation attempt
        for B=2393.

        Parameters
        ----------
        inputs : Sequence[str]
            List of DNA sequence strings.
        pooling_strategy : str, default "mean"
            "mean" / "max" / "median" pooling over genomic bins.
        batch_size : int, default 4
            Number of sequences to run through the model per forward pass.
            Default 4 is the largest we've verified to fit comfortably on
            an 80 GB GPU; tune up/down if you have more/less memory.
        **kwargs : Any
            Currently unused.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays (length 3072), one for each input sequence.

        Raises
        ------
        RuntimeError
            If the model is not loaded or pooling strategy is invalid.
        TypeError
            If the model output format is unexpected.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Enformer model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling strategy:'{pooling_strategy}'")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        results: list[np.ndarray] = []
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            preproc_list = [self._preprocess_sequence(seq) for seq in chunk]
            batch_tensor = torch.cat(preproc_list, dim=0).to(self.device)  # (b, 5, 196608)

            with torch.no_grad():
                out = self.model(batch_tensor, return_embeddings=True)
                if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                    emb = out[1]  # (b, num_bins, 3072)
                elif isinstance(out, torch.Tensor):
                    emb = out
                else:
                    raise TypeError(f"Unexpected Enformer output type: {type(out)}")

            if pooling_strategy == "mean":
                pooled = emb.mean(dim=1)
            elif pooling_strategy == "median":
                pooled = emb.median(dim=1).values
            else:
                pooled = emb.max(dim=1).values

            results.extend(arr.cpu().numpy() for arr in pooled)

            # Release the activation graph before the next mini-batch; this
            # is the difference between steady-state ~10 GB and runaway
            # allocation on multi-batch runs.
            del batch_tensor, emb, pooled, preproc_list, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


# ——————————————————————————————————————————————————————————————————————————
#                       BORZOI WRAPPER
# ——————————————————————————————————————————————————————————————————————————

try:
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.pytorch_borzoi_model import TRACKS_DF as _BORZOI_TRACKS_DF
    from borzoi_pytorch.pytorch_borzoi_utils import undo_squashed_scale as _undo_squashed_scale
except ImportError:
    logging.warning("borzoi_pytorch not installed; BorzoiWrapper will be nonfunctional.")
    Borzoi = None  # type: ignore
    _BORZOI_TRACKS_DF = None  # type: ignore
    _undo_squashed_scale = None  # type: ignore


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
        Mapping from nucleotide characters to indices (A=0, C=1, G=2, T=3).
    UNKNOWN_INDEX : int
        Sentinel index for ``N`` and any other non-ACGT character. One-hot
        encoding uses ``NUM_CHANNELS + 1`` classes and then drops this channel,
        so an ambiguous base becomes an all-zero column -- the same encoding the
        padding path produces, and Baskerville's default for ``N``.
    model_type : Literal["dna"]
        Indicates that this wrapper expects DNA sequence inputs.
    available_pooling_strategies : list[str]
        Supported pooling strategies ("mean", "max").
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "median", "none"]

    SEQUENCE_LENGTH = 524_288
    NUM_CHANNELS = 4
    ALPHABET_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
    UNKNOWN_INDEX = 4
    # Borzoi predicts coverage tracks at 32 bp resolution (see borzoi_pytorch's
    # `Borzoi.crop`, which crops the trunk to `target_length` bins before the
    # final head convs).
    BIN_SIZE = 32

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
        1. Uppercase the input string and map characters A/C/G/T → 0/1/2/3;
           every other character (``N``, IUPAC ambiguity codes, soft-masked
           bases) → ``UNKNOWN_INDEX``.
        2. Build an index tensor of shape (1, L_in).
        3. One-hot encode over ``NUM_CHANNELS + 1`` classes and drop the sentinel
           channel → (1, L_in, NUM_CHANNELS), then permute → (1, NUM_CHANNELS, L_in).
           Ambiguous bases therefore become all-zero columns, identical to the
           padding representation, rather than being read as adenine.
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
        # 1) Map to integer indices. Anything that is not A/C/G/T -- N, IUPAC
        # ambiguity codes, soft-masked residues -- maps to UNKNOWN_INDEX so it
        # becomes an all-zero column below. Defaulting these to 0 would silently
        # read them as adenine, which fabricates sequence content the caller
        # never supplied.
        idx = torch.tensor(
            [self.ALPHABET_MAP.get(b, self.UNKNOWN_INDEX) for b in seq],
            dtype=torch.long,
        ).unsqueeze(0)  # (1, L_in)

        n_unknown = int((idx == self.UNKNOWN_INDEX).sum())
        if n_unknown:
            logging.warning(
                "Borzoi input contains %d non-ACGT character(s) (%.2f%% of %d); "
                "encoding them as all-zero columns.",
                n_unknown,
                100.0 * n_unknown / idx.shape[1],
                idx.shape[1],
            )

        # 2) One-hot over NUM_CHANNELS + 1 classes, then drop the sentinel
        #    channel so UNKNOWN_INDEX yields [0, 0, 0, 0]. This mirrors
        #    enformer-pytorch's seq_indices_to_one_hot and matches the zero
        #    columns used for padding below.
        #    → (1, L_in, 4) then permute → (1, 4, L_in)
        oh = (
            F.one_hot(idx, num_classes=self.NUM_CHANNELS + 1)[..., : self.NUM_CHANNELS]
            .permute(0, 2, 1)
            .float()
        )  # (1, 4, L_in)

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
        return_profile: bool = False,
        is_human: bool = True,
        track_indices: Sequence[int] | None = None,
        undo_squashed_scale: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        return_profile : bool, default False
            If True, also run the model's prediction head (see
            :meth:`predict_profile`) and return ``(embedding, profile)``
            instead of just the embedding. This costs a second forward pass.
        is_human, track_indices, undo_squashed_scale
            Forwarded to :meth:`predict_profile` when ``return_profile=True``;
            ignored otherwise.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length hidden_dim representing the pooled
            Borzoi embedding, or, if ``return_profile=True``, a tuple
            ``(embedding, profile)`` where ``profile`` is the array returned
            by :meth:`predict_profile`.

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
        if pooling_strategy == "none":
            pooled_np = trunk.T.cpu().numpy()  # (num_bins, hidden_dim)
        elif pooling_strategy == "mean":
            pooled_np = trunk.mean(dim=1).to(torch.float32).cpu().numpy()  # (hidden_dim,)
        else:
            pooled_np = trunk.max(dim=1).values.to(torch.float32).cpu().numpy()

        if not return_profile:
            return pooled_np

        profile = self.predict_profile(
            input,
            is_human=is_human,
            track_indices=track_indices,
            undo_squashed_scale=undo_squashed_scale,
        )
        return pooled_np, profile

    @property
    def profile_offset_bp(self) -> int:
        """bp offset of profile bin 0 relative to the start of the model's input window.

        Borzoi crops both ends of its receptive field before the prediction
        head, so the first predicted bin does not start at position 0 of the
        524,288 bp input. This is only exact when the sequence passed to
        :meth:`predict_profile` is exactly ``SEQUENCE_LENGTH`` long (i.e. no
        additional padding/cropping was applied by :meth:`_preprocess_sequence`).
        """
        if self.model is None:
            raise RuntimeError("Borzoi model not loaded. Call load() first.")
        crop_length = self.model.crop.target_length
        return (self.SEQUENCE_LENGTH - crop_length * self.BIN_SIZE) // 2

    def predict_profile(
        self,
        input: str,
        is_human: bool = True,
        track_indices: Sequence[int] | None = None,
        undo_squashed_scale: bool = False,
    ) -> np.ndarray:
        """Predict Borzoi's full coverage profile (RNA-seq/ATAC/ChIP tracks).

        Unlike :meth:`embed`, which pools the pre-head trunk embedding, this
        runs the model's human/mouse head (``model.forward``) to obtain the
        actual predicted per-track, per-bin coverage.

        Parameters
        ----------
        input : str
            Raw DNA sequence string.
        is_human : bool, default True
            Use the human head (7,611 tracks) or the mouse head (2,608 tracks).
        track_indices : sequence of int, optional
            Restrict the output to these track indices (saves memory if only
            a handful of assays are of interest). Default: all tracks.
        undo_squashed_scale : bool, default False
            Borzoi targets are stored on a "squashed" (soft-clipped,
            power-transformed) scale during training; set True to invert
            that transform and recover approximate linear-scale coverage,
            which is required before summing bins for variant-effect scoring.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_tracks, num_bins)``. With the default
            settings ``num_bins == model.crop.target_length`` and each bin
            spans :attr:`BIN_SIZE` (32) bp; see :attr:`profile_offset_bp` for
            how bin 0 maps back to genomic coordinates.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Borzoi model not loaded. Call load() first.")

        one_hot = self._preprocess_sequence(input).to(self.device)  # (1, 4, 524288)

        with torch.no_grad():
            model: Any = self.model
            tracks = model.forward(one_hot, is_human=is_human)

        if not isinstance(tracks, torch.Tensor) or tracks.dim() != 3:
            raise RuntimeError(
                f"Unexpected Borzoi profile output: {type(tracks)}, "
                f"shape={getattr(tracks, 'shape', None)}"
            )

        tracks = tracks.squeeze(0)  # (num_tracks, num_bins)
        if track_indices is not None:
            idx = torch.as_tensor(list(track_indices), dtype=torch.long, device=tracks.device)
            tracks = tracks.index_select(0, idx)
        if undo_squashed_scale:
            if _undo_squashed_scale is None:
                raise ImportError("borzoi_pytorch not installed; cannot undo squashed scale.")
            tracks = _undo_squashed_scale(tracks.unsqueeze(0)).squeeze(0)

        return tracks.to(torch.float32).cpu().numpy()

    @staticmethod
    def get_track_metadata() -> Any:
        """Return Borzoi's bundled track metadata (one row per output channel).

        Columns include ``identifier``, ``description``, ``file``,
        ``strand_pair``, ``sum_stat`` and ``scale`` -- the same
        ``targets.txt`` table used internally by ``borzoi_pytorch`` to decode
        and unsquash predictions. Useful for mapping :meth:`predict_profile`
        track indices to assay names (e.g. "RNA:liver" or "ATAC:PBMC").

        Returns
        -------
        pandas.DataFrame
            A copy of the bundled track annotation table.
        """
        if _BORZOI_TRACKS_DF is None:
            raise ImportError("borzoi_pytorch not installed; cannot load track metadata.")
        return _BORZOI_TRACKS_DF.copy()

    # Conservative default chosen for a 80 GB H100 / A100.
    # Borzoi's first conv expands a (B, 4, 524288) input into roughly
    # (B, 512, 524288) fp32, which is ~1 GiB per sample even before the
    # rest of the trunk is materialised. With B = 4 the activation grid
    # stays under ~5 GiB; combined with the loaded weights (~6 GiB) and
    # autograd-free intermediates we sit safely below 16 GiB peak.
    # Smaller cards (e.g. V100 32 GB) should set this to 2 via the
    # micro_batch_size kwarg. Callers that previously relied on the
    # implicit "one giant forward" path will now see chunked forwards
    # but identical output ordering and values.
    DEFAULT_MICRO_BATCH_SIZE = 4

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        micro_batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Borzoi embeddings for a batch of arbitrary-length DNA sequences.

        Borzoi has a fixed 524288 bp receptive field, and the first
        convolutional block expands that to a roughly 1 GB activation
        tensor *per sample* (fp32). Concatenating the entire caller-side
        batch into one forward easily exceeds 80 GB of HBM (we have seen
        2057 sequences requesting a 1993 GiB allocation). This method
        therefore chunks the input list into ``micro_batch_size`` slices
        and runs one forward per slice, returning the per-sample
        embeddings in the original input order.

        1. For each micro-batch of size ``micro_batch_size``:
           a. Preprocess each string → one-hot (1, 4, 524288).
           b. Concatenate → tensor of shape (b, 4, 524288).
           c. Run ``get_embs_after_crop`` → (b, hidden_dim, num_bins).
           d. Pool over bins → (b, hidden_dim) and stash on CPU.
        2. Concatenate the CPU-side chunks into a single (B, hidden_dim).
        3. Return a list of 1D NumPy arrays, one per input.

        Parameters
        ----------
        inputs : Sequence[str]
            List of raw DNA sequence strings of arbitrary length.
        pooling_strategy : str, default="mean"
            "mean" or "max" pooling over the bins dimension.
        micro_batch_size : int or None, optional
            Number of sequences to push through ``get_embs_after_crop``
            in one GPU forward. Defaults to
            ``BorzoiWrapper.DEFAULT_MICRO_BATCH_SIZE`` (4), which is
            tuned for 80 GB cards. Set to 2 on V100 32 GB, or higher on
            an H200 if you have head-room. Values <= 0 fall back to the
            class default.
        **kwargs : Any
            Reserved for interface consistency; ignored.

        Returns
        -------
        list[np.ndarray]
            A list of 1D NumPy arrays (length=hidden_dim), one for each
            input sequence. Empty list iff ``inputs`` is empty.

        Raises
        ------
        RuntimeError
            If the model is not loaded or output sizing is inconsistent.
        ValueError
            If ``pooling_strategy`` is not in the supported set.
        TypeError
            If ``get_embs_after_crop`` returns an unexpected payload.
        """
        del kwargs
        if self.model is None or self.device is None:
            raise RuntimeError("Borzoi model not loaded. Call load() first.")
        if not inputs:
            return []
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling '{pooling_strategy}'. Choose from {self.available_pooling_strategies}.")
        if pooling_strategy == "none":
            # The per-bin layout requires us to keep the full
            # (hidden_dim, num_bins) tensor per sample, which is what the
            # batched path was historically incompatible with anyway. We
            # explicitly reject it here rather than emit a misleading
            # mean over the bins axis.
            raise ValueError(
                "Borzoi.embed_batch does not support pooling_strategy='none'. "
                "Call embed(seq, pooling_strategy='none') per sequence instead."
            )

        mb = (
            int(micro_batch_size)
            if (micro_batch_size is not None and int(micro_batch_size) > 0)
            else self.DEFAULT_MICRO_BATCH_SIZE
        )

        batch_model: Any = self.model
        total = len(inputs)
        pooled_chunks: list[torch.Tensor] = []

        for start in range(0, total, mb):
            chunk = inputs[start : start + mb]

            # 1. Preprocess this chunk only. We avoid materialising all
            #    2000+ (1, 4, 524288) tensors on CPU at once -- that is
            #    ~16 GB of RAM for the Replogle workload and is wasteful
            #    when only `mb` of them are live on GPU at a time.
            try:
                preproc_chunk = [self._preprocess_sequence(seq) for seq in chunk]
            except Exception as e:
                # Surface the offending sequence index for fast triage.
                # _preprocess_sequence already logs the offending head.
                logging.error(
                    f"Preprocess failed inside micro-batch starting at index {start}: {e}"
                )
                raise

            try:
                batch_tensor = torch.cat(preproc_chunk, dim=0).to(self.device)
            except Exception as e:
                logging.error(f"Failed to concatenate one-hot tensors into a batch: {e}")
                raise RuntimeError("Batch creation failed.") from e

            with torch.no_grad():
                emb_tensor = batch_model.get_embs_after_crop(batch_tensor)
                if not isinstance(emb_tensor, torch.Tensor) or emb_tensor.dim() != 3:
                    raise TypeError(
                        f"Unexpected Borzoi output from get_embs_after_crop: type={type(emb_tensor)}, "
                        f"shape={getattr(emb_tensor, 'shape', None)}"
                    )

                if pooling_strategy == "mean":
                    pooled = emb_tensor.mean(dim=2)
                elif pooling_strategy == "max":
                    pooled = emb_tensor.max(dim=2).values
                else:  # "median"
                    pooled = emb_tensor.median(dim=2).values

            # Move pooled chunk to CPU immediately and drop the GPU
            # activation; this keeps the steady-state HBM footprint
            # bounded to one micro-batch worth of trunk activations.
            pooled_chunks.append(pooled.to(torch.float32).cpu())
            del emb_tensor, batch_tensor, preproc_chunk
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            logging.debug(
                "Borzoi micro-batch %d..%d / %d done.",
                start, start + len(chunk), total,
            )

        pooled_all = torch.cat(pooled_chunks, dim=0)
        result_list = [pooled_all[i].numpy() for i in range(pooled_all.size(0))]

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
    available_pooling_strategies = ["mean", "max", "cls", "none"]

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

        if pooling_strategy == "none":
            return hidden.float().cpu().numpy()
        elif pooling_strategy == "cls":
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
    available_pooling_strategies = ["mean", "max", "cls", "none"]

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

        if pooling_strategy == "none":
            return emb_tensor.float().cpu().numpy()
        elif pooling_strategy == "cls":
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
    available_pooling_strategies = ["mean", "max", "cls", "none"]

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

        # Delegate to the shared chunk-and-pool helper so long DNA loci
        # are handled identically here and in embed_batch. GENA-LM-BERT's
        # 512-token positional buffer would otherwise crash on multi-kb
        # gene regions; see _hf_batched_embed for the chunking strategy.
        embeddings = _hf_batched_embed(
            self.model, self.tokenizer, self.device, [input],
            pooling_strategy, batch_size=1,
            target_layer=target_layer,
        )
        return embeddings[0]

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
    available_pooling_strategies = ["mean", "max", "cls", "none"]

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

        if pooling_strategy == "none":
            return emb.cpu().numpy()
        elif pooling_strategy == "cls":
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
    available_pooling_strategies = ["mean", "max", "cls", "none"]
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

        if pooling_strategy == "none":
            return emb.float().cpu().numpy()
        elif pooling_strategy == "cls":
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
    available_pooling_strategies = ["mean", "max", "cls", "last", "none"]
    # HyenaDNA replaces attention with an implicit long convolution (the Hyena
    # operator), so there are no attention weights to extract.
    has_attention = False

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

        if pooling_strategy == "none":
            return emb.float().cpu().numpy()
        elif pooling_strategy == "cls":
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
    available_pooling_strategies = ["mean", "max", "cls", "none"]
    # Caduceus is built on the Mamba/SSM architecture (bi-directional, RC-equivariant)
    # and has no attention mechanism, so there are no attention weights to extract.
    has_attention = False

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

        if pooling_strategy == "none":
            return emb.float().cpu().numpy()
        elif pooling_strategy == "cls":
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
