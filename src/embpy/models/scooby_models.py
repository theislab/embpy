import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import BaseModelWrapper


try:
    from scooby.modeling import Scooby
    from scooby.utils.utils import undo_squashed_scale as _scooby_undo_squashed_scale
except ImportError:
    logging.warning("scooby not installed; ScoobyWrapper will be nonfunctional.")
    Scooby = None  # type: ignore
    _scooby_undo_squashed_scale = None  # type: ignore


class ScoobyWrapper(BaseModelWrapper):
    """
    Wrapper for the Scooby model (via the `scooby` package, gagneurlab/scooby).

    Scooby predicts single-cell-resolution scRNA-seq/scATAC-seq coverage
    profiles from DNA sequence, conditioned on a precomputed single-cell
    embedding (e.g. a scPoli/scVI latent vector for the cell(s) of
    interest). This class handles:

      1. Padding/center-cropping an arbitrary-length DNA string to exactly
         524,288 bp (identical convention to :class:`BorzoiWrapper`).
      2. Converting one or more precomputed cell embeddings into the
         per-cell decoder conv weights via `Scooby.forward_cell_embs_only`.
      3. Running the cell-conditioned decoder to obtain per-track,
         per-bin coverage predictions, optionally pseudobulk-aggregated
         across cells of the same cell type (summed on the linear/unsquashed
         scale, matching `scooby.utils.utils.get_pseudobulk_profile_pred`).
    """

    model_type = "dna"
    available_pooling_strategies = ["mean", "max", "median", "none"]

    SEQUENCE_LENGTH = 524_288
    NUM_CHANNELS = 4
    ALPHABET_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
    # Same trunk/backbone as Borzoi -> identical bin resolution.
    BIN_SIZE = 32

    # Verified from each checkpoint's safetensors header (see module
    # docstring above), not from scooby's generic scripts/config_*.yaml
    # (whose defaults are examples for the multiome bone-marrow dataset and
    # do not match the actual released checkpoints for OneK1K/Epicardioids).
    KNOWN_CHECKPOINTS: dict[str, dict[str, Any]] = {
        "lauradmartens/onek1k-scooby": dict(
            cell_emb_dim=10,
            n_tracks=2,
            use_transform_borzoi_emb=True,
            clip_soft=5.0,
        ),
        "johahi/neurips-scooby": dict(
            cell_emb_dim=14,
            n_tracks=3,
            use_transform_borzoi_emb=True,
            clip_soft=5.0,
        ),
        "lauradmartens/epicardioids-scooby": dict(
            cell_emb_dim=50,
            n_tracks=3,
            use_transform_borzoi_emb=True,
            clip_soft=5.0,
        ),
    }

    def __init__(
        self,
        model_path_or_name: str = "lauradmartens/onek1k-scooby",
        cell_emb_dim: int | None = None,
        n_tracks: int | None = None,
        embedding_dim: int = 1920,
        use_transform_borzoi_emb: bool | None = None,
        clip_soft: float | None = None,
        **kwargs,
    ):
        """
        Initialize the ScoobyWrapper.

        Parameters
        ----------
        model_path_or_name : str, optional
            HuggingFace model identifier or local path for Scooby weights.
            Defaults to "lauradmartens/onek1k-scooby" (the OneK1K/Yazar et
            al. 2022 PBMC checkpoint -- the same cohort used elsewhere in
            this pipeline for third-cohort eQTL replication). Also accepts
            "johahi/neurips-scooby" and "lauradmartens/epicardioids-scooby",
            or any other Scooby checkpoint (local or on the Hub) provided
            `cell_emb_dim`/`n_tracks` are given explicitly.
        cell_emb_dim, n_tracks, use_transform_borzoi_emb, clip_soft
            Checkpoint architecture/postprocessing hyperparameters. Any left
            as ``None`` are looked up in :attr:`KNOWN_CHECKPOINTS` by
            `model_path_or_name`; if not found there, they must be supplied
            explicitly (this keeps the wrapper usable for future
            fine-tuned checkpoints, e.g. a Cardinal G&H/UKB-specific Scooby).
        **kwargs : Any
            Reserved for future use (none currently).

        Raises
        ------
        ValueError
            If a required hyperparameter cannot be resolved from
            :attr:`KNOWN_CHECKPOINTS` and was not passed explicitly.
        """
        super().__init__(model_path_or_name, **kwargs)
        known = self.KNOWN_CHECKPOINTS.get(model_path_or_name, {})

        def _resolve(name: str, value: Any, default: Any = None) -> Any:
            if value is not None:
                return value
            if name in known:
                return known[name]
            if default is not None:
                return default
            raise ValueError(
                f"'{name}' could not be inferred for checkpoint '{model_path_or_name}' "
                f"(not in ScoobyWrapper.KNOWN_CHECKPOINTS). Pass it explicitly to "
                "ScoobyWrapper(...)."
            )

        self.cell_emb_dim: int = _resolve("cell_emb_dim", cell_emb_dim)
        self.n_tracks: int = _resolve("n_tracks", n_tracks)
        self.use_transform_borzoi_emb: bool = _resolve(
            "use_transform_borzoi_emb", use_transform_borzoi_emb, default=True
        )
        self.clip_soft: float = _resolve("clip_soft", clip_soft, default=5.0)
        self.embedding_dim = embedding_dim

    @property
    def TRACK_NAMES(self) -> list[str]:
        """Track identifiers aligned with the model's track axis.

        Derived from `scooby.utils.utils.get_outputs`, which selects RNA
        tracks via the boolean mask ``[1, 1, 0]`` and ATAC via ``[0, 0, 1]``
        over each (RNA+, RNA-, ATAC) track triplet -- i.e. for
        ``n_tracks == 3`` (multiome checkpoints) the track order is
        ``["RNA:+", "RNA:-", "ATAC"]``; for ``n_tracks == 2`` (RNA-only
        checkpoints, e.g. OneK1K) it is ``["RNA:+", "RNA:-"]``.
        """
        if self.n_tracks == 2:
            return ["RNA:+", "RNA:-"]
        if self.n_tracks == 3:
            return ["RNA:+", "RNA:-", "ATAC"]
        return [f"track_{i}" for i in range(self.n_tracks)]

    def load(self, device: torch.device):
        """
        Load the Scooby model onto the specified device.

        Uses `scooby.modeling.Scooby.from_pretrained`, which (like Borzoi)
        is a HuggingFace `transformers.PreTrainedModel`: it downloads
        `config.json` (Borzoi backbone hyperparameters only) plus
        `model.safetensors`, and forwards `cell_emb_dim`/`embedding_dim`/
        `n_tracks`/`use_transform_borzoi_emb` to `Scooby.__init__` to build
        the cell-state decoder before loading weights.

        Parameters
        ----------
        device : torch.device
            The target device for model inference.

        Raises
        ------
        ImportError
            If `scooby` is not installed.
        RuntimeError
            If the model fails to load for any other reason.
        """
        if self.model is not None:
            logging.warning(f"Scooby '{self.model_name}' already loaded.")
            return
        if Scooby is None:
            raise ImportError("scooby not installed; cannot load ScoobyWrapper.")

        logging.info(f"Loading Scooby '{self.model_name}' …")
        try:
            scooby_model: Any = Scooby.from_pretrained(
                self.model_name,
                cell_emb_dim=self.cell_emb_dim,
                embedding_dim=self.embedding_dim,
                n_tracks=self.n_tracks,
                return_center_bins_only=True,
                disable_cache=False,
                use_transform_borzoi_emb=self.use_transform_borzoi_emb,
            )

            n_meta = 0
            for mod in scooby_model.modules():
                pos = getattr(mod, "positions", None)
                if isinstance(pos, torch.Tensor) and pos.is_meta:
                    n_rel = getattr(mod, "num_rel_pos_features", None)
                    if n_rel is None:
                        raise RuntimeError(
                            "Scooby attention module has a meta 'positions' buffer but no "
                            "num_rel_pos_features to rebuild it from."
                        )
                    from borzoi_pytorch.pytorch_borzoi_transformer import (
                        get_positional_embed,
                    )
                    mod.positions = get_positional_embed(4096, n_rel, torch.device("cpu"))
                    n_meta += 1
            if n_meta:
                logging.info(
                    "Rematerialised %d meta positional-encoding buffer(s) before moving "
                    "Scooby to %s.", n_meta, device
                )
            self.model = scooby_model.to(device).eval()
            self.device = device
            logging.info(
                f"Scooby '{self.model_name}' loaded on {device} "
                f"(cell_emb_dim={self.cell_emb_dim}, n_tracks={self.n_tracks})."
            )
        except Exception as e:
            logging.error(f"Failed to load Scooby '{self.model_name}': {e}")
            self.model = None
            self.device = None
            raise RuntimeError(f"Could not load Scooby '{self.model_name}'.") from e

    def _preprocess_sequence(self, sequence: str) -> torch.Tensor:
        """
        Internal method to preprocess a DNA sequence for Scooby.

        Identical convention to `BorzoiWrapper._preprocess_sequence`,
        including the channel-first output layout: (1, 4, SEQUENCE_LENGTH).

        Note: `Scooby.forward_seq_to_emb`'s docstring claims a channel-last
        `(batch_size, seq_len, 4)` input, but that is stale/inaccurate --
        its first op, `self.conv_dna(x)`, is Borzoi's `nn.Conv1d(4, 512,
        15)`, which requires channel-first `(batch, 4, seq_len)`. Confirmed
        against scooby's own callers: `scripts/train_rna_only.py` does
        `inputs = inputs.permute(0, 2, 1)` (dataset yields channel-last,
        permuted to channel-first) before `scooby(inputs, ...)`, and
        `docs/notebooks/Evaluate_Model.ipynb` does
        `seqs = x[0].cuda().permute(0,2,1)` for the same reason.

        Parameters
        ----------
        sequence : str
            Raw DNA sequence (e.g., "ACGT...").

        Returns
        -------
        torch.Tensor
            Float tensor of shape (1, 4, SEQUENCE_LENGTH).
        """
        seq = sequence.upper()
        idx = torch.tensor([self.ALPHABET_MAP.get(b, 0) for b in seq], dtype=torch.long).unsqueeze(0)  # (1, L_in)
        oh = F.one_hot(idx, num_classes=self.NUM_CHANNELS).permute(0, 2, 1).float()  # (1, 4, L_in)

        L_in = oh.shape[2]
        L_tar = self.SEQUENCE_LENGTH

        if L_in < L_tar:
            pad_total = L_tar - L_in
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            oh = F.pad(oh, (pad_left, pad_right), mode="constant", value=0.0)
            logging.debug(f"Padded Scooby indices from {L_in}→{L_tar} with zeros.")
        elif L_in > L_tar:
            trim = (L_in - L_tar) // 2
            oh = oh[:, :, trim : trim + L_tar]
            logging.warning(f"Truncated Scooby indices from {L_in}→{L_tar} (center‐crop).")

        if oh.shape[2] != L_tar:
            raise ValueError(f"Preprocessing error: final length {oh.shape[2]} != {L_tar}")

        return oh  # shape: (1, 4, SEQUENCE_LENGTH)

    def _prepare_cell_embeddings(self, cell_embeddings: Any) -> torch.Tensor:
        """Coerce a caller-supplied cell embedding (or embeddings) into
        the (1, num_cells, cell_emb_dim) tensor `forward_cell_embs_only` expects.

        Parameters
        ----------
        cell_embeddings : array-like
            Either a single embedding vector of shape ``(cell_emb_dim,)``
            (e.g. a scPoli/scVI latent vector for one cell, or a
            precomputed centroid representing a cell type), or a matrix of
            shape ``(num_cells, cell_emb_dim)`` (one row per single cell,
            e.g. all cells belonging to one annotated cell type -- see
            :meth:`predict_profile` with ``aggregate="pseudobulk"``).

        Returns
        -------
        torch.Tensor
            Float tensor of shape (1, num_cells, cell_emb_dim).
        """
        arr = torch.as_tensor(np.asarray(cell_embeddings), dtype=torch.float32)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)  # (1, cell_emb_dim)
        if arr.dim() != 2 or arr.shape[-1] != self.cell_emb_dim:
            raise ValueError(
                f"cell_embeddings must have shape (cell_emb_dim,) or (num_cells, cell_emb_dim) "
                f"with cell_emb_dim={self.cell_emb_dim}; got shape {tuple(arr.shape)}."
            )
        return arr.unsqueeze(0)  # (1, num_cells, cell_emb_dim)

    @property
    def profile_offset_bp(self) -> int:
        """bp offset of profile bin 0 relative to the start of the model's
        input window (identical mechanism to `BorzoiWrapper.profile_offset_bp`
        since Scooby shares Borzoi's trunk/crop)."""
        if self.model is None:
            raise RuntimeError("Scooby model not loaded. Call load() first.")
        crop_length = self.model.crop.target_length
        return (self.SEQUENCE_LENGTH - crop_length * self.BIN_SIZE) // 2

    def predict_profile(
        self,
        input: str,
        cell_embeddings: Any,
        aggregate: Literal["pseudobulk", "none"] = "pseudobulk",
        undo_squashed_scale: bool = True,
        track_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Predict Scooby's cell-conditioned coverage profile.

        Parameters
        ----------
        input : str
            Raw DNA sequence string.
        cell_embeddings : array-like
            A single precomputed cell embedding of shape ``(cell_emb_dim,)``
            (one cell, or a prototype/centroid representing a cell type),
            or ``(num_cells, cell_emb_dim)`` for multiple single cells
            (e.g. all cells of one annotated cell type). Must live in the
            same embedding space the loaded checkpoint was trained to
            condition on (for the default OneK1K checkpoint: a 10-D scPoli
            latent space fit on that cohort's highly-variable genes --
            projecting new cells into this space requires the trained
            scPoli reference model, not an independent PCA/scVI fit; see
            module notes).
        aggregate : {"pseudobulk", "none"}, default "pseudobulk"
            Only relevant when multiple cells are passed. ``"pseudobulk"``
            sums the per-cell *unsquashed* predicted coverage across cells
            (matching `scooby.utils.utils.get_pseudobulk_profile_pred`),
            returning a single ``(num_tracks, num_bins)`` array -- directly
            compatible with `embpy.tl.genomics.snp_utils.profile_variant_effect_score`/
            `SNPEmbedder.predict_variant_effect`, exactly like `BorzoiWrapper.predict_profile`.
            ``"none"`` returns per-cell profiles, shape
            ``(num_cells, num_tracks, num_bins)``.
        undo_squashed_scale : bool, default True
            Invert Scooby's soft-clipped, power-transformed training scale
            to recover approximate linear-scale coverage. Defaults to
            ``True`` here (unlike `BorzoiWrapper.predict_profile`, which
            defaults to `False`) because pseudobulk aggregation is only
            valid on the linear scale -- scooby's own evaluation code
            (`get_pseudobulk_profile_pred`) always unsquashes before summing.
        track_indices : sequence of int, optional
            Restrict the output to these track indices (see
            :attr:`TRACK_NAMES`).

        Returns
        -------
        np.ndarray
            ``(num_tracks, num_bins)`` if a single cell embedding was given
            or ``aggregate="pseudobulk"``; ``(num_cells, num_tracks, num_bins)``
            if multiple cells were given and ``aggregate="none"``. Each bin
            spans :attr:`BIN_SIZE` (32) bp; see :attr:`profile_offset_bp`.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Scooby model not loaded. Call load() first.")

        one_hot = self._preprocess_sequence(input).to(self.device)  # (1, 4, SEQUENCE_LENGTH)
        cell_emb = self._prepare_cell_embeddings(cell_embeddings).to(self.device)
        num_cells = cell_emb.shape[1]

        with torch.no_grad():
            model: Any = self.model
            conv_weights, conv_biases = model.forward_cell_embs_only(cell_emb)
            raw = model.forward_sequence_w_convs(one_hot, conv_weights, conv_biases)  # (1, num_bins, num_cells * n_tracks)

        if not isinstance(raw, torch.Tensor) or raw.dim() != 3:
            raise RuntimeError(f"Unexpected Scooby output: {type(raw)}, shape={getattr(raw, 'shape', None)}")

        num_bins = raw.shape[1]
        per_cell = raw.squeeze(0).view(num_bins, num_cells, self.n_tracks).permute(1, 2, 0)  # (num_cells, n_tracks, num_bins)

        if undo_squashed_scale:
            if _scooby_undo_squashed_scale is None:
                raise ImportError("scooby not installed; cannot undo squashed scale.")
            per_cell = _scooby_undo_squashed_scale(per_cell, clip_soft=self.clip_soft)

        if track_indices is not None:
            idx = torch.as_tensor(list(track_indices), dtype=torch.long, device=per_cell.device)
            per_cell = per_cell.index_select(1, idx)

        if num_cells == 1:
            return per_cell.squeeze(0).to(torch.float32).cpu().numpy()  # (n_tracks, num_bins)

        if aggregate == "pseudobulk":
            pooled = per_cell.sum(dim=0)  # (n_tracks, num_bins) -- sum, not mean; see get_pseudobulk_profile_pred
            return pooled.to(torch.float32).cpu().numpy()
        if aggregate == "none":
            return per_cell.to(torch.float32).cpu().numpy()  # (num_cells, n_tracks, num_bins)
        raise ValueError(f"Invalid aggregate='{aggregate}'. Choose 'pseudobulk' or 'none'.")

    def get_track_metadata(self) -> pd.DataFrame:
        """Return a minimal track-annotation table (one row per output channel).

        Mirrors `BorzoiWrapper.get_track_metadata`'s ``identifier`` column
        (used by `SNPEmbedder.predict_variant_effect`) but, since Scooby's
        tracks are cell-embedding-conditioned RNA(+ATAC) strand channels
        rather than fixed named bulk assays, only ``identifier`` is
        populated (see :attr:`TRACK_NAMES`).
        """
        return pd.DataFrame({"identifier": self.TRACK_NAMES})

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the cell-embedding-independent trunk embedding for a DNA sequence.

        This pools `Scooby.forward_seq_to_emb`'s output (the same
        fine-tuned Borzoi trunk representation the cell-state decoder is
        applied to) -- i.e. it does *not* depend on `cell_embeddings`,
        analogous to `BorzoiWrapper.embed`.

        Parameters
        ----------
        input : str
            The DNA sequence string.
        pooling_strategy : str, default "mean"
            "mean" or "max" pooling over genomic bins.
        **kwargs : Any
            Currently unused but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of length `embedding_dim`.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Scooby model not loaded. Call load() first.")
        if pooling_strategy not in self.available_pooling_strategies:
            raise ValueError(f"Invalid pooling: '{pooling_strategy}'")

        one_hot = self._preprocess_sequence(input).to(self.device)  # (1, 4, SEQUENCE_LENGTH)

        with torch.no_grad():
            model: Any = self.model
            trunk = model.forward_seq_to_emb(one_hot)  # (1, embedding_dim, num_bins)
            if not isinstance(trunk, torch.Tensor) or trunk.dim() != 3:
                raise RuntimeError(f"Unexpected Scooby trunk output: {type(trunk)}, shape={getattr(trunk, 'shape', None)}")

        trunk = trunk.squeeze(0)  # (embedding_dim, num_bins)
        if pooling_strategy == "none":
            pooled_np = trunk.T.cpu().numpy()
        elif pooling_strategy == "mean":
            pooled_np = trunk.mean(dim=1).to(torch.float32).cpu().numpy()
        elif pooling_strategy == "max":
            pooled_np = trunk.max(dim=1).values.to(torch.float32).cpu().numpy()
        else:  # "median"
            pooled_np = trunk.median(dim=1).values.to(torch.float32).cpu().numpy()

        return pooled_np

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """
        Compute Scooby trunk embeddings for a batch of sequences.

        Scooby is only validated with batch size 1 (see module docstring),
        so unlike `BorzoiWrapper.embed_batch` this does not attempt to
        concatenate sequences into one forward pass -- it simply loops
        `embed()` over `inputs`.

        Parameters
        ----------
        inputs : Sequence[str]
            List of raw DNA sequence strings.
        pooling_strategy : str, default "mean"
            Forwarded to :meth:`embed`.
        **kwargs : Any
            Forwarded to :meth:`embed`.

        Returns
        -------
        list[np.ndarray]
            One 1D embedding per input sequence.
        """
        del kwargs
        return [self.embed(seq, pooling_strategy=pooling_strategy) for seq in inputs]
