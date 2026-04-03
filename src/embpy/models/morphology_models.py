"""Morphology embedding models for fluorescence microscopy images.

Wraps SubCell ViT-MAE models trained on Human Protein Atlas images.
SubCell takes 4-channel fluorescence images (Red=microtubules,
Yellow=ER, Blue=nucleus, Green=protein of interest) and produces
rich cell morphology embeddings.

Three model variants are supported:

- ``subcell_mae`` -- masked autoencoder reconstruction only
- ``subcell_cellprot`` -- cell-specific + protein-specific SSL
- ``subcell_contrast`` -- MAE + contrastive + cell/protein SSL (best)

Requires a local checkpoint file (``.ckpt``) obtained from the
SubCell authors. Once HuggingFace weights are published, auto-download
will be supported.

Reference: https://github.com/CellProfiling/subcell-embed
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

SUBCELL_VIT_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "image_size": 448,
    "patch_size": 16,
    "num_channels": 4,
    "qkv_bias": True,
    "decoder_num_attention_heads": 16,
    "decoder_hidden_size": 512,
    "decoder_num_hidden_layers": 8,
    "decoder_intermediate_size": 2048,
    "mask_ratio": 0.0,
    "norm_pix_loss": True,
}


class SubCellWrapper(BaseModelWrapper):
    """Extract morphology embeddings from SubCell ViT-MAE models.

    SubCell models are vision transformers (ViT-B/16) trained on
    4-channel fluorescence microscopy images from the Human Protein
    Atlas using self-supervised learning (MAE + contrastive + SSL).

    Parameters
    ----------
    model_path_or_name
        Path to a local ``.ckpt`` checkpoint file, or a model variant
        name (``"subcell_mae"``, ``"subcell_cellprot"``,
        ``"subcell_contrast"``).
    variant
        Model variant: ``"mae"``, ``"cellprot"``, or ``"contrast"``.
    image_size
        Input image resolution (default 448).
    """

    model_type = "morphology"
    available_pooling_strategies = ["cls", "mean", "attention_pool"]

    def __init__(
        self,
        model_path_or_name: str = "subcell_contrast",
        variant: str = "contrast",
        image_size: int = 448,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self.variant = variant
        self.image_size = image_size
        self._encoder = None
        self._pool_model = None

    def load(self, device: torch.device) -> None:
        """Load the SubCell ViT-MAE encoder from a checkpoint.

        Parameters
        ----------
        device
            Compute device.
        """
        if self._encoder is not None:
            return

        ckpt_path = self.model_name
        if ckpt_path is None:
            raise ValueError("model_path_or_name must be set.")

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"SubCell checkpoint not found at '{ckpt_path}'. "
                "Obtain weights from the SubCell authors: "
                "https://github.com/CellProfiling/subcell-embed"
            )

        logger.info("Loading SubCell %s from %s ...", self.variant, ckpt_path)

        try:
            from transformers import ViTMAEConfig

            config = ViTMAEConfig(**SUBCELL_VIT_CONFIG)

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)

            encoder_state = {}
            pool_state = {}
            for k, v in state_dict.items():
                clean_key = k.replace("model.mae_model.vit_mae.", "").replace("model.mae_model.", "")
                if "pool_model" in k:
                    pool_key = k.split("pool_model.")[-1]
                    pool_state[pool_key] = v
                elif "decoder" not in clean_key and "mask_token" not in clean_key:
                    encoder_state[clean_key] = v

            from .morphology_models import _build_vit_encoder
            self._encoder = _build_vit_encoder(config)
            missing, unexpected = self._encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                logger.debug("Missing keys in encoder: %s", missing[:5])

            if pool_state:
                self._pool_model = _build_attention_pooler(
                    dim=config.hidden_size, int_dim=512, num_heads=2,
                )
                self._pool_model.load_state_dict(pool_state, strict=False)
                self._pool_model = self._pool_model.to(device).eval()

            self._encoder = self._encoder.to(device).eval()
            self.model = self._encoder
            self.device = device
            logger.info("SubCell %s loaded successfully (768-dim embeddings).", self.variant)

        except Exception as e:
            logger.error("Failed to load SubCell: %s", e)
            raise RuntimeError(f"Could not load SubCell '{ckpt_path}'") from e

    def _preprocess_image(
        self, image: str | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess an image for SubCell inference.

        Accepts a file path, numpy array (H,W,4) or (4,H,W), or torch tensor.
        Returns a tensor of shape (1, 4, 448, 448) normalized to [0, 1].
        """
        if isinstance(image, str):
            from PIL import Image
            img = Image.open(image)
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 4, axis=-1)
            elif arr.shape[-1] == 3:
                arr = np.concatenate([arr, arr[:, :, :1]], axis=-1)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] == 4:
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            elif image.ndim == 3 and image.shape[0] == 4:
                tensor = torch.from_numpy(image).float()
            else:
                raise ValueError(f"Expected (H,W,4) or (4,H,W), got {image.shape}")
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.shape[0] != 4:
                tensor = tensor.permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        if tensor.shape[0] != 4:
            raise ValueError(f"Expected 4 channels (RYBG), got {tensor.shape[0]}")

        for c in range(4):
            cmin = tensor[c].min()
            cmax = tensor[c].max()
            if cmax > cmin:
                tensor[c] = (tensor[c] - cmin) / (cmax - cmin)
            else:
                tensor[c] = 0.0

        if tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor.unsqueeze(0)

    def embed(
        self,
        input: str | np.ndarray | torch.Tensor,
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute morphology embedding for a single microscopy image.

        Parameters
        ----------
        input
            Path to image file, numpy array ``(H,W,4)`` or ``(4,H,W)``,
            or torch tensor ``(4,H,W)``. Channels: Red (microtubules),
            Yellow (ER), Blue (nucleus), Green (protein of interest).
        pooling_strategy
            ``"cls"`` (CLS token, 768-dim), ``"mean"`` (mean of all
            patch tokens, 768-dim), or ``"attention_pool"`` (gated
            attention pooling, 1536-dim if available).
        **kwargs
            Ignored.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        if self._encoder is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pixel_values = self._preprocess_image(input).to(self.device)

        with torch.no_grad():
            outputs = self._encoder(pixel_values=pixel_values, mask_ratio=0.0)
            hidden = outputs.last_hidden_state

        if pooling_strategy == "cls":
            emb = hidden[:, 0, :].cpu().numpy().squeeze(0)
        elif pooling_strategy == "mean":
            emb = hidden[:, 1:, :].mean(dim=1).cpu().numpy().squeeze(0)
        elif pooling_strategy == "attention_pool":
            if self._pool_model is not None:
                pooled, _ = self._pool_model(hidden[:, 1:, :])
                emb = pooled.cpu().numpy().squeeze(0)
            else:
                emb = hidden[:, 0, :].cpu().numpy().squeeze(0)
                logger.warning("No attention pooler found, falling back to CLS.")
        else:
            raise ValueError(f"Unknown pooling '{pooling_strategy}'")

        return emb.astype(np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str | np.ndarray | torch.Tensor],
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute embeddings for a batch of microscopy images.

        Parameters
        ----------
        inputs
            List of images (paths, arrays, or tensors).
        pooling_strategy
            Pooling strategy.
        **kwargs
            Ignored.

        Returns
        -------
        list of np.ndarray
        """
        results = []
        for img in inputs:
            try:
                results.append(self.embed(img, pooling_strategy=pooling_strategy))
            except Exception as e:  # noqa: BLE001
                logger.warning("SubCell embedding failed: %s", e)
                results.append(np.zeros(768, dtype=np.float32))
        return results


def _build_vit_encoder(config) -> nn.Module:
    """Build the ViT-MAE encoder from a HuggingFace config."""
    from transformers import ViTMAEModel as HFViTMAEModel

    return HFViTMAEModel(config)


def _build_attention_pooler(
    dim: int = 768, int_dim: int = 512, num_heads: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Build the gated attention pooler matching SubCell's architecture."""

    class GatedAttentionPooler(nn.Module):
        def __init__(self, dim, int_dim, num_heads, dropout):
            super().__init__()
            self.attention_v = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, int_dim), nn.Tanh(),
            )
            self.attention_u = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, int_dim), nn.GELU(),
            )
            self.attention = nn.Linear(int_dim, num_heads)
            self.softmax = nn.Softmax(dim=-1)
            self.out_dim = dim * num_heads

        def forward(self, x):
            v = self.attention_v(x)
            u = self.attention_u(x)
            attn = self.attention(v * u).permute(0, 2, 1)
            attn = self.softmax(attn)
            x = torch.bmm(attn, x)
            x = x.view(x.shape[0], -1)
            return x, attn

    return GatedAttentionPooler(dim, int_dim, num_heads, dropout)
