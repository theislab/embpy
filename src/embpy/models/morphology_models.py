"""Morphology embedding models for fluorescence microscopy images.

Wraps SubCell ViT-MAE models trained on Human Protein Atlas images.
SubCell takes multi-channel fluorescence images and produces rich
cell morphology embeddings (1536-dim via gated attention pooling).

8 model variants across 4 channel configurations x 2 architectures:

Channel configs:
- ``rybg`` (4ch): Red=microtubules, Yellow=ER, Blue=nucleus, Green=protein
- ``rbg`` (3ch): microtubules + nucleus + protein
- ``ybg`` (3ch): ER + nucleus + protein
- ``bg`` (2ch): nucleus + protein

Architectures:
- ``mae_contrast_supcon_model`` -- MAE + contrastive + supervised contrastive
- ``vit_supcon_model`` -- ViT + supervised contrastive

Weights are auto-downloaded from the public CZI S3 bucket on first use.

Reference: https://github.com/CellProfiling/SubCellPortable
"""

from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

S3_BASE = "https://czi-subcell-public.s3.amazonaws.com/models"

SUBCELL_MODELS = {
    "subcell_mae_rybg": {
        "encoder": f"{S3_BASE}/all_channels_MAE-CellS-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/all_channels_MAE_MLP_classifier/all_channels_MAE_MLP_classifier_seed_0.pth",
        "channels": "rybg",
        "num_channels": 4,
    },
    "subcell_vit_rybg": {
        "encoder": f"{S3_BASE}/all_channels_ViT-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/all_channels_ViT_MLP_classifier/all_channels_ViT_MLP_classifier_seed_0.pth",
        "channels": "rybg",
        "num_channels": 4,
    },
    "subcell_mae_rbg": {
        "encoder": f"{S3_BASE}/MT-DNA-Protein_MAE-CellS-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/MT-DNA-Protein_MAE_MLP_classifier/MT-DNA-Protein_MAE_MLP_classifier_seed_0.pth",
        "channels": "rbg",
        "num_channels": 3,
    },
    "subcell_vit_rbg": {
        "encoder": f"{S3_BASE}/MT-DNA-Protein_ViT-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/MT-DNA-Protein_ViT_MLP_classifier/MT-DNA-Protein_ViT_MLP_classifier_seed_0.pth",
        "channels": "rbg",
        "num_channels": 3,
    },
    "subcell_mae_ybg": {
        "encoder": f"{S3_BASE}/ER-DNA-Protein_MAE-CellS-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/ER-DNA-Protein_MAE_MLP_classifier/ER-DNA-Protein_MAE_MLP_classifier_seed_0.pth",
        "channels": "ybg",
        "num_channels": 3,
    },
    "subcell_vit_ybg": {
        "encoder": f"{S3_BASE}/ER-DNA-Protein_ViT-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/ER-DNA-Protein_ViT_MLP_classifier/ER-DNA-Protein_ViT_MLP_classifier_seed_0.pth",
        "channels": "ybg",
        "num_channels": 3,
    },
    "subcell_mae_bg": {
        "encoder": f"{S3_BASE}/DNA-Protein_MAE-CellS-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/DNA-Protein_MAE_MLP_classifier/DNA-Protein_MAE_MLP_classifier_seed_0.pth",
        "channels": "bg",
        "num_channels": 2,
    },
    "subcell_vit_bg": {
        "encoder": f"{S3_BASE}/DNA-Protein_ViT-ProtS-Pool.pth",
        "classifier": f"{S3_BASE}/DNA-Protein_ViT_MLP_classifier/DNA-Protein_ViT_MLP_classifier_seed_0.pth",
        "channels": "bg",
        "num_channels": 2,
    },
}

SUBCELL_ALIASES = {
    "subcell_mae": "subcell_mae_rybg",
    "subcell_cellprot": "subcell_mae_rybg",
    "subcell_contrast": "subcell_mae_rybg",
    "subcell_vit": "subcell_vit_rybg",
}

SUBCELL_VIT_CONFIG_BASE = {
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
    "qkv_bias": True,
    "decoder_num_attention_heads": 16,
    "decoder_hidden_size": 512,
    "decoder_num_hidden_layers": 8,
    "decoder_intermediate_size": 2048,
    "mask_ratio": 0.0,
    "norm_pix_loss": True,
}


def _download_weight(url: str, dest: Path) -> Path:
    """Download a model weight file from S3 if not cached."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SubCell weight: %s", dest.name)
    urllib.request.urlretrieve(url, str(dest))
    logger.info("Downloaded %.1f MB -> %s", dest.stat().st_size / 1e6, dest)
    return dest


class SubCellWrapper(BaseModelWrapper):
    """Extract morphology embeddings from SubCell ViT-MAE models.

    Weights are auto-downloaded from CZI's public S3 bucket on first use.

    Parameters
    ----------
    model_path_or_name
        Model key (e.g. ``"subcell_mae_rybg"``) or path to a local
        ``.pth`` encoder file. Available keys: see ``SUBCELL_MODELS``.
    variant
        Alias: ``"mae"``, ``"vit"``, ``"contrast"``, ``"cellprot"``.
    image_size
        Input image resolution (default 448).
    """

    model_type = "morphology"
    available_pooling_strategies = ["cls", "mean", "attention_pool", "none"]

    def __init__(
        self,
        model_path_or_name: str = "subcell_mae_rybg",
        variant: str | None = None,
        image_size: int = 448,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self.image_size = image_size
        self._encoder = None
        self._pool_model = None
        self._num_channels = 4

        key = model_path_or_name
        if key in SUBCELL_ALIASES:
            key = SUBCELL_ALIASES[key]
        if variant and f"subcell_{variant}" in SUBCELL_ALIASES:
            key = SUBCELL_ALIASES[f"subcell_{variant}"]

        self._model_key = key if key in SUBCELL_MODELS else None

    def load(self, device: torch.device) -> None:
        """Load the SubCell encoder, downloading weights if needed."""
        if self._encoder is not None:
            return

        from transformers import ViTMAEConfig

        if self._model_key and self._model_key in SUBCELL_MODELS:
            model_info = SUBCELL_MODELS[self._model_key]
            self._num_channels = model_info["num_channels"]

            cache_dir = Path(
                os.environ.get("EMBPY_CACHE", Path.home() / ".cache" / "embpy")
            ) / "subcell"

            encoder_url = model_info["encoder"]
            encoder_name = Path(encoder_url).name
            encoder_path = _download_weight(encoder_url, cache_dir / encoder_name)

            vit_config = {**SUBCELL_VIT_CONFIG_BASE, "num_channels": self._num_channels}
            config = ViTMAEConfig(**vit_config)

            logger.info("Loading SubCell encoder: %s (%dch)...", self._model_key, self._num_channels)
            self._encoder = _build_vit_encoder(config)

            state_dict = torch.load(str(encoder_path), map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            clean_state = {}
            pool_state = {}
            for k, v in state_dict.items():
                if "pool_model" in k or "pool" in k.split(".")[0]:
                    pool_key = k.split("pool_model.")[-1] if "pool_model." in k else k.split("pool.")[-1]
                    pool_state[pool_key] = v
                elif "classifier" not in k and "decoder" not in k and "mask_token" not in k:
                    clean_key = k
                    for prefix in ["model.mae_model.vit_mae.", "model.mae_model.", "model.", "vit_mae.", "encoder."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    clean_state[clean_key] = v

            self._encoder.load_state_dict(clean_state, strict=False)

            if pool_state:
                self._pool_model = _build_attention_pooler(
                    dim=768, int_dim=512, num_heads=2,
                )
                self._pool_model.load_state_dict(pool_state, strict=False)
                self._pool_model = self._pool_model.to(device).eval()

        elif self.model_name and Path(self.model_name).exists():
            vit_config = {**SUBCELL_VIT_CONFIG_BASE, "num_channels": self._num_channels}
            config = ViTMAEConfig(**vit_config)
            self._encoder = _build_vit_encoder(config)

            state_dict = torch.load(self.model_name, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._encoder.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(
                f"SubCell model '{self.model_name}' not found. "
                f"Available: {list(SUBCELL_MODELS.keys())}"
            )

        self._encoder = self._encoder.to(device).eval()
        self.model = self._encoder
        self.device = device
        logger.info("SubCell loaded. Embedding dim: %s",
                     "1536 (attention_pool)" if self._pool_model else "768 (cls/mean)")

    def _preprocess_image(
        self, image: str | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess an image for SubCell inference.

        Returns a tensor of shape (1, C, 448, 448) normalized to [0, 1].
        """
        if isinstance(image, str):
            from PIL import Image
            img = Image.open(image)
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * self._num_channels, axis=-1)
            tensor = torch.from_numpy(arr).float()
            if tensor.ndim == 3 and tensor.shape[-1] <= 4:
                tensor = tensor.permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] <= 4 and image.shape[0] > 4:
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            elif image.ndim == 3 and image.shape[0] <= 4:
                tensor = torch.from_numpy(image).float()
            else:
                raise ValueError(f"Expected (H,W,C) or (C,H,W), got {image.shape}")
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.shape[0] > 4 and tensor.shape[-1] <= 4:
                tensor = tensor.permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        n_ch = tensor.shape[0]
        if n_ch < self._num_channels:
            padding = torch.zeros(self._num_channels - n_ch, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, padding], dim=0)
        elif n_ch > self._num_channels:
            tensor = tensor[:self._num_channels]

        for c in range(self._num_channels):
            cmin, cmax = tensor[c].min(), tensor[c].max()
            if cmax > cmin:
                tensor[c] = (tensor[c] - cmin) / (cmax - cmin)

        if tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        return tensor.unsqueeze(0)

    def embed(
        self,
        input: str | np.ndarray | torch.Tensor,
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute morphology embedding for a microscopy image.

        Parameters
        ----------
        input
            Image path, numpy array, or torch tensor.
        pooling_strategy
            ``"cls"`` (768d), ``"mean"`` (768d),
            ``"attention_pool"`` (1536d, recommended), or
            ``"none"`` (num_tokens x 768 -- raw per-patch tokens).

        Returns
        -------
        np.ndarray
            1D for pooled strategies, 2D ``(num_tokens, 768)`` for ``"none"``.
        """
        if self._encoder is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pixel_values = self._preprocess_image(input).to(self.device)

        with torch.no_grad():
            outputs = self._encoder(pixel_values=pixel_values, mask_ratio=0.0)
            hidden = outputs.last_hidden_state

        if pooling_strategy == "none":
            emb = hidden.squeeze(0).cpu().numpy()
        elif pooling_strategy == "cls":
            emb = hidden[:, 0, :].cpu().numpy().squeeze(0)
        elif pooling_strategy == "mean":
            emb = hidden[:, 1:, :].mean(dim=1).cpu().numpy().squeeze(0)
        elif pooling_strategy == "attention_pool":
            if self._pool_model is not None:
                pooled, _ = self._pool_model(hidden[:, 1:, :])
                emb = pooled.cpu().numpy().squeeze(0)
            else:
                logger.warning("No attention pooler; falling back to CLS.")
                emb = hidden[:, 0, :].cpu().numpy().squeeze(0)
        else:
            raise ValueError(f"Unknown pooling '{pooling_strategy}'")

        return emb.astype(np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str | np.ndarray | torch.Tensor],
        pooling_strategy: str = "cls",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute embeddings for a batch of images."""
        n_patches = (self.image_size // 16) ** 2 + 1
        results = []
        for img in inputs:
            try:
                results.append(self.embed(img, pooling_strategy=pooling_strategy))
            except Exception as e:  # noqa: BLE001
                logger.warning("SubCell embedding failed: %s", e)
                if pooling_strategy == "none":
                    results.append(np.zeros((n_patches, 768), dtype=np.float32))
                elif pooling_strategy == "attention_pool" and self._pool_model:
                    results.append(np.zeros(1536, dtype=np.float32))
                else:
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
