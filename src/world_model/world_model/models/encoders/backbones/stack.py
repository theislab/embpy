"""STACK backbone provider.

Thin adapter over :class:`embpy.models.singlecell_models.StackWrapper`.
The ``arc-stack`` package is loaded lazily inside :meth:`_load`.

STACK does not expose a latent-to-expression decoder; its analog is
*in-context generation* via :meth:`StackWrapper.generate_cells`. We
surface that as an explicit optional accessor (``generate_cells``) so
notebooks can still use it, but :meth:`decode` itself raises
``NotImplementedError`` because the provider contract is
``(N, embedding_dim) -> (N, n_genes)`` and STACK doesn't fit that.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import load_cached, save_cached
from .provider import StateBackboneProvider
from .state import _hash_adata, _hash_file

logger = logging.getLogger(__name__)


class StackBackbone(StateBackboneProvider):
    """STACK (Arc Institute) cell-embedding backbone."""

    name: str = "stack"
    supports_decode: bool = False

    def __init__(
        self,
        checkpoint: str = "",
        *,
        genelist: str = "",
        gene_name_col: str | None = None,
        device: str = "auto",
        freeze: bool = True,
        batch_size: int = 32,
        cache_dir: Path | str | None = None,
        require_cache_hit: bool = False,
    ) -> None:
        if not checkpoint:
            raise ValueError("StackBackbone requires `checkpoint`.")
        if not genelist:
            raise ValueError("StackBackbone requires `genelist`.")
        self._checkpoint = checkpoint
        self._genelist = genelist
        self._gene_name_col = gene_name_col
        self._device_cfg = device
        self._freeze_default = bool(freeze)
        self._batch_size = int(batch_size)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._require_cache_hit = bool(require_cache_hit)
        self._wrapper: Any = None
        self._embedding_dim: int | None = None
        self._ckpt_hash = hashlib.sha256(
            (_hash_file(checkpoint) + ":" + _hash_file(genelist)).encode("utf-8")
        ).hexdigest()[:16]

    def _resolve_device(self) -> str:
        if self._device_cfg != "auto":
            return self._device_cfg
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._wrapper is not None:
            return
        try:
            from embpy.models.singlecell_models import StackWrapper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "StackBackbone requires the `arc-stack` package. "
                "Install with: pip install 'embpy[stack]'  (or: pip install arc-stack)"
            ) from exc

        wrapper = StackWrapper(
            checkpoint=self._checkpoint,
            genelist=self._genelist,
            gene_name_col=self._gene_name_col,
            batch_size=self._batch_size,
        )
        wrapper.load(self._resolve_device())
        self._wrapper = wrapper
        model = wrapper._model  # noqa: SLF001
        # arc-stack's StateICLModel does not expose a single `embedding_dim`
        # attribute. Per stack/models/core/inference.py:432 the cell
        # embedding shape is (n_cells, n_hidden * token_dim), so we derive
        # the dim from those two architectural hyperparameters. Tolerate
        # alternative names that future STACK releases may introduce
        # before falling back to an explicit error.
        n_hidden = getattr(model, "n_hidden", None)
        token_dim = getattr(model, "token_dim", None)
        if n_hidden is not None and token_dim is not None:
            dim: int | None = int(n_hidden) * int(token_dim)
        else:
            dim = (
                getattr(model, "embedding_dim", None)
                or getattr(model, "hidden_size", None)
            )
        if dim is None:
            raise RuntimeError(
                "StackBackbone could not infer embedding_dim from the loaded "
                "model. Expected either `n_hidden` + `token_dim` (current "
                "arc-stack StateICLModel) or `embedding_dim` / `hidden_size`."
            )
        self._embedding_dim = int(dim)
        if self._freeze_default:
            self.freeze()

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load()
        return int(self._embedding_dim or 0)

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint

    def _cache_key(self, adata: Any) -> tuple[str, str]:
        return self._ckpt_hash, _hash_adata(adata)

    def encode(self, adata: Any, *, batch_size: int | None = None) -> np.ndarray:
        del batch_size
        ckpt_hash, ds_hash = self._cache_key(adata)
        if self._cache_dir is not None:
            cached = load_cached(
                self._cache_dir,
                backbone=self.name,
                ckpt_hash=ckpt_hash,
                dataset_hash=ds_hash,
            )
            if cached is not None:
                if self._embedding_dim is None:
                    self._embedding_dim = int(cached[0].shape[1])
                logger.info(
                    "STACK cache hit (%s/%s) -- %d cells, dim=%d.",
                    ckpt_hash, ds_hash, cached[0].shape[0], self._embedding_dim,
                )
                return cached[0]
            if self._require_cache_hit:
                raise RuntimeError(
                    f"require_cache_hit=True but no cached STACK embeddings for "
                    f"({ckpt_hash}, {ds_hash}) under {self._cache_dir}."
                )

        self._load()
        assert self._wrapper is not None
        embeddings = self._wrapper.embed_cells(adata)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._cache_dir is not None:
            save_cached(
                self._cache_dir,
                backbone=self.name,
                ckpt_hash=ckpt_hash,
                dataset_hash=ds_hash,
                embeddings=embeddings,
                adata_hash=ds_hash,
            )
        return embeddings

    def decode(
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        del latent, gene_names, kwargs
        raise NotImplementedError(
            "STACK does not implement a latent-to-expression decoder. For "
            "in-context generation use StackBackbone.generate_cells(base_adata, test_adata)."
        )

    def generate_cells(self, base_adata: Any, test_adata: Any, **kwargs: Any) -> np.ndarray:
        """Forward to :meth:`StackWrapper.generate_cells`.

        Exists so users who want STACK's in-context generation head can
        reach it via the provider without dropping back to the wrapper.
        """
        self._load()
        assert self._wrapper is not None
        return self._wrapper.generate_cells(base_adata, test_adata, **kwargs)

    def freeze(self) -> None:
        if self._wrapper is None:
            return
        model = self._wrapper._model  # noqa: SLF001
        if not isinstance(model, torch.nn.Module):
            return
        n = 0
        for p in model.parameters():
            p.requires_grad_(False)
            n += p.numel()
        logger.info("StackBackbone froze %d parameters.", n)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        if self._wrapper is None:
            return iter(())
        model = self._wrapper._model  # noqa: SLF001
        if not isinstance(model, torch.nn.Module):
            return iter(())
        return model.parameters()

    def train_mode(self, flag: bool) -> None:
        if self._wrapper is None:
            return
        model = self._wrapper._model  # noqa: SLF001
        if isinstance(model, torch.nn.Module):
            model.train(bool(flag))


__all__ = ["StackBackbone"]
