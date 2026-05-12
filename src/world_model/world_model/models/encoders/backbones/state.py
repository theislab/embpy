"""STATE backbone provider.

Thin adapter over :class:`embpy.models.singlecell_models.StateEmbeddingWrapper`.
The wrapper itself is the consumer of the upstream ``arc-state`` package;
we never import ``state.*`` here. ``arc-state`` only gets loaded when
:meth:`_load` runs (i.e. the first :meth:`encode` call).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import cache_path_for, load_cached, save_cached
from .provider import StateBackboneProvider

logger = logging.getLogger(__name__)


def _hash_seq(strs: Sequence[str]) -> str:
    h = hashlib.sha256()
    for s in strs:
        h.update(s.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _hash_file(path: str | None) -> str:
    if not path:
        return "no-ckpt"
    try:
        st = Path(path).stat()
        # Avoid streaming multi-GB checkpoints: hash (path, size, mtime).
        # If the file changes mtime we treat it as a different checkpoint.
        h = hashlib.sha256()
        h.update(str(Path(path).resolve()).encode("utf-8"))
        h.update(str(int(st.st_size)).encode("utf-8"))
        h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
        return h.hexdigest()[:16]
    except OSError:
        return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def _hash_adata(adata: Any) -> str:
    parts: list[str] = []
    # NOTE: `getattr(adata, "var_names", []) or []` looks innocuous but
    # raises "ValueError: The truth value of a Index is ambiguous" when
    # var_names is a pandas Index with len > 1 -- Python evaluates `or`
    # by calling `__bool__`, which Index rejects. Use an explicit None
    # check instead.
    var_names = getattr(adata, "var_names", None)
    obs_names = getattr(adata, "obs_names", None)
    var = list(var_names) if var_names is not None else []
    obs = list(obs_names) if obs_names is not None else []
    parts.append(f"vn:{len(var)}")
    parts.append(f"on:{len(obs)}")
    parts.append("v:" + _hash_seq([str(x) for x in var]))
    parts.append("o:" + _hash_seq([str(x) for x in obs]))
    return _hash_seq(parts)


class StateBackbone(StateBackboneProvider):
    """STATE (Arc Institute) cell-embedding backbone."""

    name: str = "state"
    supports_decode: bool = True

    def __init__(
        self,
        checkpoint: str = "",
        *,
        model_folder: str | None = None,
        protein_embeddings: str | None = None,
        config: str | None = None,
        device: str = "auto",
        freeze: bool = True,
        batch_size: int = 64,
        cache_dir: Path | str | None = None,
        require_cache_hit: bool = False,
    ) -> None:
        if not checkpoint and not model_folder:
            raise ValueError("StateBackbone requires either `checkpoint` or `model_folder`.")
        self._checkpoint = checkpoint
        self._model_folder = model_folder
        self._protein_embeddings = protein_embeddings
        self._config = config
        self._device_cfg = device
        self._freeze_default = bool(freeze)
        self._batch_size = int(batch_size)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._require_cache_hit = bool(require_cache_hit)
        self._wrapper: Any = None
        self._embedding_dim: int | None = None
        self._ckpt_hash: str = _hash_file(checkpoint or (model_folder or ""))

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._device_cfg != "auto":
            return self._device_cfg
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._wrapper is not None:
            return
        try:
            from embpy.models.singlecell_models import StateEmbeddingWrapper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "StateBackbone requires the `arc-state` package. "
                "Install with: pip install 'embpy[state]'  (or: pip install arc-state)"
            ) from exc

        wrapper = StateEmbeddingWrapper(
            checkpoint=self._checkpoint or None,
            model_folder=self._model_folder,
            protein_embeddings=self._protein_embeddings,
            config=self._config,
            batch_size=self._batch_size,
        )
        wrapper.load(self._resolve_device())
        self._wrapper = wrapper
        # See state.emb.Inference: z_dim is the per-cell embedding width and
        # z_dim_ds (when present) is concatenated as a dataset embedding at
        # encode time, matching encode_adata's output shape.
        # Ref: https://github.com/ArcInstitute/state/blob/main/src/state/emb/inference.py
        model = wrapper._inferer.model  # noqa: SLF001 -- documented upstream API
        z_dim = int(getattr(model, "z_dim"))
        z_dim_ds = int(getattr(model, "z_dim_ds", 0) or 0)
        self._embedding_dim = z_dim + z_dim_ds
        if self._freeze_default:
            self.freeze()

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load()
        return int(self._embedding_dim or 0)

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint or (self._model_folder or "")

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
                logger.info(
                    "STATE cache hit (%s/%s) -- %d cells.",
                    ckpt_hash, ds_hash, cached[0].shape[0],
                )
                return cached[0]
            if self._require_cache_hit:
                raise RuntimeError(
                    f"require_cache_hit=True but no cached STATE embeddings for "
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
        self._load()
        assert self._wrapper is not None
        return self._wrapper.decode_cells(latent, gene_names=list(gene_names), **kwargs)

    def freeze(self) -> None:
        if self._wrapper is None:
            return
        n = 0
        for p in self._wrapper._inferer.model.parameters():  # noqa: SLF001
            p.requires_grad_(False)
            n += p.numel()
        logger.info("StateBackbone froze %d parameters.", n)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        if self._wrapper is None:
            return iter(())
        return self._wrapper._inferer.model.parameters()  # noqa: SLF001

    def train_mode(self, flag: bool) -> None:
        if self._wrapper is None:
            return
        self._wrapper._inferer.model.train(bool(flag))  # noqa: SLF001


__all__ = ["StateBackbone"]
