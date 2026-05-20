"""Provider ABC + metadata dataclass for state backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class ProviderMetadata:
    """Metadata persisted to ``runs/<run>/state_backbone_meta.json``."""

    kind: str
    name: str
    embedding_dim: int
    supports_decode: bool
    checkpoint_hash: str = ""
    checkpoint_path: str = ""
    n_cells_encoded: int = 0
    cache_hit: bool = False
    cache_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class StateBackboneProvider(ABC):
    """Common interface for state-encoder backbones.

    Two flavours of usage:

    * **Local** -- the provider *is* the state-stack encoder. Forward
      operates on raw expression tensors of shape ``(B, T, K, G)``.
      No pre-encoding step is required.
    * **Foreign (STATE / STACK)** -- the provider holds a frozen
      foundation model. The dataloader pre-encodes the AnnData once
      through :meth:`encode`, caches the result on disk, and replaces
      the in-memory expression matrix with the cached embeddings.
      Downstream the world model sees only ``(B, T, K, embedding_dim)``
      tensors after the dataset's stack step.

    Implementations must keep their underlying torch model importable
    and operable behind a *single* :meth:`_load` call -- the provider
    is constructed cheaply (no network / GPU traffic) and only the
    first :meth:`encode` (or an explicit :meth:`load`) materialises
    the backbone.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in metadata and logs (``"local"``, ``"state"``, ``"stack"``, ...)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output dimensionality of :meth:`encode`."""

    @property
    @abstractmethod
    def supports_decode(self) -> bool:
        """Whether :meth:`decode` is implemented (True for STATE; False for STACK)."""

    @abstractmethod
    def encode(self, adata: Any, *, batch_size: int | None = None) -> np.ndarray:
        """Encode an AnnData of ``N`` cells into ``(N, embedding_dim)`` float32.

        ``adata`` may be any object accepted by the underlying wrapper.
        ``batch_size`` lets the caller override the construction-time
        default for a one-off encode (useful at smoke-test time).
        """

    def decode(
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Decode ``(N, embedding_dim)`` latents back to ``(N, n_genes)``.

        Default implementation raises -- only :class:`StateBackbone`
        actually implements a latent-to-expression decoder. STACK
        offers in-context generation (see :meth:`StackBackbone.generate_cells`)
        rather than a pure decoder, so its :meth:`decode` also raises.
        """
        raise NotImplementedError(
            f"{self.name} does not support decode(). Use a STATE backbone "
            f"or, for STACK, call generate_cells via the optional accessor."
        )

    @abstractmethod
    def freeze(self) -> None:
        """Set ``requires_grad=False`` on every parameter the provider exposes.

        Implementations should log how many params were frozen so the
        training log can be eyeballed for accidental unfreezes.
        """

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Iterable of backbone parameters.

        ``Local`` returns an empty iterable because its weights are
        already registered inside :class:`world_model.WorldModel` and
        :meth:`iter_trainable_params` deduplicates by id. ``STATE`` /
        ``STACK`` return the underlying torch model's parameters so the
        optimizer can pick them up when ``freeze=False``.
        """

    @abstractmethod
    def train_mode(self, flag: bool) -> None:
        """Toggle ``train()`` vs ``eval()`` on the underlying torch model.

        Called by the trainer at training start / eval boundaries when
        ``freeze=False`` so dropout / norm statistics behave correctly.
        ``Local`` is a no-op (the wrapped encoder is a child module of
        the world model and already participates in ``model.train()``).
        """

    # ----- Optional override hooks --------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return ""

    def metadata(self, *, n_cells_encoded: int, cache_hit: bool, cache_path: str) -> ProviderMetadata:
        """Build a :class:`ProviderMetadata` describing this encode call."""
        return ProviderMetadata(
            kind=self.name,
            name=self.name,
            embedding_dim=self.embedding_dim,
            supports_decode=self.supports_decode,
            checkpoint_path=self.checkpoint_path,
            n_cells_encoded=int(n_cells_encoded),
            cache_hit=bool(cache_hit),
            cache_path=cache_path,
        )


__all__ = ["ProviderMetadata", "StateBackboneProvider"]
