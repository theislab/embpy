"""Abstract base class for cellular / perturbation world models.

A *world model* in ``embpy`` is a learned model of cell-state dynamics. Given

* a **basal** latent state ``z`` (an embedding of an unperturbed cell), and
* a **perturbation** condition ``c`` (drug embedding, gene-KO embedding,
  cytokine vector, dose, time delta, ...),

it predicts the **perturbed** latent state ``z'`` that the cell would occupy
after applying ``c``. Implementations may be deterministic
(``z' = f(z, c)``), stochastic (predicting a distribution over ``z'``), or
flow-based (predicting a velocity field ``v(z, c, t)``); they all share the
same interface defined here.

The abstraction is deliberately kept latent-only: encoding gene expression
``X`` to ``z`` and decoding ``z'`` back to ``X'`` is delegated to the
single-cell foundation models in :mod:`embpy.models.singlecell_models`
(scVI, STATE, Stack, ...). This keeps the world model lightweight and
swappable across encoders.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseWorldModel(nn.Module, ABC):
    """Abstract latent-space world model for cell-state dynamics.

    Subclasses implement :meth:`forward` (the predictive model) and may
    override :meth:`rollout` for multi-step prediction. The class extends
    :class:`torch.nn.Module` so trained models can be saved/loaded with
    standard PyTorch tooling.

    Attributes
    ----------
    latent_dim
        Dimensionality of the cell-state latent ``z``.
    cond_dim
        Dimensionality of the perturbation conditioning vector ``c``.
    name
        Short identifier used in logging and registry lookups.
    """

    name: str = "base_world_model"

    def __init__(self, latent_dim: int, cond_dim: int, **kwargs: Any) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be non-negative, got {cond_dim}")
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.config: dict[str, Any] = dict(kwargs)

    # ------------------------------------------------------------------
    # Core predictive interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the perturbed latent state.

        Parameters
        ----------
        z
            Basal latent of shape ``(batch, latent_dim)``.
        cond
            Perturbation conditioning of shape ``(batch, cond_dim)``.
            May be ``None`` for unconditional models.
        **kwargs
            Implementation-specific extras (e.g. ``t`` for flow models,
            ``dose``, ``time_delta``).

        Returns
        -------
        torch.Tensor
            Predicted perturbed latent of shape ``(batch, latent_dim)``.
        """

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        z: torch.Tensor,
        cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Eval-mode forward pass with gradients disabled."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(z, cond, **kwargs)
        finally:
            if was_training:
                self.train()
        return out

    def rollout(
        self,
        z0: torch.Tensor,
        conds: Sequence[torch.Tensor | None],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Apply a sequence of perturbations starting from ``z0``.

        Default behaviour treats each step as ``z_{t+1} = f(z_t, c_t)``.
        Override for models with explicit time integration (e.g. flow-
        matching) where intermediate trajectories matter.

        Parameters
        ----------
        z0
            Initial latent ``(batch, latent_dim)``.
        conds
            Sequence of conditioning tensors, one per step. Each entry may
            be ``None`` to skip conditioning at that step.
        **kwargs
            Forwarded to :meth:`forward` at every step.

        Returns
        -------
        list[torch.Tensor]
            Latents ``[z0, z1, ..., zT]`` of length ``len(conds) + 1``.
        """
        trajectory: list[torch.Tensor] = [z0]
        z = z0
        for c in conds:
            z = self.predict(z, c, **kwargs)
            trajectory.append(z)
        return trajectory

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state-dict and config to ``path``.

        The file is a standard ``torch.save`` archive containing both the
        weights and the constructor config so subclasses can reconstruct
        themselves via :meth:`load`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)
        logger.info("Saved %s to %s", self.__class__.__name__, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> BaseWorldModel:
        """Load a saved world model.

        Subclasses with non-trivial constructors should override this to
        unpack additional fields from ``payload["config"]``.
        """
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            latent_dim=payload["latent_dim"],
            cond_dim=payload["cond_dim"],
            **payload.get("config", {}),
        )
        model.load_state_dict(payload["state_dict"])
        logger.info("Loaded %s from %s", cls.__name__, path)
        return model

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of (trainable) parameters."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        return f"latent_dim={self.latent_dim}, cond_dim={self.cond_dim}, name='{self.name}'"
