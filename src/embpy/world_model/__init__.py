"""Cellular / perturbation world models.

A world model in :mod:`embpy` learns the latent dynamics of cell states
under perturbation. Given a basal cell-state latent ``z`` and a
perturbation conditioning vector ``c`` (drug embedding, gene-KO vector,
cytokine + dose, ...), it predicts the perturbed latent ``z'``::

    z' = f_theta(z, c)

The latent is produced by an upstream encoder (typically a single-cell
foundation model from :mod:`embpy.models.singlecell_models`, e.g. scVI,
STATE or Stack); the world model itself is encoder-agnostic and operates
purely on latents. This separation lets you swap encoders without
retraining the dynamics, and keeps the world model lightweight.

Public components
-----------------

* :class:`BaseWorldModel` -- abstract :class:`torch.nn.Module` interface
  every world model implements.
* :class:`LatentDynamicsWorldModel` -- a residual, FiLM-conditioned
  baseline implementation. Recommended starting point.
* :class:`PerturbationLatentDataset` -- AnnData-backed dataset producing
  ``(z_basal, z_perturbed, cond)`` triplets.
* :class:`WorldModelTrainer` / :class:`TrainConfig` -- minimal PyTorch
  training loop.
* Loss helpers in :mod:`embpy.world_model.losses`
  (:func:`latent_mse`, :func:`delta_mse`, :func:`gaussian_nll`,
  :func:`info_nce`).
* Building blocks in :mod:`embpy.world_model.modules`
  (:class:`MLP`, :class:`FiLM`, :class:`ResidualBlock`,
  :class:`SinusoidalTimeEmbedding`).

Examples
--------
>>> import embpy
>>> from embpy.world_model import (
...     LatentDynamicsWorldModel, PerturbationLatentDataset,
...     WorldModelTrainer, TrainConfig,
... )
>>> dataset = PerturbationLatentDataset(
...     adata, embedding_key="X_scvi",
...     perturbation_key="perturbation",
...     cond_embeddings=drug_embedding_lookup,
... )
>>> model = LatentDynamicsWorldModel(
...     latent_dim=dataset.latent_dim, cond_dim=dataset.cond_dim,
... )
>>> trainer = WorldModelTrainer(model, config=TrainConfig(n_epochs=10))
>>> from torch.utils.data import DataLoader
>>> trainer.fit(DataLoader(dataset, batch_size=128, shuffle=True))
"""

from __future__ import annotations

from . import losses, modules
from .base import BaseWorldModel
from .dataset import PerturbationLatentDataset
from .latent_dynamics import LatentDynamicsWorldModel
from .losses import delta_mse, gaussian_nll, info_nce, latent_mse
from .modules import MLP, FiLM, ResidualBlock, SinusoidalTimeEmbedding
from .trainer import TrainConfig, WorldModelTrainer

__all__ = [
    "BaseWorldModel",
    "FiLM",
    "LatentDynamicsWorldModel",
    "MLP",
    "PerturbationLatentDataset",
    "ResidualBlock",
    "SinusoidalTimeEmbedding",
    "TrainConfig",
    "WorldModelTrainer",
    "delta_mse",
    "gaussian_nll",
    "info_nce",
    "latent_mse",
    "losses",
    "modules",
]
