"""Perturbation world model for single-cell transcriptomics.

A modular implementation of an autoregressive (state, action) world
model where:

* the **state** is a stack of gene-expression vectors plus their
  learned embeddings,
* the **action** is the embedding of the perturbed gene(s),
* the **dynamics** is a GPT-style causal transformer that predicts the
  next state token given the past states and actions.

The architecture is the transcriptomics analogue of the model proposed
in `Learning World Models for Unconstrained Goal Navigation
<https://arxiv.org/pdf/2405.18193>`_ and the implementation pattern is
adapted from `In-Context-Symmetries
<https://github.com/Sharut/In-Context-Symmetries>`_.

Quick start
-----------

.. code-block:: python

    from embpy.world_model.configs import load_yaml_config
    from embpy.world_model.data import build_dataloaders
    from embpy.world_model.models.world_model import build_world_model
    from embpy.world_model.training import WorldModelTrainer

    cfg = load_yaml_config("src/embpy/world_model/configs/replogle.yaml")
    train_loader, val_loader, gene_table, indexer, gene_symbols = build_dataloaders(
        cfg.data, seed=cfg.seed,
    )
    import torch
    model = build_world_model(
        n_genes=len(gene_symbols),
        gene_embedding_table=torch.from_numpy(gene_table),
        d_model=cfg.encoder.d_model,
        stack_size=cfg.data.stack_size,
        max_sequence_length=cfg.dynamics.max_sequence_length,
    )
    trainer = WorldModelTrainer(
        model, optim_cfg=cfg.optim, loss_cfg=cfg.loss, train_cfg=cfg.train,
        run_name=cfg.run_name, output_dir=cfg.output_dir,
    )
    trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

from . import configs, data, evaluation, models, training, utils
from .configs import (
    ActionEmbeddingConfig,
    DataConfig,
    DynamicsConfig,
    EncoderConfig,
    LossConfig,
    OptimConfig,
    TrainConfig,
    WorldModelConfig,
    load_yaml_config,
)
from .data import (
    ActionEmbeddingProvider,
    BioEmbedderProvider,
    PrecomputedProvider,
    build_provider,
)
from .models.world_model import WorldModel, build_world_model
from .training import WorldModelTrainer

__all__ = [
    "ActionEmbeddingConfig",
    "ActionEmbeddingProvider",
    "BioEmbedderProvider",
    "DataConfig",
    "DynamicsConfig",
    "EncoderConfig",
    "LossConfig",
    "OptimConfig",
    "PrecomputedProvider",
    "TrainConfig",
    "WorldModel",
    "WorldModelConfig",
    "WorldModelTrainer",
    "build_provider",
    "build_world_model",
    "configs",
    "data",
    "evaluation",
    "load_yaml_config",
    "models",
    "training",
    "utils",
]
