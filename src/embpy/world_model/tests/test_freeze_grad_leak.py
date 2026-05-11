"""Freeze / unfreeze gradient leakage tests.

Two contracts:

* With ``freeze=True``, the backbone's parameters have
  ``requires_grad=False`` and never appear in the optimizer. Running
  a forward + backward + step does not modify them.
* With ``freeze=False``, the backbone's parameters DO appear in the
  optimizer's param groups (via :func:`iter_trainable_params`).

We use the *local* backbone for the actual forward-backward (because
its parameters are real torch tensors registered inside the world
model). For the foreign backbone we exercise only the optimizer
plumbing, with a tiny fake provider whose parameters live outside the
world-model graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import torch

from embpy.world_model.configs import (
    LossConfig,
    OptimConfig,
    StateBackboneConfig,
    TrainConfig,
)
from embpy.world_model.models.encoders.backbones import StateBackboneProvider
from embpy.world_model.models.world_model import build_world_model
from embpy.world_model.training.trainer import (
    WorldModelTrainer,
    iter_trainable_params,
)


def _make_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    B, T, K, G = 2, 3, 2, 8
    return {
        "obs_stack": torch.randn(B, T, K, G, generator=g),
        "next_obs_stack": torch.randn(B, T, K, G, generator=g),
        "action_indices": torch.randint(0, 3, (B, T, 2), generator=g),
        "next_expression": torch.randn(B, T, G, generator=g),
    }


def _build_local_model(*, freeze: bool) -> torch.nn.Module:
    torch.manual_seed(0)
    gene_table = torch.randn(4, 16)
    cfg = StateBackboneConfig(kind="local", freeze=freeze)
    return build_world_model(
        n_genes=8,
        gene_embedding_table=gene_table,
        d_model=16,
        stack_size=2,
        encoder_layers=1,
        encoder_heads=2,
        dynamics_layers=1,
        dynamics_heads=2,
        dropout=0.0,
        max_sequence_length=8,
        use_action_token=True,
        decoder_hidden_dims=(16,),
        enable_decoder=True,
        state_backbone_cfg=cfg,
    )


def test_local_freeze_true_leaves_encoder_grads_none() -> None:
    model = _build_local_model(freeze=True)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2,
    )
    batch = _make_batch()
    loss, _ = model.loss(batch)
    optim.zero_grad()
    loss.backward()

    for p in model.encoder.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_local_freeze_false_emits_encoder_grads() -> None:
    model = _build_local_model(freeze=False)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2,
    )
    batch = _make_batch()
    loss, _ = model.loss(batch)
    optim.zero_grad()
    loss.backward()

    has_nonzero = any(
        p.grad is not None and torch.linalg.norm(p.grad) > 0
        for p in model.encoder.parameters()
    )
    assert has_nonzero


# ---------------------------------------------------------------------
# Foreign-backbone optimizer plumbing (via iter_trainable_params)
# ---------------------------------------------------------------------


class _FakeForeignProvider(StateBackboneProvider):
    name: str = "state"
    supports_decode: bool = False

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._params = [torch.nn.Parameter(torch.randn(4, 4))]

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, adata: Any, *, batch_size: int | None = None) -> np.ndarray:
        del adata, batch_size
        return np.zeros((1, self._dim), dtype=np.float32)

    def freeze(self) -> None:
        for p in self._params:
            p.requires_grad_(False)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return iter(self._params)

    def train_mode(self, flag: bool) -> None:
        del flag


def _build_foreign_model() -> torch.nn.Module:
    torch.manual_seed(0)
    gene_table = torch.randn(4, 16)
    cfg = StateBackboneConfig(kind="state", state_checkpoint="fake.ckpt", freeze=False)
    provider = _FakeForeignProvider(dim=16)
    model = build_world_model(
        n_genes=8,
        gene_embedding_table=gene_table,
        d_model=16,
        stack_size=2,
        encoder_layers=1,
        encoder_heads=2,
        dynamics_layers=1,
        dynamics_heads=2,
        dropout=0.0,
        max_sequence_length=8,
        use_action_token=True,
        enable_decoder=False,
        state_backbone_cfg=cfg,
        state_backbone_provider=provider,
        state_backbone_embedding_dim=provider.embedding_dim,
    )
    return model, cfg, provider  # type: ignore[return-value]


def test_iter_trainable_params_excludes_backbone_when_frozen() -> None:
    model, _, provider = _build_foreign_model()
    cfg_frozen = StateBackboneConfig(
        kind="state", state_checkpoint="fake.ckpt", freeze=True,
    )
    provider.freeze()
    params, counts = iter_trainable_params(model, provider, cfg_frozen)
    assert counts["backbone_trainable"] == 0
    assert not any("backbone." in name for name, _ in params)


def test_iter_trainable_params_includes_backbone_when_unfrozen() -> None:
    model, cfg, provider = _build_foreign_model()
    params, counts = iter_trainable_params(model, provider, cfg)
    assert counts["backbone_trainable"] > 0
    assert any(name.startswith("backbone.") for name, _ in params)


def test_trainer_logs_param_groups_and_optimizer_omits_frozen_backbone() -> None:
    model, _, provider = _build_foreign_model()
    cfg_frozen = StateBackboneConfig(
        kind="state", state_checkpoint="fake.ckpt", freeze=True,
    )
    provider.freeze()
    trainer = WorldModelTrainer(
        model=model,
        optim_cfg=OptimConfig(lr=1e-3),
        loss_cfg=LossConfig(),
        train_cfg=TrainConfig(n_epochs=1, device="cpu"),
        output_dir="outputs/_test_freeze_grad_leak",
        run_name="t",
        hooks=[],
        state_backbone_cfg=cfg_frozen,
        backbone_provider=provider,
    )
    flat = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    for p in provider.parameters():
        assert id(p) not in {id(q) for q in flat}
