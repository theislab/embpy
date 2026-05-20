"""Action adapters: trainable projections from foundation embeddings to dynamics tokens.

Foundation models (the BioEmbedder backends) stay frozen; the adapter is
the *only* learned bridge from their output dimension ``d_in`` to the
world model's token width ``d_model``.

Three families:

* :class:`LinearAdapter`  -- single ``nn.Linear``; baseline.
* :class:`MLPAdapter`     -- 2-layer MLP with optional dropout.
* :class:`LoRAAdapter`    -- frozen base linear + low-rank residual; the
                             foundation-style way of fine-tuning a
                             large->small projection.

All three share the :class:`ActionAdapter` base; build them through
:func:`build_action_adapter` so the dynamics module never has to care
which one is in use.
"""

from __future__ import annotations

from .base import ActionAdapter
from .factory import build_action_adapter
from .linear import LinearAdapter
from .lora import LoRAAdapter
from .mlp import MLPAdapter

__all__ = [
    "ActionAdapter",
    "LinearAdapter",
    "LoRAAdapter",
    "MLPAdapter",
    "build_action_adapter",
]
