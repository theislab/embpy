"""Model components for the perturbation world model.

The package is split by role so that each component is independently
testable:

* :mod:`encoders`  -- maps stacks of gene-expression vectors to state tokens.
* :mod:`action`    -- maps perturbation labels to action tokens via gene
  embeddings.
* :mod:`dynamics`  -- causal transformer that predicts next-state tokens
  from past state and action tokens.
* :mod:`decoders`  -- optional projection back to gene-expression space.
* :mod:`world_model` -- composition class.
"""

from __future__ import annotations

from .action.gene_embedding_action import GeneEmbeddingAction
from .blocks import MLP, CausalSelfAttention, TransformerBlock
from .decoders.expression_decoder import ExpressionDecoder
from .dynamics.gpt_autoregressive import GPTAutoregressiveDynamics
from .encoders.state_stack_encoder import (
    MLPStateStackEncoder,
    StateStackEncoder,
    TransformerStateStackEncoder,
    build_state_stack_encoder,
)
from .world_model import WorldModel

__all__ = [
    "CausalSelfAttention",
    "ExpressionDecoder",
    "GPTAutoregressiveDynamics",
    "GeneEmbeddingAction",
    "MLP",
    "MLPStateStackEncoder",
    "StateStackEncoder",
    "TransformerBlock",
    "TransformerStateStackEncoder",
    "WorldModel",
    "build_state_stack_encoder",
]
