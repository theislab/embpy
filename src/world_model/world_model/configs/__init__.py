"""Config schemas for the world model.

We use plain :mod:`dataclasses` rather than Hydra/OmegaConf. The reasons:

1. Zero new dependencies. ``embpy`` already ships with a slim wheel.
2. IDE-friendly: dataclass fields autocomplete and type-check.
3. Trivial YAML interop via :func:`load_yaml_config` -- the user-supplied
   YAML is loaded with PyYAML (already pulled in transitively by
   anndata) and merged into the dataclass.
4. ``extends:`` supports one-parent YAML composition without adding
   Hydra/OmegaConf or hiding the final dataclass shape.
"""

from __future__ import annotations

from .base import (
    ActionAdapterConfig,
    ActionEmbeddingConfig,
    DataConfig,
    DynamicsConfig,
    EncoderConfig,
    EvalConfig,
    LossConfig,
    OptimConfig,
    SplitConfig,
    StateBackboneConfig,
    TrainConfig,
    WorldModelConfig,
    apply_cli_overrides,
    load_yaml_config,
    validate_world_model_data_sources,
)

__all__ = [
    "ActionAdapterConfig",
    "ActionEmbeddingConfig",
    "DataConfig",
    "DynamicsConfig",
    "EncoderConfig",
    "EvalConfig",
    "LossConfig",
    "OptimConfig",
    "SplitConfig",
    "StateBackboneConfig",
    "TrainConfig",
    "WorldModelConfig",
    "apply_cli_overrides",
    "load_yaml_config",
    "validate_world_model_data_sources",
]
