"""Action-encoder ablation utilities.

Two responsibilities, kept in two files for testability:

* :mod:`grid`      -- declarative spec dataclass and the default grid,
                      plus YAML loader / filter helpers (no model code).
* :mod:`aggregate` -- read per-run artifacts produced by
                      ``scripts/train.py`` and assemble the long / wide
                      summary CSVs and plots.

The runner itself lives at
``world_model.scripts.ablate_action_encoder`` so it sits next to
the other entry points.
"""

from __future__ import annotations

from .aggregate import aggregate_ablation, aggregate_adapter
from .grid import (
    DEFAULT_ACTION_ADAPTER_GRID,
    DEFAULT_ACTION_ENCODER_GRID,
    ActionAdapterSpec,
    ActionEncoderSpec,
    filter_adapter_grid,
    filter_grid,
    load_adapter_grid,
    load_grid,
    resolve_adapter_grid,
    resolve_grid,
)

__all__ = [
    "DEFAULT_ACTION_ADAPTER_GRID",
    "DEFAULT_ACTION_ENCODER_GRID",
    "ActionAdapterSpec",
    "ActionEncoderSpec",
    "aggregate_ablation",
    "aggregate_adapter",
    "filter_adapter_grid",
    "filter_grid",
    "load_adapter_grid",
    "load_grid",
    "resolve_adapter_grid",
    "resolve_grid",
]
