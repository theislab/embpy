"""Dataclass-based config schema for the world model package.

The schema is a tree of small dataclasses keyed by component:

    WorldModelConfig
      |- data:     DataConfig
      |- encoder:  EncoderConfig
      |- dynamics: DynamicsConfig
      |- loss:     LossConfig
      |- optim:    OptimConfig
      |- train:    TrainConfig
      |- seed: int
      |- run_name: str
      |- output_dir: str

Every component-level config maps 1:1 to a constructor in this package,
so swapping a value in YAML is sufficient to swap the underlying module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# yaml is a soft dep: it ships with anndata's runtime, but we still
# guard the import so importing the schema works in lean environments.
try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class DataConfig:
    """Where to find data and how to assemble training tuples."""

    dataset: str = "replogle"
    """One of ``{"replogle", "nadig"}``. Picks the adapter in :mod:`world_model.data.datasets`."""

    h5ad_path: str = ""
    """Absolute path to the .h5ad file."""

    gene_embedding_path: str = ""
    """CSV / NPZ table mapping gene_symbol -> embedding vector."""

    perturbation_key: str = "perturbation"
    """Column in ``adata.obs`` naming the perturbed gene."""

    control_label: str = "non-targeting"
    """Value of ``perturbation_key`` that flags control cells."""

    cell_type_key: str | None = "cell_type"
    """Optional column for context-aware control sampling. ``None`` disables it."""

    n_top_genes: int = 5000
    """Highly-variable-gene filter applied during preprocessing. Set <= 0 to disable."""

    log_normalize: bool = True
    """Apply log1p(library-size normalize) on the way in."""

    stack_size: int = 4
    """K -- number of past observations stacked into the state encoder."""

    sequence_length: int = 8
    """T -- length of the (state, action) sequence presented to the dynamics."""

    batch_size: int = 64
    val_fraction: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EncoderConfig:
    """Configuration for the state-stack encoder."""

    kind: str = "transformer"
    """One of ``{"transformer", "mlp"}`` -- selects the implementation."""

    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class DynamicsConfig:
    """Configuration for the autoregressive dynamics."""

    kind: str = "gpt"
    """Currently only ``"gpt"`` is implemented; here for forward compat."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 64
    """Upper bound on T used to size the positional embedding table."""

    use_action_token: bool = True
    """If False, dynamics reduces to a state-only causal transformer (ablation)."""


@dataclass
class LossConfig:
    """Weights of the multi-term training objective."""

    latent_mse: float = 1.0
    decoder_mse: float = 0.5
    info_nce: float = 0.0
    info_nce_temperature: float = 0.1


@dataclass
class OptimConfig:
    """AdamW + cosine schedule defaults."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float | None = 1.0
    scheduler: str = "cosine"
    """One of ``{"cosine", "constant", "linear_warmup"}``."""

    warmup_steps: int = 1000


@dataclass
class TrainConfig:
    """Training-loop specific knobs."""

    n_epochs: int = 50
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    device: str = "auto"
    """``"auto"`` resolves to cuda > mps > cpu."""

    amp: bool = True
    """Enable autocast + GradScaler on CUDA. Ignored on CPU/MPS."""


@dataclass
class WorldModelConfig:
    """Top-level config tying everything together."""

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    seed: int = 0
    run_name: str = "wm_run"
    output_dir: str = "outputs/world_model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_into_dataclass(dc: Any, overrides: dict[str, Any]) -> Any:
    """Recursively merge ``overrides`` into a dataclass instance."""
    if not is_dataclass(dc):
        return dc
    field_names = {f.name for f in fields(dc)}
    unknown = set(overrides) - field_names
    if unknown:
        raise KeyError(f"Unknown config keys for {type(dc).__name__}: {sorted(unknown)}")
    for f in fields(dc):
        if f.name not in overrides:
            continue
        cur = getattr(dc, f.name)
        new = overrides[f.name]
        if is_dataclass(cur) and isinstance(new, dict):
            _merge_into_dataclass(cur, new)
        else:
            setattr(dc, f.name, new)
    return dc


def load_yaml_config(path: str | Path) -> WorldModelConfig:
    """Load a :class:`WorldModelConfig` from a YAML file.

    Unspecified keys fall back to dataclass defaults. Unknown keys raise
    :class:`KeyError` so typos surface immediately.
    """
    if not _HAS_YAML:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    with open(path) as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config root must be a mapping, got {type(raw).__name__}")
    cfg = WorldModelConfig()
    return _merge_into_dataclass(cfg, raw)


__all__ = [
    "DataConfig",
    "DynamicsConfig",
    "EncoderConfig",
    "LossConfig",
    "OptimConfig",
    "TrainConfig",
    "WorldModelConfig",
    "load_yaml_config",
]
