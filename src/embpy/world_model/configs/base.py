"""Dataclass-based config schema for the world model package.

The schema is a tree of small dataclasses keyed by component:

    WorldModelConfig
      |- data:      DataConfig
      |- encoder:   EncoderConfig
      |- dynamics:  DynamicsConfig
      |- loss:      LossConfig
      |- optim:     OptimConfig
      |- train:     TrainConfig
      |- split:     SplitConfig
      |- transfer:  TransferConfig
      |- eval:      EvalConfig
      |- seed: int
      |- run_name: str
      |- output_dir: str
      |- mode: str

YAML files override defaults; unknown keys raise KeyError so typos
surface immediately. Dotted CLI overrides (encoder.d_model=512) are
also supported via :func:`apply_cli_overrides`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class DataConfig:
    """Where to find data and how to assemble training tuples."""

    dataset: str = "replogle"
    """One of ``{"replogle", "nadig"}``."""

    h5ad_path: str = ""
    """Absolute path to the .h5ad file."""

    gene_embedding_path: str = ""
    """CSV / NPZ table mapping gene_symbol -> embedding vector."""

    perturbation_key: str = "perturbation"
    control_label: str = "non-targeting"
    cell_type_key: str | None = "cell_type"

    n_top_genes: int = 5000
    log_normalize: bool = True

    stack_size: int = 4
    sequence_length: int = 8
    n_pert: int = 2

    batch_size: int = 64
    val_fraction: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    n_sequences_per_epoch: int | None = None
    """Override for dataset epoch length. ``None`` uses the dataset default."""


@dataclass
class EncoderConfig:
    kind: str = "transformer"
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class DynamicsConfig:
    kind: str = "gpt"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 64
    use_action_token: bool = True


@dataclass
class LossConfig:
    latent_mse: float = 1.0
    decoder_mse: float = 0.5
    info_nce: float = 0.0
    info_nce_temperature: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float | None = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 1000


@dataclass
class TrainConfig:
    n_epochs: int = 50
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    device: str = "auto"
    amp: bool = True
    enable_tensorboard: bool = True
    enable_csv_log: bool = True


@dataclass
class SplitConfig:
    """Train/test split policy.

    The default splits by perturbation identity, which is the
    scientifically meaningful out-of-distribution setting. ``"cell"``
    is a configurable fallback for sanity checks.
    """

    split_by: str = "perturbation"
    """One of ``{"perturbation", "cell"}``."""

    train_fraction: float = 0.8
    """Fraction of held units (perturbations or cells) used for training."""

    seed: int = 0
    cache_path: str | None = None
    """Optional path under output_dir where the split is persisted as NPZ."""

    keep_control_in_test: bool = True
    """Whether the test split keeps access to control cells (needed by baselines)."""


@dataclass
class TransferConfig:
    """Pretrain on one dataset, fine-tune on a fraction of another."""

    enabled: bool = False

    pretrain_h5ad_path: str = ""
    pretrain_dataset: str = "nadig"
    pretrain_gene_embedding_path: str = ""
    pretrain_epochs: int = 30

    finetune_fraction: float = 0.1
    """Fraction of training perturbations (or cells) used for fine-tuning."""

    finetune_epochs: int = 20
    pretrain_checkpoint: str | None = None
    """If set, skip the pretrain phase and load this checkpoint."""

    freeze_encoder_during_finetune: bool = False
    freeze_dynamics_during_finetune: bool = False


@dataclass
class EvalConfig:
    """Knobs for the end-of-training evaluation pipeline."""

    n_control_samples_per_pert: int = 32
    """How many control cells to sample when predicting each perturbation."""

    use_cell_eval: bool = True
    """If True, attempt to import and run the ``cell_eval`` package."""

    deg_top_k: int = 50
    save_predictions: bool = True
    """Persist real and predicted AnnData under outputs/<run>/eval/."""


@dataclass
class WorldModelConfig:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    seed: int = 0
    run_name: str = "wm_run"
    output_dir: str = "outputs/world_model"

    mode: str = "single"
    """One of ``{"single", "transfer"}``."""

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
    """Load a :class:`WorldModelConfig` from a YAML file."""
    if not _HAS_YAML:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    with open(path) as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config root must be a mapping, got {type(raw).__name__}")
    cfg = WorldModelConfig()
    return _merge_into_dataclass(cfg, raw)


def apply_cli_overrides(cfg: WorldModelConfig, overrides: list[str]) -> WorldModelConfig:
    """Apply a list of dotted ``key.subkey=value`` overrides to ``cfg`` in place.

    Values are parsed with ``yaml.safe_load`` so YAML scalar conventions
    (``true``, ``null``, ``1.0e-3``, lists) all work. Unknown keys raise
    :class:`KeyError` to keep typo detection on par with YAML loading.
    """
    if not overrides:
        return cfg
    if not _HAS_YAML:
        raise ImportError("PyYAML is required for CLI overrides.")
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"CLI override must be 'key=value', got {spec!r}")
        key, raw_value = spec.split("=", 1)
        value = yaml.safe_load(raw_value)
        path = key.split(".")
        target: Any = cfg
        for part in path[:-1]:
            if not is_dataclass(target):
                raise KeyError(f"Cannot descend into non-dataclass at {part!r}")
            field_names = {f.name for f in fields(target)}
            if part not in field_names:
                raise KeyError(
                    f"Unknown key {key!r} (no field {part!r} on {type(target).__name__})"
                )
            target = getattr(target, part)
        last = path[-1]
        if not is_dataclass(target):
            raise KeyError(f"Cannot set {last!r} on non-dataclass {type(target).__name__}")
        field_names = {f.name for f in fields(target)}
        if last not in field_names:
            raise KeyError(f"Unknown key {key!r} (no field {last!r} on {type(target).__name__})")
        setattr(target, last, value)
    return cfg


__all__ = [
    "DataConfig",
    "DynamicsConfig",
    "EncoderConfig",
    "EvalConfig",
    "LossConfig",
    "OptimConfig",
    "SplitConfig",
    "TrainConfig",
    "TransferConfig",
    "WorldModelConfig",
    "apply_cli_overrides",
    "load_yaml_config",
]
