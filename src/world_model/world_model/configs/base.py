"""Dataclass-based config schema for the world model package.

The schema is a tree of small dataclasses keyed by component:

    WorldModelConfig
      |- data:              DataConfig
      |- encoder:           EncoderConfig
      |- dynamics:          DynamicsConfig
      |- loss:              LossConfig
      |- optim:             OptimConfig
      |- train:             TrainConfig
      |- split:             SplitConfig
      |- transfer:          TransferConfig
      |- eval:              EvalConfig
      |- action_embedding:  ActionEmbeddingConfig
      |- action_adapter:    ActionAdapterConfig
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
    """Pretrain on one dataset, fine-tune on a fraction of another.

    The new ``pretrain_action_encoder`` / ``finetune_action_encoder``
    fields let a single transfer run mix two different
    :class:`ActionEmbeddingConfig` blocks (one per phase). Resolution
    rule: when either field is ``None``, fall back to the run's
    top-level ``action_embedding``. ``swap_strategy`` then picks how the
    fine-tune model reuses the pretrained weights when the two providers
    disagree on dimension or identity.
    """

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

    pretrain_action_encoder: "ActionEmbeddingConfig | None" = None
    """If None, fall back to the run's top-level ``action_embedding``."""

    finetune_action_encoder: "ActionEmbeddingConfig | None" = None
    """If None, fall back to the run's top-level ``action_embedding``."""

    swap_strategy: str = "none"
    """One of ``{"none", "reset_adapter", "learn_alignment", "reset_all_action"}``."""

    alignment_epochs: int = 5
    """Only used when ``swap_strategy == "learn_alignment"``."""

    alignment_lr: float = 1e-3


@dataclass
class EvalConfig:
    """Knobs for the end-of-training evaluation pipeline."""

    n_control_samples_per_pert: int = 32
    """How many control cells to sample when predicting each perturbation."""

    use_cell_eval: bool = True
    """If True, attempt to import and run the ``cell_eval`` package."""

    deg_top_k: int = 50
    save_predictions: bool = True
    """Persist real and predicted AnnData under runs/<run>/eval/."""


@dataclass
class ActionEmbeddingConfig:
    """How to materialise the action (perturbation) embedding table.

    Two backends are supported:

    * ``"precomputed"`` -- read a CSV / NPZ from ``path``. If ``path``
      is empty, falls back to :attr:`DataConfig.gene_embedding_path`
      so the legacy YAMLs keep working unchanged.
    * ``"bio_embedder"`` -- delegate to :class:`embpy.embedder.BioEmbedder`.
      ``model_name`` must be a key in ``embpy.embedder.MODEL_REGISTRY``.
    """

    source: str = "precomputed"
    """One of ``{"precomputed", "bio_embedder"}``."""

    path: str = ""
    """``precomputed`` only -- CSV / NPZ. Empty means fall back to data.gene_embedding_path."""

    model_name: str = "esm2_650M"
    """``bio_embedder`` only -- key in :attr:`embpy.embedder.MODEL_REGISTRY`."""

    organism: str = "human"
    id_type: str = "symbol"
    """``bio_embedder`` only -- one of ``{"symbol", "ensembl_id"}``."""

    region: str = "full"
    """``bio_embedder`` (DNA only) -- one of ``{"full", "exons", "introns"}``."""

    pooling_strategy: str = "mean"

    resolver_backend: str = "api"
    """``bio_embedder`` only -- one of ``{"api", "local"}``."""

    mart_file: str | None = None
    chromosome_folder: str | None = None
    device: str = "auto"

    cache_dir: str = "runs/_cache/action_embeddings"
    """Disk cache root. Use ``""`` to disable caching (not recommended)."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Forwarded to :meth:`BioEmbedder.embed_genes_batch` (e.g. ``biotype``)."""

    fail_on_unresolved: bool = False
    """If True, raise after the run summary when any RESOLVED row is in
    fact UNRESOLVED. Useful for cluster jobs that must never silently
    zero-fill. Off by default to keep notebook / exploration loops
    forgiving."""

    control_extra_labels: list[str] = field(default_factory=list)
    """Extra dataset-specific labels to treat as control regardless of
    the default regexes. Useful when a study uses a sentinel like
    ``"GFP_only"`` for non-targeting guides."""

    control_patterns: list[str] = field(default_factory=list)
    """If non-empty, overrides the default control regex set. Use to
    tighten or replace the curated default patterns in
    :data:`embpy.resources.gene.control.DEFAULT_CONTROL_PATTERNS`."""

    control_strict: bool = False
    """If True, raise on labels that mix control + gene components."""

    control_sentinel_seed: int = 0
    """Seed for the deterministic CONTROL sentinel vector. Override only
    if you need orthogonal control tokens across datasets in the same
    run; the default is fixed so cache files compare cleanly across
    machines."""


@dataclass
class ActionAdapterConfig:
    """Adapter that maps the foundation embedding dim to the dynamics ``d_model``.

    ``kind="linear"`` reproduces the pre-Phase-3 single ``nn.Linear``
    projection byte-equivalently (regression-tested). ``"mlp"`` and
    ``"lora"`` add small trainable capacity on top of the still-frozen
    foundation model.
    """

    kind: str = "linear"
    """One of ``{"linear", "mlp", "lora"}``."""

    hidden_dim: int = 512
    """Hidden width for ``kind="mlp"``."""

    dropout: float = 0.0
    """Dropout for ``kind="mlp"`` (between layers) and ``"lora"`` (on the LoRA path)."""

    activation: str = "gelu"
    """One of ``{"gelu", "relu"}`` -- used by ``kind="mlp"`` only."""

    lora_rank: int = 0
    """Rank of the LoRA residual. ``0`` -> degenerates to W0 only (frozen baseline)."""

    lora_alpha: float = 1.0
    """LoRA scale: residual is multiplied by ``alpha / rank`` before being added to W0."""


@dataclass
class StateBackboneConfig:
    """Foundation backbone used as the state (observation) encoder.

    Three flavours via ``kind``:

    * ``"local"`` -- the existing :class:`StateStackEncoder` wrapped as
      a :class:`LocalBackbone`. Default. Byte-equivalent to the
      pre-Phase-5 path. No extra dependencies.
    * ``"state"`` -- STATE / SE-600M (Arc Institute) via
      :class:`StateEmbeddingWrapper`. Requires the ``arc-state``
      package. Cell embeddings are pre-computed and cached.
    * ``"stack"`` -- STACK (Arc Institute) via :class:`StackWrapper`.
      Requires the ``arc-stack`` package.

    ``freeze=True`` (default) keeps the foundation backbone in eval
    mode and out of the optimizer. Flip to ``False`` to fine-tune the
    backbone end-to-end (a learnable head is always included).
    """

    kind: str = "local"
    """One of ``{"local", "state", "stack"}``."""

    state_checkpoint: str = ""
    """Path to a STATE ``.ckpt`` (``kind="state"``). Optional if ``state_model_folder`` is set."""

    state_model_folder: str | None = None
    """Folder holding STATE checkpoint + ``protein_embeddings.pt`` (``kind="state"``)."""

    state_protein_embeddings: str | None = None
    """Override path to STATE protein embeddings ``.pt`` (``kind="state"``)."""

    state_config: str | None = None
    """Optional YAML override for STATE ``Inference`` config (``kind="state"``)."""

    stack_checkpoint: str = ""
    """Path to a STACK ``.ckpt`` (``kind="stack"``)."""

    stack_genelist: str = ""
    """Path to STACK pickled gene list (``kind="stack"``)."""

    stack_gene_name_col: str | None = None
    """Column in ``adata.var`` with gene symbols for STACK (``kind="stack"``)."""

    device: str = "auto"
    """``"auto"`` resolves to ``"cuda"`` when available else ``"cpu"``."""

    freeze: bool = True
    """Freeze backbone parameters. Default True."""

    batch_size: int = 64
    """Encode batch size passed to the underlying wrapper."""

    cache_dir: str = "runs/_cache/state_backbone"
    """Disk cache root for cell embeddings."""

    require_cache_hit: bool = False
    """If True and a cache miss happens, raise instead of running the backbone.

    Useful for cluster jobs that should only re-use a pre-warmed cache
    (e.g. evaluation-only runs that must not wait on GPU encoding).
    Meaningful only for ``state`` / ``stack``; warned-against for ``local``.
    """


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
    action_embedding: ActionEmbeddingConfig = field(default_factory=ActionEmbeddingConfig)
    action_adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    state_backbone: StateBackboneConfig = field(default_factory=StateBackboneConfig)

    seed: int = 0
    run_name: str = "wm_run"
    output_dir: str = "runs/world_model"

    mode: str = "single"
    """One of ``{"single", "transfer"}``."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Cross-field sanity checks. Called explicitly by training entrypoints."""
        sb = self.state_backbone
        if sb.kind not in {"local", "state", "stack"}:
            raise ValueError(
                f"state_backbone.kind must be one of {{'local','state','stack'}}, got {sb.kind!r}"
            )
        if sb.kind == "state" and not (sb.state_checkpoint or sb.state_model_folder):
            raise ValueError(
                "state_backbone.kind='state' requires either state_checkpoint "
                "or state_model_folder to be set."
            )
        if sb.kind == "stack" and not (sb.stack_checkpoint and sb.stack_genelist):
            raise ValueError(
                "state_backbone.kind='stack' requires both stack_checkpoint "
                "and stack_genelist to be set."
            )
        if sb.kind == "local" and sb.require_cache_hit:
            import logging

            logging.getLogger(__name__).warning(
                "state_backbone.require_cache_hit=True is ignored for kind='local' "
                "(no foundation cache is consulted on the local path)."
            )


# Fields whose declared default is ``None`` but whose YAML override may
# arrive as a dict. We materialise the dict into the dataclass type
# named here so the rest of the merge stays uniform.
_OPTIONAL_DATACLASS_FIELDS: dict[str, type] = {}


def _register_optional_dataclass_field(field_name: str, dc_type: type) -> None:
    _OPTIONAL_DATACLASS_FIELDS[field_name] = dc_type


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
        elif cur is None and isinstance(new, dict) and f.name in _OPTIONAL_DATACLASS_FIELDS:
            sub = _OPTIONAL_DATACLASS_FIELDS[f.name]()
            _merge_into_dataclass(sub, new)
            setattr(dc, f.name, sub)
        else:
            setattr(dc, f.name, new)
    return dc


_register_optional_dataclass_field("pretrain_action_encoder", ActionEmbeddingConfig)
_register_optional_dataclass_field("finetune_action_encoder", ActionEmbeddingConfig)


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
    "TransferConfig",
    "WorldModelConfig",
    "apply_cli_overrides",
    "load_yaml_config",
]
