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

    dataset: str = ""
    """Human-readable dataset label used for run names and split-cache files.

    This is not a loader switch. The world model reads any perturbation
    AnnData file using the keys below.
    """

    h5ad_path: str = ""
    """Absolute path to the .h5ad file."""

    state_obsm_key: str = "X_state"
    """Key in ``adata.obsm`` containing per-cell state embeddings.

    World-model training consumes this matrix directly. Expression in
    ``adata.X`` is not transformed on the training path; any STATE,
    STACK, PCA, HVG, or other cell representation must already have
    been generated and attached to the AnnData before training starts.
    """

    perturbation_key: str = "perturbation"
    control_label: str = "non-targeting"

    stack_size: int = 4
    sequence_length: int = 8
    n_pert: int = 2

    context_mode: str = "trajectory"
    """How ``__getitem__`` assembles a sample.

    * ``"trajectory"`` (default, unchanged) -- a fabricated length-T
      chain ``control -> pertA -> pertB -> ...`` for the causal GPT
      dynamics. Keeps every existing run byte-identical.
    * ``"incontext_set"`` -- one *task* per sample: a permutation-free
      SET of ``incontext_support_size`` support triplets
      ``(s_control, a_i, s_pert_i)`` plus ONE query triplet
      ``(s_control, a_q, ?)`` whose perturbed state is the target. Used
      by the bidirectional in-context dynamics
      (``dynamics.kind = "incontext_set"``)."""

    incontext_support_size: int = 16
    """``context_mode="incontext_set"`` only -- number of support
    triplets per task. The query is one additional triplet."""

    incontext_support_strategy: str = "random"
    """``context_mode="incontext_set"`` only -- how support perturbations
    are selected after the query perturbation has been chosen.

    * ``"random"`` preserves the original behavior: draw support labels
      at random from the same context bucket while excluding the query
      perturbation whenever possible.
    * ``"action_similarity"`` keeps the same bucket/leakage constraints
      but ranks candidate support perturbations by cosine similarity in
      the support action-embedding table and takes the nearest labels.
      This tests whether in-context examples help when they are action
      relevant instead of merely bucket-matched."""

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    n_sequences_per_epoch: int | None = None
    """Override for dataset epoch length. ``None`` uses the dataset default."""

    sequence_bucket_key: str | None = None
    """``adata.obs`` context columns used to anchor samples.

    ``None`` disables bucketing. A single column name (``"batch"``) or
    a composite list (``"cell_type,batch"`` or ``"cell_type+batch"``)
    constrains each emitted sequence/task to that exact context
    neighborhood. ``"auto"`` picks common biological/technical columns
    when present, currently one cell-identity column and one
    batch/sample/donor column. This is especially important for
    in-context training because support triplets should describe the
    same biological/technical substrate as the query."""


@dataclass
class EncoderConfig:
    """Observation encoder hyperparameters."""

    kind: str = "transformer"
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class DynamicsConfig:
    """Latent dynamics model hyperparameters."""

    kind: str = "gpt"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 64
    use_action_token: bool = True
    latent_normalization: str = "none"
    """In-context only -- optional normalization of encoded state latents
    before dynamics/loss. One of ``{"none", "layer_norm", "l2"}``.
    ``"layer_norm"`` uses non-affine per-sample LayerNorm so the
    normalization itself cannot learn a new scale."""

    prediction_mode: str = "absolute"
    """In-context only -- what the dynamics head predicts.

    * ``"absolute"`` preserves the original behavior and predicts the
      perturbed latent state directly.
    * ``"residual_delta"`` predicts the perturbation effect
      ``delta_hat`` and forms ``s_hat = query_s + delta_hat``. The
      latent objective is then MSE against
      ``delta_target = s_target - query_s`` while still logging absolute
      ``latent_mse`` for comparability."""

    residual_output_init_scale: float = 0.01
    """In-context residual mode only -- multiplicative scale applied to
    the freshly initialized dynamics output head. A small value makes
    the initial residual close to zero without freezing gradients."""



@dataclass
class LossConfig:
    """Training loss weights and contrastive knobs."""

    latent_mse: float = 1.0
    decoder_mse: float = 0.5
    info_nce: float = 0.0
    info_nce_temperature: float = 0.1
    # Option 3: hard-negative mining. When True, the InfoNCE denominator
    # excludes (s_hat[i], s_target[j]) pairs where i and j share the same
    # perturbation -- the model is no longer punished for embedding two
    # cells under the same action close to each other.
    info_nce_mask_same_pert: bool = True
    # Option 1: counterfactual-action contrast. Runs a second dynamics
    # forward with action indices permuted across the batch dim and asks
    # "is my real-action prediction more similar to the target than the
    # counterfactual-action prediction is?". Forces the dynamics module
    # to actually use the action token. 0.0 disables; ~0.1 is a sane
    # starting weight.
    action_counterfactual: float = 0.1
    action_counterfactual_temperature: float = 0.1


@dataclass
class OptimConfig:
    """Optimizer and scheduler hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float | None = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 1000


@dataclass
class TrainConfig:
    """Training loop runtime options."""

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

    Training uses one run-time backend:

    * ``"anndata_obsm"`` -- read per-cell perturbation embeddings from
      ``adata.obsm`` and deduplicate them by perturbation label.

    CSV / NPZ tables and BioEmbedder models are still supported by
    ``world_model.scripts.embed_perturbations`` as offline attachment
    sources, but the model itself only reads AnnData ``.obsm``.
    """

    source: str = "anndata_obsm"
    """Must be ``"anndata_obsm"`` for training."""

    path: str = ""
    """Deprecated training input; use only with offline attachment scripts."""

    h5ad_path: str = ""
    """Deprecated override; training reads action embeddings from ``data.h5ad_path``."""

    obsm_key: str = "X_pert"
    """``anndata_obsm`` only -- key in ``adata.obsm`` with per-cell action vectors."""

    perturbation_key: str = ""
    """``anndata_obsm`` only -- optional obs column override.

    Empty means reuse ``data.perturbation_key``."""

    store_path: str = ""
    """Deprecated ``.emstore`` path kept so older YAMLs get a migration hint."""

    store_key: str = ""
    """Deprecated ``.emstore`` key kept so older YAMLs get a migration hint."""

    model_name: str = "esm2_650M"
    """Offline attachment metadata/model key; not used directly by training."""

    organism: str = "human"
    id_type: str = "symbol"
    """Offline attachment only -- one of ``{"symbol", "ensembl_id"}``."""

    region: str = "full"
    """Offline DNA attachment only -- one of ``{"full", "exons", "introns"}``."""

    pooling_strategy: str = "mean"

    resolver_backend: str = "api"
    """Offline attachment only -- one of ``{"api", "local"}``."""

    mart_file: str | None = None
    chromosome_folder: str | None = None
    device: str = "auto"

    cache_dir: str = "runs/_cache/action_embeddings"
    """Offline embedding cache root. Training does not consult it."""

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

    Training consumes already-attached state embeddings from
    ``adata.obsm[data.state_obsm_key]``. ``kind`` now selects the small
    train-time head placed on top of those embeddings:

    * ``"local"`` -- the existing :class:`StateStackEncoder` over the
      state embedding stack. Default.
    * ``"state"`` / ``"stack"`` -- a mean-pool projection head for
      foreign cell embeddings that were generated offline.

    The heavy STATE / STACK forward pass is not run by the dataloader.
    Use ``world_model.scripts.encode_cells`` or another offline step to
    create the AnnData ``.obsm`` matrix first.
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

    Useful for offline state-embedding attachment jobs that should only
    re-use cached encoder outputs instead of waiting on GPU encoding.
    Meaningful only for ``state`` / ``stack``; warned-against for ``local``.
    """


@dataclass
class WorldModelConfig:
    """Top-level resolved world-model configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    action_embedding: ActionEmbeddingConfig = field(default_factory=ActionEmbeddingConfig)
    query_action_embedding: ActionEmbeddingConfig = field(default_factory=lambda: ActionEmbeddingConfig(obsm_key=""))
    action_adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    state_backbone: StateBackboneConfig = field(default_factory=StateBackboneConfig)

    seed: int = 0
    run_name: str = "wm_run"
    output_dir: str = "runs/world_model"

    mode: str = "single"
    """Training mode. Only ``"single"`` is supported."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation."""
        return asdict(self)

    def validate(self) -> None:
        """Cross-field sanity checks. Called explicitly by training entrypoints."""
        if not self.data.state_obsm_key:
            raise ValueError("data.state_obsm_key is required; training reads states from adata.obsm.")
        support_strategy = getattr(self.data, "incontext_support_strategy", "random")
        if support_strategy not in {"random", "action_similarity"}:
            raise ValueError(
                "data.incontext_support_strategy must be one of "
                "{'random', 'action_similarity'}, got "
                f"{support_strategy!r}."
            )
        if self.action_embedding.source != "anndata_obsm":
            raise ValueError(
                "action_embedding.source must be 'anndata_obsm' for training. "
                "Generate or attach embeddings first with world_model.scripts.embed_perturbations."
            )
        if not self.action_embedding.obsm_key:
            raise ValueError("action_embedding.obsm_key is required; training reads actions from adata.obsm.")
        query_enabled = bool(self.query_action_embedding.obsm_key)
        if query_enabled:
            incontext_data = self.data.context_mode == "incontext_set"
            incontext_dynamics = self.dynamics.kind in {"incontext_set", "incontext_tokens"}
            if not (incontext_data and incontext_dynamics):
                raise ValueError(
                    "query_action_embedding is only valid for in-context training. "
                    "Set data.context_mode='incontext_set' and dynamics.kind to "
                    "'incontext_set' or 'incontext_tokens', or remove "
                    "query_action_embedding.obsm_key."
                )
            if self.query_action_embedding.source != "anndata_obsm":
                raise ValueError(
                    "query_action_embedding.source must be 'anndata_obsm'. "
                    "Generate or attach query/action embeddings to AnnData first."
                )
        if self.mode != "single":
            raise ValueError("Only mode='single' is supported.")
        latent_norm = getattr(self.dynamics, "latent_normalization", "none")
        if latent_norm not in {"none", "layer_norm", "l2"}:
            raise ValueError(
                "dynamics.latent_normalization must be one of "
                "{'none', 'layer_norm', 'l2'}, got "
                f"{latent_norm!r}."
            )
        prediction_mode = getattr(self.dynamics, "prediction_mode", "absolute")
        if prediction_mode not in {"absolute", "residual_delta"}:
            raise ValueError(
                "dynamics.prediction_mode must be one of "
                "{'absolute', 'residual_delta'}, got "
                f"{prediction_mode!r}."
            )
        if float(getattr(self.dynamics, "residual_output_init_scale", 0.01)) < 0.0:
            raise ValueError("dynamics.residual_output_init_scale must be non-negative.")
        sb = self.state_backbone
        if sb.kind not in {"local", "state", "stack"}:
            raise ValueError(f"state_backbone.kind must be one of {{'local','state','stack'}}, got {sb.kind!r}")
        if sb.kind == "local" and sb.require_cache_hit:
            import logging

            logging.getLogger(__name__).warning(
                "state_backbone.require_cache_hit=True is ignored for kind='local' "
                "(no foundation cache is consulted on the local path)."
            )


_CONFIG_ROOT = Path(__file__).resolve().parent


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


def _deep_merge_mapping(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two YAML mappings without mutating either input."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mapping(current, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config mapping, resolving one-parent ``extends:`` chains."""
    if not _HAS_YAML:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    resolved = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        chain = " -> ".join(str(p) for p in [*seen, resolved])
        raise ValueError(f"Config extends cycle detected: {chain}")
    seen.add(resolved)
    with open(resolved) as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config root must be a mapping, got {type(raw).__name__}: {path}")
    parent_spec = raw.pop("extends", None)
    if parent_spec is None:
        return raw
    parent = _resolve_config_parent(resolved, parent_spec)
    parent_raw = _load_yaml_mapping(parent, seen=seen)
    return _deep_merge_mapping(parent_raw, raw)


def _resolve_config_parent(path: Path, parent_spec: Any) -> Path:
    if not isinstance(parent_spec, str) or not parent_spec:
        raise ValueError(f"Config 'extends' in {path} must be a non-empty string.")
    parent = Path(parent_spec).expanduser()
    if parent.is_absolute():
        candidates = [parent]
    else:
        candidates = [path.parent / parent, _CONFIG_ROOT / parent]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Config parent for {path} not found. Tried: {tried}")


def load_yaml_config(path: str | Path) -> WorldModelConfig:
    """Load a :class:`WorldModelConfig` from YAML, resolving optional ``extends:``."""
    raw = _load_yaml_mapping(Path(path))
    cfg = WorldModelConfig()
    return _merge_into_dataclass(cfg, raw)


def validate_world_model_data_sources(cfg: WorldModelConfig) -> None:
    """Fail early when a local H5AD violates the world-model input contract."""
    _validate_h5ad_contract(
        cfg.data.h5ad_path,
        perturbation_key=cfg.data.perturbation_key,
        control_label=cfg.data.control_label,
        state_obsm_key=cfg.data.state_obsm_key,
        action_obsm_key=cfg.action_embedding.obsm_key,
        query_action_obsm_key=(cfg.query_action_embedding.obsm_key if cfg.query_action_embedding.obsm_key else None),
        stage="training data",
    )


def _validate_h5ad_contract(
    h5ad_path: str,
    *,
    perturbation_key: str,
    control_label: str,
    state_obsm_key: str,
    action_obsm_key: str,
    query_action_obsm_key: str | None = None,
    stage: str,
) -> None:
    if not h5ad_path:
        return
    path = Path(h5ad_path).expanduser()
    if not path.exists():
        return
    try:
        import anndata as ad  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            f"world-model config validation: anndata is required to inspect local H5AD files for {stage}."
        ) from exc
    try:
        adata = ad.read_h5ad(path, backed="r")
    except (OSError, ValueError) as exc:
        raise ValueError(
            "world-model config validation: failed to inspect local H5AD for "
            f"{stage}: {path}. Check that the file exists, is a valid .h5ad, "
            "and is not truncated."
        ) from exc
    try:
        if perturbation_key not in adata.obs.columns:
            raise ValueError(
                "world-model config validation: perturbation key "
                f"{perturbation_key!r} is missing from {stage} at {path}. "
                f"Available obs columns: {list(adata.obs.columns)}."
            )
        labels = adata.obs[perturbation_key].astype(str)
        if not labels.eq(str(control_label)).any():
            examples = labels.value_counts().head(10).index.tolist()
            raise ValueError(
                "world-model config validation: control label "
                f"{control_label!r} was not found in {stage} at {path} "
                f"under obs[{perturbation_key!r}]. Example labels: {examples}. "
                "Set data.control_label to the exact control string used by the dataset."
            )
        missing_obsm = [
            key for key in (state_obsm_key, action_obsm_key, query_action_obsm_key) if key and key not in adata.obsm
        ]
        if missing_obsm:
            raise ValueError(
                "world-model config validation: required AnnData obsm key(s) "
                f"{missing_obsm} are missing from {stage} at {path}. "
                f"Available obsm keys: {list(adata.obsm.keys())}. "
                "Training only consumes pre-attached state/action embeddings."
            )
    finally:
        if hasattr(adata, "file"):
            adata.file.close()


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
                raise KeyError(f"Unknown key {key!r} (no field {part!r} on {type(target).__name__})")
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
    "WorldModelConfig",
    "apply_cli_overrides",
    "load_yaml_config",
    "validate_world_model_data_sources",
]
