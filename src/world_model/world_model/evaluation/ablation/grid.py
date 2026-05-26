"""Action-encoder ablation grid: declarative specs + YAML loader.

A :class:`ActionEncoderSpec` is purely declarative -- it captures the
five knobs that disambiguate one run of the world model from another
when the *only* thing that varies is the action representation. The
runner translates each spec into an ``action_embedding`` config block
on a deep-copy of the user-supplied base ``WorldModelConfig``.

The default grid is small on purpose: one representative per modality
(DNA, protein, text) plus a few extra sizes that fit on a single GPU.
Override / extend it from a YAML file -- ``grid:`` mapping below the
top level.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionEncoderSpec:
    """One row of the ablation grid.

    The ``key`` is a short, file-system-safe handle used as the per-run
    output sub-directory and as the column key in the summary CSVs.
    Every other field maps 1:1 onto an ``action_embedding`` knob.
    """

    key: str
    model_name: str
    id_type: str = "symbol"
    region: str = "full"
    pooling: str = "mean"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "key": self.key,
            "model_name": self.model_name,
            "id_type": self.id_type,
            "region": self.region,
            "pooling": self.pooling,
            "extra_kwargs": dict(self.extra_kwargs),
            "notes": self.notes,
        }


# Order matters: the runner walks the list in order so the first
# success is the cheapest sanity check (text -> protein -> DNA).
DEFAULT_ACTION_ENCODER_GRID: list[ActionEncoderSpec] = [
    ActionEncoderSpec(
        key="borzoi",
        model_name="borzoi_v0",
        id_type="symbol",
        region="full",
        pooling="mean",
        notes="DNA, replicate 0",
    ),
    ActionEncoderSpec(
        key="enformer",
        model_name="enformer_human_rough",
        id_type="symbol",
        region="full",
        pooling="mean",
        notes="DNA, rough",
    ),
    ActionEncoderSpec(
        key="nt_v2_500m",
        model_name="nt_v2_500m",
        id_type="symbol",
        region="full",
        pooling="mean",
        notes="DNA, multi-species",
    ),
    ActionEncoderSpec(
        key="esm2_650m",
        model_name="esm2_650M",
        id_type="symbol",
        region="full",
        pooling="mean",
        notes="Protein",
    ),
    ActionEncoderSpec(
        key="minilm",
        model_name="minilm_l6_v2",
        id_type="symbol",
        region="full",
        pooling="mean",
        notes="Text baseline",
    ),
]


def load_grid(path: str | Path | None) -> list[ActionEncoderSpec]:
    """Load specs from a YAML file. Returns the default grid when ``path`` is None."""
    if path is None:
        return list(DEFAULT_ACTION_ENCODER_GRID)
    raw = _load_grid_yaml(path, kind="Ablation")
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = raw.get("grid", [])
    else:
        raise ValueError(f"Grid YAML must be a list or a mapping with 'grid:'; got {type(raw).__name__}.")
    if not isinstance(rows, list):
        raise ValueError("'grid' must be a list of mappings.")
    specs: list[ActionEncoderSpec] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Grid entry {i} must be a mapping; got {type(row).__name__}.")
        try:
            specs.append(_spec_from_mapping(row))
        except KeyError as exc:
            raise ValueError(f"Grid entry {i} missing required field: {exc}") from exc
    _check_unique_keys(specs)
    return specs


def _spec_from_mapping(row: dict[str, Any]) -> ActionEncoderSpec:
    if "key" not in row or "model_name" not in row:
        raise KeyError("'key' and 'model_name' are required.")
    return ActionEncoderSpec(
        key=str(row["key"]),
        model_name=str(row["model_name"]),
        id_type=str(row.get("id_type", "symbol")),
        region=str(row.get("region", "full")),
        pooling=str(row.get("pooling", "mean")),
        extra_kwargs=dict(row.get("extra_kwargs", {})),
        notes=str(row.get("notes", "")),
    )


def _check_unique_keys(specs: Iterable[ActionEncoderSpec]) -> None:
    seen: dict[str, int] = {}
    for s in specs:
        seen[s.key] = seen.get(s.key, 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    if dups:
        raise ValueError(f"Duplicate grid keys: {dups}. Each spec.key must be unique.")


def filter_grid(
    specs: list[ActionEncoderSpec],
    *,
    only: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
) -> list[ActionEncoderSpec]:
    """Apply ``--only`` / ``--skip`` filters by ``spec.key``.

    Both filters accept either an iterable of strings or a comma-separated
    string. ``only`` is applied before ``skip``.
    """
    only_set = _parse_filter(only)
    skip_set = _parse_filter(skip)
    if only_set is not None:
        unknown = only_set - {s.key for s in specs}
        if unknown:
            raise KeyError(f"--only references unknown grid keys: {sorted(unknown)}")
        specs = [s for s in specs if s.key in only_set]
    if skip_set:
        specs = [s for s in specs if s.key not in skip_set]
    if not specs:
        raise ValueError("Filter produced an empty grid; nothing to run.")
    return specs


def _parse_filter(value: Iterable[str] | str | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {p.strip() for p in value.split(",") if p.strip()}
    return {str(p).strip() for p in value if str(p).strip()}


def resolve_grid(
    *,
    grid_path: str | Path | None,
    only: Iterable[str] | str | None = None,
    skip: Iterable[str] | str | None = None,
) -> list[ActionEncoderSpec]:
    """Convenience wrapper: load + filter in one call."""
    specs = load_grid(grid_path)
    _check_unique_keys(specs)
    return filter_grid(specs, only=only, skip=skip)


# ----------------------------------------------------------------------
# Adapter ablation grid (Phase 3).
#
# Decoupled from the encoder grid so the two sweeps share machinery
# (filter / resolve / dup-key checks) but stay independently editable.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ActionAdapterSpec:
    """One row of the adapter ablation grid.

    Maps onto :class:`ActionAdapterConfig` overrides; everything not
    listed here keeps the base run's default. ``key`` is the per-run
    output sub-directory and the column key in summary CSVs.
    """

    key: str
    kind: str
    hidden_dim: int = 512
    dropout: float = 0.0
    activation: str = "gelu"
    lora_rank: int = 0
    lora_alpha: float = 1.0
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "key": self.key,
            "kind": self.kind,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "notes": self.notes,
        }


DEFAULT_ACTION_ADAPTER_GRID: list[ActionAdapterSpec] = [
    ActionAdapterSpec(key="linear", kind="linear", notes="Phase-1/2 baseline."),
    ActionAdapterSpec(key="mlp_h256", kind="mlp", hidden_dim=256, dropout=0.1),
    ActionAdapterSpec(key="mlp_h1024", kind="mlp", hidden_dim=1024, dropout=0.1),
    ActionAdapterSpec(key="lora_r4", kind="lora", lora_rank=4),
    ActionAdapterSpec(key="lora_r16", kind="lora", lora_rank=16),
    ActionAdapterSpec(key="lora_r64", kind="lora", lora_rank=64),
]


def _adapter_spec_from_mapping(row: dict[str, Any]) -> ActionAdapterSpec:
    if "key" not in row or "kind" not in row:
        raise KeyError("'key' and 'kind' are required.")
    return ActionAdapterSpec(
        key=str(row["key"]),
        kind=str(row["kind"]),
        hidden_dim=int(row.get("hidden_dim", 512)),
        dropout=float(row.get("dropout", 0.0)),
        activation=str(row.get("activation", "gelu")),
        lora_rank=int(row.get("lora_rank", 0)),
        lora_alpha=float(row.get("lora_alpha", 1.0)),
        notes=str(row.get("notes", "")),
    )


def load_adapter_grid(path: str | Path | None) -> list[ActionAdapterSpec]:
    """Load adapter specs from YAML, or return the default adapter grid."""
    if path is None:
        return list(DEFAULT_ACTION_ADAPTER_GRID)
    raw = _load_grid_yaml(path, kind="Adapter ablation")
    rows: list[Any]
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = raw.get("grid", [])
    else:
        raise ValueError(
            f"Grid YAML must be a list or a mapping with 'grid:'; got {type(raw).__name__}."
        )
    if not isinstance(rows, list):
        raise ValueError("'grid' must be a list of mappings.")
    specs: list[ActionAdapterSpec] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Grid entry {i} must be a mapping; got {type(row).__name__}.")
        try:
            specs.append(_adapter_spec_from_mapping(row))
        except KeyError as exc:
            raise ValueError(f"Adapter grid entry {i} missing required field: {exc}") from exc
    _check_unique_adapter_keys(specs)
    return specs


def _load_grid_yaml(path: str | Path, *, kind: str) -> Any:
    """Load a grid YAML file with a tiny ``extends:`` compatibility hook."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{kind} grid file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if isinstance(raw, dict) and "extends" in raw:
        if set(raw) - {"extends"}:
            raise ValueError(
                f"{kind} grid alias {p} may only contain 'extends'; "
                f"found extra keys {sorted(set(raw) - {'extends'})}."
            )
        parent = _resolve_grid_parent(p, raw["extends"])
        return _load_grid_yaml(parent, kind=kind)
    return raw


def _resolve_grid_parent(path: Path, parent_spec: Any) -> Path:
    if not isinstance(parent_spec, str) or not parent_spec:
        raise ValueError(f"Grid 'extends' in {path} must be a non-empty string.")
    parent = Path(parent_spec)
    config_root = Path(__file__).parents[2] / "configs"
    candidates = [parent] if parent.is_absolute() else [path.parent / parent, config_root / parent]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Grid parent for {path} not found. Tried: {tried}")


def _check_unique_adapter_keys(specs: Iterable[ActionAdapterSpec]) -> None:
    seen: dict[str, int] = {}
    for s in specs:
        seen[s.key] = seen.get(s.key, 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    if dups:
        raise ValueError(f"Duplicate adapter grid keys: {dups}.")


def filter_adapter_grid(
    specs: list[ActionAdapterSpec],
    *,
    only: Iterable[str] | str | None = None,
    skip: Iterable[str] | str | None = None,
) -> list[ActionAdapterSpec]:
    """Apply ``--only`` / ``--skip`` filters by adapter key."""
    only_set = _parse_filter(only)
    skip_set = _parse_filter(skip)
    if only_set is not None:
        unknown = only_set - {s.key for s in specs}
        if unknown:
            raise KeyError(f"--only references unknown adapter grid keys: {sorted(unknown)}")
        specs = [s for s in specs if s.key in only_set]
    if skip_set:
        specs = [s for s in specs if s.key not in skip_set]
    if not specs:
        raise ValueError("Filter produced an empty adapter grid; nothing to run.")
    return specs


def resolve_adapter_grid(
    *,
    grid_path: str | Path | None,
    only: Iterable[str] | str | None = None,
    skip: Iterable[str] | str | None = None,
) -> list[ActionAdapterSpec]:
    """Load and filter the adapter ablation grid in one call."""
    specs = load_adapter_grid(grid_path)
    _check_unique_adapter_keys(specs)
    return filter_adapter_grid(specs, only=only, skip=skip)


__all__ = [
    "DEFAULT_ACTION_ADAPTER_GRID",
    "DEFAULT_ACTION_ENCODER_GRID",
    "ActionAdapterSpec",
    "ActionEncoderSpec",
    "filter_adapter_grid",
    "filter_grid",
    "load_adapter_grid",
    "load_grid",
    "resolve_adapter_grid",
    "resolve_grid",
]
