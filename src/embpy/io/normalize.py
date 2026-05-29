"""Input normalization for the standardized embedding path.

This module owns the messy front door: lists, NumPy arrays, pandas
objects, AnnData axes and file paths are converted into one predictable
``NormalizedInput`` record before model inference starts.  It deliberately
does *not* canonicalize biological identifiers and it does not know about
embedding models.  Canonicalization stays in :mod:`embpy.io._canon`;
model inference stays in :mod:`embpy.embedder`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:  # AnnData is an embpy dependency, but keep import-time failures tidy.
    from anndata import AnnData
except ImportError:  # pragma: no cover - import guard for unusual envs
    AnnData = Any  # type: ignore[misc, assignment]


AnndataAxis = Literal["obs", "var"]


@dataclass(frozen=True, slots=True)
class NormalizedInput:
    """Identifiers plus input metadata, before biological canonicalization."""

    identifiers: tuple[str, ...]
    input_kind: str
    source: str | None = None
    id_column: str | None = None
    anndata_axis: AnndataAxis | None = None
    alias_columns: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_requested(self) -> int:
        """Number of identifiers requested by the user."""
        return len(self.identifiers)


def normalize_embedding_input(
    data: Any,
    *,
    entity_type: str | None = None,
    id_column: str | None = None,
    identifier_column: str | None = None,
    anndata_axis: AnndataAxis | None = None,
    obs_column: str | None = None,
    var_column: str | None = None,
    source: str | None = None,
) -> NormalizedInput:
    """Normalize accepted embedding inputs into ``NormalizedInput``.

    Parameters
    ----------
    data
        Sequence, NumPy array, pandas Series/DataFrame, AnnData, or a
        CSV/TSV/Parquet path.
    entity_type
        Optional hint used only for AnnData axis defaults. Genes and
        proteins default to ``var``; molecules, sequences and text default
        to ``obs``.
    id_column, identifier_column
        Explicit table identifier column. ``identifier_column`` is the
        public spelling; ``id_column`` is accepted as a shorter alias.
    anndata_axis, obs_column, var_column
        AnnData extraction controls.

    Returns
    -------
    NormalizedInput

    Raises
    ------
    FileNotFoundError, ValueError, TypeError
        Messages are prefixed with ``input loading`` or
        ``input normalization`` so callers can report the failing stage.
    """
    col = _coalesce_id_column(id_column, identifier_column)

    if isinstance(data, str | Path) and _looks_like_input_path(data):
        return _normalize_path(Path(data), id_column=col, source=source)

    if _is_anndata(data):
        return _normalize_anndata(
            data,
            entity_type=entity_type,
            id_column=col,
            anndata_axis=anndata_axis,
            obs_column=obs_column,
            var_column=var_column,
            source=source,
        )

    if isinstance(data, pd.DataFrame):
        return _normalize_dataframe(data, id_column=col, source=source or "dataframe")

    if isinstance(data, pd.Series):
        ids = _coerce_identifier_values(data.tolist(), context="pandas Series")
        return NormalizedInput(
            identifiers=tuple(ids),
            input_kind="pandas_series",
            source=source,
            id_column=data.name if data.name is not None else None,
        )

    if isinstance(data, np.ndarray):
        ids = _normalize_numpy_array(data)
        return NormalizedInput(
            identifiers=tuple(ids),
            input_kind="numpy_array",
            source=source,
        )

    if isinstance(data, str):
        # A non-path string is a single identifier, not a sequence of chars.
        return NormalizedInput(
            identifiers=(data,),
            input_kind="scalar",
            source=source,
        )

    if isinstance(data, Sequence):
        ids = _coerce_identifier_values(list(data), context="sequence")
        return NormalizedInput(
            identifiers=tuple(ids),
            input_kind="sequence",
            source=source,
        )

    raise TypeError(
        "input normalization: unsupported input type "
        f"{type(data).__name__}. Expected a sequence of identifiers, NumPy "
        "array, pandas Series/DataFrame, AnnData, or CSV/TSV/Parquet path."
    )


def _coalesce_id_column(
    id_column: str | None,
    identifier_column: str | None,
) -> str | None:
    if id_column and identifier_column and id_column != identifier_column:
        raise ValueError(
            "input normalization: received both id_column="
            f"{id_column!r} and identifier_column={identifier_column!r}; "
            "pass only one identifier-column argument."
        )
    return identifier_column or id_column


def _looks_like_input_path(value: str | Path) -> bool:
    p = Path(value)
    if p.exists():
        return True
    return p.suffix.lower() in {".csv", ".tsv", ".parquet"}


def _normalize_path(
    path: Path,
    *,
    id_column: str | None,
    source: str | None,
) -> NormalizedInput:
    if not path.exists():
        raise FileNotFoundError(f"input loading: input path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input loading: input path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"input loading: unsupported input file suffix {suffix!r} for {path}. Expected .csv, .tsv, or .parquet."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"input loading: failed to read {path}: {type(exc).__name__}: {exc}") from exc

    norm = _normalize_dataframe(df, id_column=id_column, source=str(path))
    return NormalizedInput(
        identifiers=norm.identifiers,
        input_kind=f"{suffix.lstrip('_').lstrip('.')}_path",
        source=source or str(path),
        id_column=norm.id_column,
        alias_columns=norm.alias_columns,
        metadata={**norm.metadata, "path": str(path), "suffix": suffix},
    )


def _normalize_dataframe(
    df: pd.DataFrame,
    *,
    id_column: str | None,
    source: str | None,
) -> NormalizedInput:
    if df.empty and len(df.columns) == 0:
        raise ValueError("input normalization: table has no columns.")

    cols = [str(c) for c in df.columns]
    table = df.copy()
    table.columns = cols

    if id_column is None:
        if len(cols) == 1:
            id_column = cols[0]
        else:
            raise ValueError(
                "input normalization: ambiguous table identifier column. "
                f"Found columns {cols}; more than one column could contain "
                "identifiers, so inference is ambiguous. Pass "
                "identifier_column='<column>' (or id_column='<column>')."
            )
    if id_column not in table.columns:
        raise ValueError(
            f"input normalization: identifier column {id_column!r} was not found. Available columns: {cols}."
        )

    ids = _coerce_identifier_values(
        table[id_column].tolist(),
        context=f"table column {id_column!r}",
    )
    aliases = {c: tuple(_coerce_alias_values(table[c].tolist())) for c in table.columns if c != id_column}
    return NormalizedInput(
        identifiers=tuple(ids),
        input_kind="dataframe",
        source=source,
        id_column=id_column,
        alias_columns=aliases,
        metadata={"columns": cols},
    )


def _normalize_numpy_array(arr: np.ndarray) -> list[str]:
    a = np.asarray(arr)
    if a.ndim == 0:
        raise ValueError(
            "input normalization: NumPy array must be 1D string/object data "
            "or a 2D single-column array; got scalar shape ()."
        )
    if a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)
    elif a.ndim != 1:
        raise ValueError(
            "input normalization: NumPy array must be 1D string/object data "
            "or a 2D single-column array; got shape "
            f"{a.shape!r}."
        )

    if not (np.issubdtype(a.dtype, np.str_) or np.issubdtype(a.dtype, np.bytes_) or a.dtype == object):
        raise ValueError(
            "input normalization: NumPy array identifiers must have string, "
            f"bytes, or object dtype; got dtype {a.dtype}."
        )
    return _coerce_identifier_values(a.tolist(), context="NumPy array")


def _is_anndata(value: Any) -> bool:
    try:
        from anndata import AnnData as _AnnData
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, _AnnData)


def _normalize_anndata(
    adata: AnnData,
    *,
    entity_type: str | None,
    id_column: str | None,
    anndata_axis: AnndataAxis | None,
    obs_column: str | None,
    var_column: str | None,
    source: str | None,
) -> NormalizedInput:
    explicit_cols = [x is not None for x in (obs_column, var_column)]
    if sum(explicit_cols) > 1:
        raise ValueError(
            "input normalization: pass only one AnnData identifier source; received both obs_column and var_column."
        )
    if id_column is not None and (obs_column is not None or var_column is not None):
        raise ValueError(
            "input normalization: pass either identifier_column/id_column or "
            "obs_column/var_column for AnnData, not both."
        )

    axis = anndata_axis
    column = None
    if obs_column is not None:
        axis, column = "obs", obs_column
    elif var_column is not None:
        axis, column = "var", var_column
    elif id_column is not None:
        in_obs = id_column in adata.obs.columns
        in_var = id_column in adata.var.columns
        if in_obs and not in_var:
            axis, column = "obs", id_column
        elif in_var and not in_obs:
            axis, column = "var", id_column
        elif in_obs and in_var:
            raise ValueError(
                "input normalization: AnnData identifier column "
                f"{id_column!r} exists in both .obs and .var. Pass "
                "anndata_axis='obs' or anndata_axis='var', or use "
                "obs_column/var_column."
            )
        else:
            raise ValueError(
                "input normalization: AnnData identifier column "
                f"{id_column!r} was not found in .obs columns "
                f"{list(adata.obs.columns)} or .var columns {list(adata.var.columns)}."
            )

    if axis is None:
        axis = _default_anndata_axis(entity_type)

    if axis not in ("obs", "var"):
        raise ValueError(f"input normalization: anndata_axis must be 'obs' or 'var', got {axis!r}.")

    if column is None:
        values = list(adata.obs_names if axis == "obs" else adata.var_names)
        id_name = "obs_names" if axis == "obs" else "var_names"
    else:
        frame = adata.obs if axis == "obs" else adata.var
        if column not in frame.columns:
            raise ValueError(
                "input normalization: AnnData "
                f".{axis} column {column!r} was not found. Available "
                f".{axis} columns: {list(frame.columns)}."
            )
        values = frame[column].tolist()
        id_name = column

    ids = _coerce_identifier_values(values, context=f"AnnData .{axis} {id_name}")
    return NormalizedInput(
        identifiers=tuple(ids),
        input_kind="anndata",
        source=source or "anndata",
        id_column=id_name,
        anndata_axis=axis,
        metadata={
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "source_axis": axis,
            "source_column": column,
        },
    )


def _default_anndata_axis(entity_type: str | None) -> AnndataAxis:
    if entity_type in {"gene", "protein"}:
        return "var"
    if entity_type in {"molecule", "sequence", "text", "cell", "perturbation"}:
        return "obs"
    raise ValueError(
        "input normalization: AnnData has both .obs_names and .var_names. "
        "Pass anndata_axis='obs' or anndata_axis='var' (or obs_column/"
        "var_column) so embpy knows which identifiers to embed."
    )


def _coerce_identifier_values(values: Sequence[Any], *, context: str) -> list[str]:
    ids: list[str] = []
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(
                f"input normalization: identifiers cannot be missing; {context} has a missing value at position {i}."
            )
        text = str(value)
        if text == "" or text.lower() == "nan":
            raise ValueError(
                f"input normalization: identifiers cannot be empty; {context} has an empty value at position {i}."
            )
        ids.append(text)
    return ids


def _coerce_alias_values(values: Sequence[Any]) -> list[str | None]:
    out: list[str | None] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(None)
        else:
            text = str(value)
            out.append(None if text == "" or text.lower() == "nan" else text)
    return out


__all__ = ["AnndataAxis", "NormalizedInput", "normalize_embedding_input"]
