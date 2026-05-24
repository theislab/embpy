"""Pure exporters from :class:`EmbeddingResult` to the two output backends.

These are free functions, not methods, on purpose: the result object
stays format-agnostic and every format concern lives here. Each exporter
takes an :class:`EmbeddingResult` and returns the backend object
(``pd.DataFrame`` / ``AnnData``), optionally writing to disk.

Layout of a tabular export (rows = entities, columns = dimensions):

    index            <id_scheme>     -- canonical key, e.g. ENSG...
    columns          [<alias cols>]  -- optional display cross-refs
                     dim_0 ... dim_{d-1}

Provenance never fits cleanly inside parquet/csv, so it is written to a
``<path>.meta.json`` sidecar next to the data file.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .result import EmbeddingResult

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

TableFormat = Literal["parquet", "csv"]
OutputFormat = Literal["anndata", "table"]

_TABLE_SUFFIX_TO_FMT: dict[str, TableFormat] = {
    ".parquet": "parquet",
    ".csv": "csv",
}
_VAR_ENTITY_TYPES = {"gene", "protein"}
_OBS_ENTITY_TYPES = {"molecule", "sequence", "text", "cell", "perturbation"}
_UNS_ENTITY_TYPES = {"protein_isoform"}


def _alias_schemes(result: EmbeddingResult) -> list[str]:
    """Sorted union of alias schemes present across entities (deterministic)."""
    if not result.aliases:
        return []
    schemes: set[str] = set()
    for mapping in result.aliases.values():
        schemes.update(mapping.keys())
    return sorted(schemes)


def _build_frame(result: EmbeddingResult) -> pd.DataFrame:
    """Assemble the canonical DataFrame: index=id_scheme, alias cols, dim cols."""
    index = pd.Index(result.entity_ids, name=result.id_scheme)

    # Display cross-refs first (so a human scanning the file sees
    # symbol / uniprot / name next to the canonical id), then dims.
    alias_data: dict[str, list[str | None]] = {}
    for scheme in _alias_schemes(result):
        alias_data[scheme] = [
            (result.aliases.get(eid, {}) or {}).get(scheme)  # type: ignore[union-attr]
            for eid in result.entity_ids
        ]
    alias_frame = pd.DataFrame(alias_data, index=index) if alias_data else None

    dim_frame = pd.DataFrame(
        result.matrix,
        index=index,
        columns=result.dim_names,
    )
    if alias_frame is None:
        return dim_frame
    return pd.concat([alias_frame, dim_frame], axis=1)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _write_sidecar(path: Path, result: EmbeddingResult, *, key: str | None = None) -> Path:
    meta = {
        "key": key or _default_key(result),
        "entity_type": result.entity_type,
        "id_scheme": result.id_scheme,
        "n_entities": result.n_entities,
        "n_dims": result.n_dims,
        "alias_schemes": _alias_schemes(result),
        "provenance": result.provenance.to_dict(),
    }
    side = _sidecar_path(path)
    side.write_text(json.dumps(meta, indent=2))
    return side


def to_table(
    result: EmbeddingResult,
    *,
    path: str | Path | None = None,
    fmt: TableFormat = "parquet",
) -> pd.DataFrame:
    """Export to a tabular DataFrame, optionally writing it to ``path``.

    Parameters
    ----------
    result
        The canonical embedding.
    path
        If given, write the frame to this file and a
        ``<path>.meta.json`` provenance sidecar. The returned frame is
        identical whether or not ``path`` is set.
    fmt
        ``"parquet"`` (default; pyarrow is already a dependency) or
        ``"csv"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``result.entity_ids`` with ``index.name =
        result.id_scheme``; columns are the optional alias cross-refs
        followed by ``dim_0 ... dim_{n_dims-1}``.
    """
    fmt = _validate_table_format(fmt)

    frame = _build_frame(result)

    if path is None:
        return frame

    out, fmt = _resolve_single_table_path(Path(path), result=result, fmt=fmt)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            frame.to_parquet(out, engine="pyarrow")
        else:
            frame.to_csv(out)
    except Exception as exc:
        raise ValueError(
            f"file writing: failed to write embedding table to {out}: {type(exc).__name__}: {exc}"
        ) from exc
    side = _write_sidecar(out, result)
    logger.info(
        "Wrote %s embedding table (%d x %d) to %s; provenance -> %s",
        result.entity_type,
        result.n_entities,
        result.n_dims,
        out,
        side.name,
    )
    return frame


def to_tables(
    results: Sequence[EmbeddingResult],
    *,
    path: str | Path | None = None,
    fmt: TableFormat = "parquet",
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Export one or more results to tabular outputs.

    A single result preserves :func:`to_table`'s return contract. Multiple
    results return ``{output_key: DataFrame}``. When writing multiple
    results, ``path`` must be an output directory (existing or not);
    filenames are derived from the deterministic output key.
    """
    items = _as_result_list(results)
    if len(items) == 1:
        return to_table(items[0], path=path, fmt=fmt)

    fmt = _validate_table_format(fmt)
    keyed = _keyed_results(items)
    frames = {key: _build_frame(result) for key, result in keyed.items()}
    if path is None:
        return frames

    out_dir = Path(path)
    if out_dir.suffix:
        raise ValueError(
            "output routing: multiple embedding results cannot be written to "
            f"a single file path {out_dir}. Pass an output directory instead."
        )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, result in keyed.items():
            out = out_dir / f"{key}.{fmt}"
            if fmt == "parquet":
                frames[key].to_parquet(out, engine="pyarrow")
            else:
                frames[key].to_csv(out)
            _write_sidecar(out, result, key=key)
    except Exception as exc:
        raise ValueError(
            f"file writing: failed to write embedding tables under {out_dir}: {type(exc).__name__}: {exc}"
        ) from exc
    return frames


def _validate_table_format(fmt: str) -> TableFormat:
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}.")
    return fmt  # type: ignore[return-value]


def _resolve_single_table_path(
    path: Path,
    *,
    result: EmbeddingResult,
    fmt: TableFormat,
) -> tuple[Path, TableFormat]:
    if path.exists() and path.is_dir():
        return path / f"{_default_key(result)}.{fmt}", fmt
    if not path.suffix:
        return path / f"{_default_key(result)}.{fmt}", fmt
    suffix = path.suffix.lower()
    if suffix not in _TABLE_SUFFIX_TO_FMT:
        raise ValueError(f"file writing: unsupported output suffix {suffix!r} for {path}. Expected .parquet or .csv.")
    return path, _TABLE_SUFFIX_TO_FMT[suffix]


def _default_key(result: EmbeddingResult) -> str:
    parts = ["X_emb", result.entity_type, result.provenance.model]
    if result.provenance.pooling:
        parts.append(f"pool_{result.provenance.pooling}")
    if result.provenance.layer is not None:
        parts.append(f"layer_{result.provenance.layer}")
    extra = dict(result.provenance.extra)
    for name in ("region", "isoform_mode"):
        if extra.get(name):
            parts.append(f"{name}_{extra[name]}")
    if result.provenance.harmonized_n_components is not None:
        parts.append(f"dim_{result.provenance.harmonized_n_components}")
    return _sanitize_key("__".join(str(p) for p in parts))


def _sanitize_key(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return clean or "X_emb"


def _keyed_results(results: Sequence[EmbeddingResult]) -> dict[str, EmbeddingResult]:
    keyed: dict[str, EmbeddingResult] = {}
    for result in results:
        base = _default_key(result)
        key = base
        i = 2
        while key in keyed:
            key = f"{base}__{i}"
            i += 1
        keyed[key] = result
    return keyed


def _uns_block(result: EmbeddingResult) -> dict:
    return {
        "entity_type": result.entity_type,
        "id_scheme": result.id_scheme,
        "n_entities": result.n_entities,
        "n_dims": result.n_dims,
        "provenance": result.provenance.to_dict(),
        "alias_schemes": _alias_schemes(result),
    }


def _standalone_anndata(result: EmbeddingResult) -> AnnData:
    return to_anndata_many([result])


def _standalone_anndata_many(
    results: Sequence[EmbeddingResult],
    *,
    keys: Sequence[str] | None = None,
) -> AnnData:
    from anndata import AnnData

    items = _as_result_list(results)
    obs_ids: list[str] = []
    var_ids: list[str] = []
    obs_name = "embpy_entity_id"
    var_name = "embpy_feature_id"

    for result in items:
        axis = _standalone_axis(result)
        if axis == "obs":
            obs_name = result.id_scheme
            _extend_unique(obs_ids, result.entity_ids)
        elif axis == "var":
            var_name = result.id_scheme
            _extend_unique(var_ids, result.entity_ids)

    if not obs_ids:
        obs_ids = ["embpy_placeholder_obs"]
        obs_name = "embpy_placeholder_obs"
    if not var_ids:
        var_ids = ["embpy_placeholder_feature"]
        var_name = "embpy_placeholder_feature"

    obs = pd.DataFrame(index=pd.Index(obs_ids, name=obs_name))
    var = pd.DataFrame(index=pd.Index(var_ids, name=var_name))
    adata = AnnData(X=csr_matrix((len(obs_ids), len(var_ids)), dtype=np.float32), obs=obs, var=var)
    _init_embpy_uns(adata, placeholder=True)

    keyed = dict(zip(keys, items, strict=True)) if keys is not None else _keyed_results(items)
    for key, result in keyed.items():
        _attach_to_standalone_axis(adata, result, key)

    if len(items) == 1:
        result = items[0]
        adata.uns["embpy"].update(_uns_block(result))
    logger.info(
        "Built standalone AnnData (%d obs x %d vars) carrying %d embedding result(s).",
        adata.n_obs,
        adata.n_vars,
        len(items),
    )
    return adata


def _init_embpy_uns(adata: AnnData, *, placeholder: bool) -> None:
    block = adata.uns.setdefault("embpy", {})
    block.setdefault("embeddings", {})
    if placeholder:
        block["placeholder_X"] = {
            "is_placeholder": True,
            "reason": (
                "Standalone AnnData created by embpy to carry embeddings; "
                ".X is sparse placeholder data and never contains generated embeddings."
            ),
            "format": "scipy.sparse.csr_matrix",
        }


def _extend_unique(out: list[str], values: Sequence[str]) -> None:
    seen = set(out)
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)


def _standalone_axis(result: EmbeddingResult) -> Literal["obs", "var", "uns"]:
    if result.entity_type in _UNS_ENTITY_TYPES:
        return "uns"
    if result.entity_type in _VAR_ENTITY_TYPES:
        return "var"
    if result.entity_type in _OBS_ENTITY_TYPES:
        return "obs"
    return "obs"


def _attach_to_standalone_axis(
    adata: AnnData,
    result: EmbeddingResult,
    key: str,
) -> None:
    axis = _standalone_axis(result)
    if axis == "uns":
        _store_uns_embedding(adata, result, key)
        return
    if axis == "obs":
        adata.obsm[key] = _reindex_matrix(result, list(adata.obs_names), "nan", "obs")
        _add_alias_columns(adata.obs, result)
    else:
        adata.varm[key] = _reindex_matrix(result, list(adata.var_names), "nan", "var")
        _add_alias_columns(adata.var, result)
    _record_embedding_uns(adata, result, key)


def _resolve_axis(
    result: EmbeddingResult,
    target: AnnData,
    attach_to: Literal["auto", "obs", "var"],
    min_overlap: float,
) -> Literal["obs", "var"]:
    """Pick obs vs var by id overlap; never guess silently."""
    ids = set(result.entity_ids)
    obs_names = list(target.obs_names)
    var_names = list(target.var_names)
    obs_hits = sum(1 for n in obs_names if n in ids)
    var_hits = sum(1 for n in var_names if n in ids)
    n = result.n_entities
    obs_frac = obs_hits / n if n else 0.0
    var_frac = var_hits / n if n else 0.0

    if attach_to == "obs":
        chosen = "obs"
    elif attach_to == "var":
        chosen = "var"
    else:  # auto
        obs_ok = obs_frac >= min_overlap
        var_ok = var_frac >= min_overlap
        if obs_ok and not var_ok:
            chosen = "obs"
        elif var_ok and not obs_ok:
            chosen = "var"
        else:
            raise ValueError(
                "Cannot auto-resolve attach axis for "
                f"{result.entity_type!r} [{result.id_scheme}]: "
                f"{obs_hits}/{n} entities match target.obs_names "
                f"({obs_frac:.0%}), {var_hits}/{n} match target.var_names "
                f"({var_frac:.0%}); threshold is {min_overlap:.0%}. "
                "Pass attach_to='obs'/'var' explicitly, or fix the ids."
            )
    chosen_hits = obs_hits if chosen == "obs" else var_hits
    logger.info(
        "Attaching %s embedding to .%sm (overlap %d/%d on %s axis).",
        result.entity_type,
        "obs" if chosen == "obs" else "var",
        chosen_hits,
        n,
        chosen,
    )
    return chosen


def _reindex_matrix(
    result: EmbeddingResult,
    axis_names: list[str],
    missing: Literal["error", "nan"],
    axis: str,
) -> np.ndarray:
    """Re-order rows of the matrix to match the target axis order."""
    pos = {eid: i for i, eid in enumerate(result.entity_ids)}
    missing_names = [n for n in axis_names if n not in pos]
    if missing_names and missing == "error":
        preview = missing_names[:10]
        raise ValueError(
            f"{len(missing_names)} target .{axis}_names have no embedding "
            f"(missing='error'). First missing: {preview}. "
            "Pass missing='nan' to NaN-fill, or fix the ids."
        )
    if missing_names:
        logger.info(
            "NaN-filling %d target .%s_names absent from the embedding.",
            len(missing_names),
            axis,
        )
    out = np.full((len(axis_names), result.n_dims), np.nan, dtype=np.float32)
    for i, n in enumerate(axis_names):
        j = pos.get(n)
        if j is not None:
            out[i] = result.matrix[j]
    return out


def _add_alias_columns(frame: pd.DataFrame, result: EmbeddingResult) -> None:
    if not result.aliases:
        return
    for scheme in _alias_schemes(result):
        col = scheme
        values = []
        for eid in frame.index:
            values.append((result.aliases.get(str(eid), {}) or {}).get(scheme))
        frame[col] = values


def _record_embedding_uns(adata: AnnData, result: EmbeddingResult, key: str) -> None:
    _init_embpy_uns(adata, placeholder=False)
    block = _uns_block(result)
    adata.uns["embpy"]["embeddings"][key] = block
    adata.uns["embpy"][key] = block


def _store_uns_embedding(adata: AnnData, result: EmbeddingResult, key: str) -> None:
    _init_embpy_uns(adata, placeholder=False)
    payload = {
        **_uns_block(result),
        "entity_ids": list(result.entity_ids),
        "matrix": np.asarray(result.matrix, dtype=np.float32),
        "aliases": {k: dict(v) for k, v in (result.aliases or {}).items()},
    }
    adata.uns[key] = payload
    adata.uns["embpy"]["embeddings"][key] = {
        **_uns_block(result),
        "storage": "uns",
    }
    adata.uns["embpy"][key] = adata.uns["embpy"]["embeddings"][key]


def _as_result_list(results: EmbeddingResult | Sequence[EmbeddingResult]) -> list[EmbeddingResult]:
    if isinstance(results, EmbeddingResult):
        return [results]
    items = list(results)
    if not items:
        raise ValueError("output routing: no embedding results to export.")
    for item in items:
        if not isinstance(item, EmbeddingResult):
            raise TypeError(f"output routing: expected EmbeddingResult objects, got {type(item).__name__}.")
    return items


def to_anndata(
    result: EmbeddingResult,
    *,
    target: AnnData | None = None,
    attach_to: Literal["auto", "obs", "var"] = "auto",
    key: str | None = None,
    missing: Literal["error", "nan"] = "error",
    min_overlap: float = 1.0,
) -> AnnData:
    """Export to AnnData -- standalone, or attached to a user's AnnData.

    Standalone (``target=None``): a new AnnData container used to carry
    embeddings in ``.obsm``, ``.varm`` or ``.uns`` when no user AnnData
    was provided. Its ``.X`` is sparse placeholder data only, never the
    generated embedding matrix.

    Attach (``target`` given): place the (re-indexed) matrix into
    ``target.obsm[key]`` or ``target.varm[key]`` depending on which axis
    the entity ids align to. ``attach_to="auto"`` decides by overlap and
    **raises** (never guesses) when both or neither axis clears
    ``min_overlap``. The matrix is re-indexed to the chosen axis order;
    entities missing from the target are an error (``missing="error"``)
    or NaN-filled (``missing="nan"``).
    """
    if target is None:
        return to_anndata_many([result], keys=[key] if key is not None else None)

    out_key = key or _default_key(result)
    if result.entity_type in _UNS_ENTITY_TYPES:
        _store_uns_embedding(target, result, out_key)
        return target

    axis = _resolve_axis(result, target, attach_to, min_overlap)
    if axis == "obs":
        mat = _reindex_matrix(result, list(target.obs_names), missing, "obs")
        target.obsm[out_key] = mat
        _add_alias_columns(target.obs, result)
    else:
        mat = _reindex_matrix(result, list(target.var_names), missing, "var")
        target.varm[out_key] = mat
        _add_alias_columns(target.var, result)
    _record_embedding_uns(target, result, out_key)
    logger.info(
        "Attached %s embedding into target.%sm[%r].",
        result.entity_type,
        axis,
        out_key,
    )
    return target


def to_anndata_many(
    results: EmbeddingResult | Sequence[EmbeddingResult],
    *,
    target: AnnData | None = None,
    attach_to: Literal["auto", "obs", "var"] = "auto",
    keys: Sequence[str] | None = None,
    missing: Literal["error", "nan"] = "error",
    min_overlap: float = 1.0,
) -> AnnData:
    """Export one or more results to AnnData without writing embeddings into ``.X``."""
    items = _as_result_list(results)
    if keys is not None and len(keys) != len(items):
        raise ValueError(f"output routing: keys length must match results length ({len(keys)} != {len(items)}).")
    if target is None:
        return _standalone_anndata_many(items, keys=keys)

    keyed = dict(zip(keys, items, strict=False)) if keys is not None else _keyed_results(items)
    for key, result in keyed.items():
        to_anndata(
            result,
            target=target,
            attach_to=attach_to,
            key=key,
            missing=missing,
            min_overlap=min_overlap,
        )
    return target


def route_output(
    result: EmbeddingResult | Sequence[EmbeddingResult],
    *,
    output: OutputFormat = "anndata",
    target: AnnData | None = None,
    attach_to: Literal["auto", "obs", "var"] = "auto",
    harmonize_dim: int | None = None,
    path: str | Path | None = None,
    fmt: TableFormat = "parquet",
    missing: Literal["error", "nan"] = "error",
    key: str | None = None,
    min_overlap: float = 1.0,
    random_state: int = 0,
) -> AnnData | pd.DataFrame | dict[str, pd.DataFrame]:
    """The single user-facing output contract shared by ``BioEmbedder.embed``.

    Optionally harmonizes, then dispatches to the chosen backend with
    loud, specific validation:

    * ``output="table"`` returns / writes a DataFrame and **ignores**
      ``target`` (warns if one was passed).
    * ``output="anndata"`` builds a standalone AnnData when ``target`` is
      ``None``, else attaches into ``target.obsm``/``.varm``.
    """
    if output not in ("anndata", "table"):
        raise ValueError(f"output must be 'anndata' or 'table', got {output!r}.")

    results = _as_result_list(result)
    if harmonize_dim is not None:
        from .harmonize import harmonize

        results = [harmonize(item, harmonize_dim, random_state=random_state) for item in results]

    if key is not None and len(results) > 1:
        raise ValueError(
            "output routing: a single key cannot name multiple embedding "
            "results. Omit key or pass one result at a time."
        )

    if output == "table":
        if target is not None:
            logger.warning("output='table' ignores the provided target AnnData; returning/writing a table instead.")
        return to_tables(results, path=path, fmt=fmt)

    keys = [key] if key is not None else None
    return to_anndata_many(
        results,
        target=target,
        attach_to=attach_to,
        keys=keys,
        missing=missing,
        min_overlap=min_overlap,
    )


__all__ = ["to_table", "to_tables", "to_anndata", "to_anndata_many", "route_output"]
