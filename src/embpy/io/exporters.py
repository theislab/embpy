"""Pure exporters from :class:`EmbeddingResult` to output backends.

These are free functions, not methods, on purpose: the result object
stays format-agnostic and every format concern lives here. Each exporter
takes an :class:`EmbeddingResult` and returns the backend object
(``pd.DataFrame`` / ``AnnData`` / payload ``dict``), optionally writing
to disk.

Layout of a tabular export (rows = entities, columns = dimensions):

    index            <id_scheme>     -- canonical key, e.g. ENSG...
    columns          [<alias cols>]  -- optional display cross-refs
                     dim_0 ... dim_{d-1}

Provenance never fits cleanly inside flat CSV or NPZ files, so it is
also written to a ``<path>.meta.json`` sidecar next to the data file.
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

FileFormat = Literal["csv", "npz", "zarr"]
TableFormat = FileFormat
OutputFormat = Literal["anndata", "table", "payload"]
PayloadMetadataMode = Literal["minimal", "full"]

_FILE_SUFFIX_TO_FMT: dict[str, FileFormat] = {
    ".csv": "csv",
    ".npz": "npz",
    ".zarr": "zarr",
}
_VAR_ENTITY_TYPES = {"gene", "protein"}
_OBS_ENTITY_TYPES = {"molecule", "sequence", "text", "cell"}
_UNS_ENTITY_TYPES = {"protein_isoform", "perturbation"}


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


def _metadata_dict(result: EmbeddingResult, *, key: str | None = None) -> dict:
    return {
        "key": key or _default_key(result),
        "entity_type": result.entity_type,
        "id_scheme": result.id_scheme,
        "n_entities": result.n_entities,
        "n_dims": result.n_dims,
        "alias_schemes": _alias_schemes(result),
        "provenance": result.provenance.to_dict(),
    }


def _write_sidecar(path: Path, result: EmbeddingResult, *, key: str | None = None) -> Path:
    meta = _metadata_dict(result, key=key)
    side = _sidecar_path(path)
    side.write_text(json.dumps(meta, indent=2))
    return side


def to_table(
    result: EmbeddingResult,
    *,
    path: str | Path | None = None,
    fmt: FileFormat = "npz",
) -> pd.DataFrame:
    """Export to a DataFrame, optionally writing a scverse-friendly file.

    Parameters
    ----------
    result
        The canonical embedding.
    path
        If given, write to this file and a ``<path>.meta.json``
        provenance sidecar. ``.npz`` is the default compact matrix
        format, ``.csv`` is human-readable, and ``.zarr`` writes a
        directory with the matrix in a Zarr array and metadata in attrs.
        The returned frame is identical whether or not ``path`` is set.
    fmt
        ``"npz"`` (default), ``"csv"``, or ``"zarr"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``result.entity_ids`` with ``index.name =
        result.id_scheme``; columns are the optional alias cross-refs
        followed by ``dim_0 ... dim_{n_dims-1}``.
    """
    fmt = _validate_file_format(fmt)

    frame = _build_frame(result)

    if path is None:
        return frame

    out, fmt = _resolve_single_output_path(Path(path), result=result, fmt=fmt)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_embedding_file(out, result=result, frame=frame, fmt=fmt)
    except Exception as exc:
        raise ValueError(
            f"file writing: failed to write embedding output to {out}: {type(exc).__name__}: {exc}"
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
    fmt: FileFormat = "npz",
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

    fmt = _validate_file_format(fmt)
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
            _write_embedding_file(out, result=result, frame=frames[key], fmt=fmt, key=key)
            _write_sidecar(out, result, key=key)
    except Exception as exc:
        raise ValueError(
            f"file writing: failed to write embedding tables under {out_dir}: {type(exc).__name__}: {exc}"
        ) from exc
    return frames


def _validate_file_format(fmt: str) -> FileFormat:
    if fmt not in ("csv", "npz", "zarr"):
        raise ValueError(f"fmt must be 'csv', 'npz', or 'zarr', got {fmt!r}.")
    return fmt  # type: ignore[return-value]


def _resolve_single_output_path(
    path: Path,
    *,
    result: EmbeddingResult,
    fmt: FileFormat,
) -> tuple[Path, FileFormat]:
    suffix = path.suffix.lower()
    if suffix in _FILE_SUFFIX_TO_FMT:
        return path, _FILE_SUFFIX_TO_FMT[suffix]
    if path.exists() and path.is_dir():
        return path / f"{_default_key(result)}.{fmt}", fmt
    if not path.suffix:
        return path / f"{_default_key(result)}.{fmt}", fmt
    raise ValueError(f"file writing: unsupported output suffix {suffix!r} for {path}. Expected .csv, .npz, or .zarr.")


def _write_embedding_file(
    path: Path,
    *,
    result: EmbeddingResult,
    frame: pd.DataFrame,
    fmt: FileFormat,
    key: str | None = None,
) -> None:
    if fmt == "csv":
        frame.to_csv(path)
    elif fmt == "npz":
        _write_npz(path, result, key=key)
    else:
        _write_zarr(path, result, key=key)


def _write_npz(path: Path, result: EmbeddingResult, *, key: str | None = None) -> None:
    metadata = _metadata_dict(result, key=key)
    np.savez_compressed(
        path,
        matrix=np.asarray(result.matrix, dtype=np.float32),
        entity_ids=np.asarray(result.entity_ids, dtype=str),
        dim_names=np.asarray(result.dim_names, dtype=str),
        alias_schemes=np.asarray(_alias_schemes(result), dtype=str),
        aliases_json=np.asarray(json.dumps({k: dict(v) for k, v in (result.aliases or {}).items()})),
        metadata_json=np.asarray(json.dumps(metadata, default=str)),
    )


def _write_zarr(path: Path, result: EmbeddingResult, *, key: str | None = None) -> None:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - zarr is in dev env
        raise ImportError("zarr is required for fmt='zarr'. Install zarr or use fmt='npz'/'csv'.") from exc

    root = zarr.open_group(str(path), mode="w")
    root.create_array("matrix", data=np.asarray(result.matrix, dtype=np.float32), chunks="auto")
    root.attrs.update(
        _json_safe(
            {
                **_metadata_dict(result, key=key),
                "schema_version": "embpy.embedding.zarr.v1",
                "entity_ids": list(result.entity_ids),
                "dim_names": result.dim_names,
                "aliases": {k: dict(v) for k, v in (result.aliases or {}).items()},
            }
        )
    )


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


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


def to_payload(
    result: EmbeddingResult,
    *,
    key: str | None = None,
    metadata_mode: PayloadMetadataMode = "full",
    include_matrix: bool = True,
) -> dict:
    """Return a JSON/AnnData-``uns`` friendly embedding payload.

    The payload is entity-aligned rather than observation-aligned:
    ``entity_ids[i]`` is the canonical identifier for ``matrix[i]``.
    This is the preferred representation for perturbation/action
    embeddings because a perturbation is not itself an observation.
    """
    if metadata_mode not in ("minimal", "full"):
        raise ValueError(f"metadata_mode must be 'minimal' or 'full', got {metadata_mode!r}.")
    out_key = key or _default_key(result)
    provenance = result.provenance.to_dict()
    payload: dict = {
        "schema_version": "embpy.uns_embedding.v1",
        "key": out_key,
        "storage": "uns",
        "entity_type": result.entity_type,
        "id_scheme": result.id_scheme,
        "entity_ids": list(result.entity_ids),
        "n_entities": int(result.n_entities),
        "n_dims": int(result.n_dims),
        "model": {
            "name": result.provenance.model,
            "pooling": result.provenance.pooling,
            "layer": result.provenance.layer,
        },
        "provenance": provenance,
    }
    if include_matrix:
        payload["matrix"] = np.asarray(result.matrix, dtype=np.float32)
        payload["dim_names"] = result.dim_names
    if metadata_mode == "full":
        payload["aliases"] = {k: dict(v) for k, v in (result.aliases or {}).items()}
        payload["alias_schemes"] = _alias_schemes(result)
        payload["requested"] = {
            k: v
            for k, v in dict(result.provenance.extra).items()
            if k.startswith("input_") or k.startswith("n_") or k in {"entity_type", "organism"}
        }
        payload["model_config"] = {
            k: v
            for k, v in dict(result.provenance.extra).items()
            if k
            not in {
                "entity_type",
                "organism",
                "input_kind",
                "input_source",
                "input_id_column",
                "input_id_type",
                "n_requested_inputs",
                "n_successfully_embedded_entities",
                "n_embedding_failures",
                "n_unresolved_identifiers",
                "duplicate_canonical_ids_dropped",
                "n_dropped_or_unresolved_entities",
                "canonical_id_scheme",
            }
        }
    return payload


def to_payloads(
    results: EmbeddingResult | Sequence[EmbeddingResult],
    *,
    keys: Sequence[str] | None = None,
    metadata_mode: PayloadMetadataMode = "full",
    include_matrix: bool = True,
) -> dict | dict[str, dict]:
    """Return one or more entity-aligned embedding payloads."""
    items = _as_result_list(results)
    if keys is not None and len(keys) != len(items):
        raise ValueError(f"output routing: keys length must match results length ({len(keys)} != {len(items)}).")
    keyed = dict(zip(keys, items, strict=True)) if keys is not None else _keyed_results(items)
    payloads = {
        key: to_payload(
            result,
            key=key,
            metadata_mode=metadata_mode,
            include_matrix=include_matrix,
        )
        for key, result in keyed.items()
    }
    if len(items) == 1:
        return next(iter(payloads.values()))
    return payloads


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
    _init_embedding_uns(adata, placeholder=True)

    keyed = dict(zip(keys, items, strict=True)) if keys is not None else _keyed_results(items)
    for key, result in keyed.items():
        _attach_to_standalone_axis(adata, result, key)

    if len(items) == 1:
        result = items[0]
        adata.uns.update(_uns_block(result))
    logger.info(
        "Built standalone AnnData (%d obs x %d vars) carrying %d embedding result(s).",
        adata.n_obs,
        adata.n_vars,
        len(items),
    )
    return adata


def _init_embedding_uns(adata: AnnData, *, placeholder: bool) -> None:
    adata.uns.setdefault("embeddings", {})
    adata.uns.setdefault("uns_embeddings", {})
    adata.uns.setdefault("perturbations", {})
    if placeholder:
        adata.uns["placeholder_X"] = {
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
    if _is_perturbation_result(result):
        return "obs"
    if result.entity_type in _UNS_ENTITY_TYPES:
        return "uns"
    if result.entity_type in _VAR_ENTITY_TYPES:
        return "var"
    if result.entity_type in _OBS_ENTITY_TYPES:
        return "obs"
    return "obs"


def _is_perturbation_result(result: EmbeddingResult) -> bool:
    return bool(result.provenance.extra.get("is_perturbation"))


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
    ids = set(_entity_lookup(result))
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
    pos = _entity_lookup(result)
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


def _entity_lookup(result: EmbeddingResult) -> dict[str, int]:
    """Map canonical ids and display aliases to result rows."""
    pos: dict[str, int] = {str(eid): i for i, eid in enumerate(result.entity_ids)}
    if not result.aliases:
        return pos

    canonical_pos = {str(eid): i for i, eid in enumerate(result.entity_ids)}
    for canonical_id, aliases in result.aliases.items():
        row = canonical_pos.get(str(canonical_id))
        if row is None:
            continue
        for value in (aliases or {}).values():
            if value is not None:
                pos.setdefault(str(value), row)
    return pos


def _add_alias_columns(frame: pd.DataFrame, result: EmbeddingResult) -> None:
    if not result.aliases:
        return
    lookup = _entity_lookup(result)
    for scheme in _alias_schemes(result):
        col = scheme
        values = []
        for label in frame.index:
            row = lookup.get(str(label))
            eid = result.entity_ids[row] if row is not None else str(label)
            values.append((result.aliases.get(str(eid), {}) or {}).get(scheme))
        frame[col] = values


def _record_embedding_uns(adata: AnnData, result: EmbeddingResult, key: str) -> None:
    _init_embedding_uns(adata, placeholder=False)
    block = _uns_block(result)
    adata.uns["embeddings"][key] = block
    adata.uns[key] = block


def _store_uns_embedding(adata: AnnData, result: EmbeddingResult, key: str) -> None:
    _init_embedding_uns(adata, placeholder=False)
    payload = to_payload(result, key=key, metadata_mode="full", include_matrix=True)
    collection = (
        "perturbations" if result.entity_type == "perturbation" or key.startswith("X_pert") else "uns_embeddings"
    )
    adata.uns[key] = payload
    adata.uns[collection][key] = payload
    adata.uns["embeddings"][key] = {
        **_uns_block(result),
        "storage": "uns",
        "uns_collection": collection,
    }


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
    attach_to: Literal["auto", "obs", "var", "uns"] = "auto",
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
    ``target.obsm[key]`` or ``target.varm[key]``. ``attach_to="auto"``
    uses entity-specific defaults first: genes attach to ``.varm``
    unless provenance marks them as perturbation labels, in which case
    they attach to ``.obsm``. Other entities are resolved by overlap and
    **raise** (never guess) when both or neither axis clears
    ``min_overlap``. The matrix is re-indexed to the chosen axis order;
    entities missing from the target are an error (``missing="error"``)
    or NaN-filled (``missing="nan"``).
    """
    if target is None:
        return to_anndata_many([result], keys=[key] if key is not None else None)

    out_key = key or _default_key(result)
    if attach_to == "uns" or result.entity_type in _UNS_ENTITY_TYPES:
        _store_uns_embedding(target, result, out_key)
        return target

    if attach_to == "auto":
        if _is_perturbation_result(result):
            attach_to = "obs"
        elif result.entity_type == "gene":
            attach_to = "var"

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
    attach_to: Literal["auto", "obs", "var", "uns"] = "auto",
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
    attach_to: Literal["auto", "obs", "var", "uns"] = "auto",
    harmonize_dim: int | None = None,
    path: str | Path | None = None,
    fmt: FileFormat = "npz",
    missing: Literal["error", "nan"] = "error",
    key: str | None = None,
    metadata_mode: PayloadMetadataMode = "full",
    include_matrix: bool = True,
    min_overlap: float = 1.0,
    random_state: int = 0,
) -> AnnData | pd.DataFrame | dict[str, pd.DataFrame] | dict | dict[str, dict]:
    """The single user-facing output contract shared by ``BioEmbedder.embed``.

    Optionally harmonizes, then dispatches to the chosen backend with
    loud, specific validation:

    * ``output="payload"`` returns an entity-aligned dict with canonical
      ids, embeddings, aliases and provenance. This is the simplest
      representation for perturbation/action embedding tables.
    * ``output="table"`` returns / writes a DataFrame and **ignores**
      ``target`` (warns if one was passed).
    * ``output="anndata"`` builds a standalone AnnData when ``target`` is
      ``None``, else attaches into ``target.obsm``/``.varm``.
    """
    if output not in ("anndata", "table", "payload"):
        raise ValueError(f"output must be 'anndata', 'table', or 'payload', got {output!r}.")

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
    if output == "payload":
        return to_payloads(
            results,
            keys=[key] if key is not None else None,
            metadata_mode=metadata_mode,
            include_matrix=include_matrix,
        )

    keys = [key] if key is not None else None
    return to_anndata_many(
        results,
        target=target,
        attach_to=attach_to,
        keys=keys,
        missing=missing,
        min_overlap=min_overlap,
    )


def materialize_perturbation_obsm(
    adata: AnnData,
    *,
    embedding_key: str,
    perturbation_key: str = "perturbation",
    obsm_key: str | None = None,
    missing: Literal["error", "zero", "nan"] = "error",
) -> AnnData:
    """Expand an entity-aligned perturbation embedding in ``.uns`` to ``.obsm``.

    This is an explicit projection step for models that require one
    action vector per observation. The source-of-truth embedding remains
    ``adata.uns["perturbations"][embedding_key]`` or ``adata.uns[embedding_key]``.
    """
    if perturbation_key not in adata.obs.columns:
        raise KeyError(f"{perturbation_key!r} not in adata.obs (available: {list(adata.obs.columns)})")
    if missing not in ("error", "zero", "nan"):
        raise ValueError(f"missing must be 'error', 'zero', or 'nan', got {missing!r}.")

    payload = _find_uns_payload(adata, embedding_key)
    matrix = np.asarray(payload.get("matrix"), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"uns embedding {embedding_key!r} must contain a 2D matrix, got {matrix.shape!r}.")
    ids = [str(x) for x in payload.get("entity_ids", [])]
    if len(ids) != matrix.shape[0]:
        raise ValueError(
            f"uns embedding {embedding_key!r} has {len(ids)} entity_ids but matrix has {matrix.shape[0]} rows."
        )

    label_to_row: dict[str, int] = {eid: i for i, eid in enumerate(ids)}
    aliases = payload.get("aliases", {}) or {}
    if isinstance(aliases, dict):
        for eid, mapping in aliases.items():
            row = label_to_row.get(str(eid))
            if row is None or not isinstance(mapping, dict):
                continue
            for value in mapping.values():
                if value is not None:
                    label_to_row.setdefault(str(value), row)

    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    out_key = obsm_key or embedding_key
    fill = np.nan if missing == "nan" else 0.0
    out = np.full((adata.n_obs, matrix.shape[1]), fill, dtype=np.float32)
    status: list[str] = []
    missing_labels: list[str] = []
    for i, label in enumerate(labels):
        row = label_to_row.get(str(label))
        if row is None:
            status.append("missing")
            missing_labels.append(str(label))
            continue
        out[i] = matrix[row]
        status.append("resolved")

    if missing_labels and missing == "error":
        preview = list(dict.fromkeys(missing_labels))[:10]
        raise KeyError(
            f"{len(missing_labels)} observations have no perturbation embedding in {embedding_key!r}; "
            f"first missing labels: {preview}. Pass missing='zero' or missing='nan' to fill."
        )

    adata.obsm[out_key] = out
    adata.obs[f"{out_key}_status"] = status
    root = adata.uns.setdefault("world_model_action_embeddings", {})
    root[out_key] = {
        "source": "embpy_uns",
        "uns_key": embedding_key,
        "obsm_key": out_key,
        "perturbation_key": perturbation_key,
        "embedding_dim": int(matrix.shape[1]),
        "n_entities": int(matrix.shape[0]),
        "n_obs": int(adata.n_obs),
        "n_missing_obs": int(sum(s == "missing" for s in status)),
        "model_name": payload.get("model", {}).get("name") if isinstance(payload.get("model"), dict) else None,
        "id_scheme": payload.get("id_scheme"),
    }
    return adata


def _find_uns_payload(adata: AnnData, embedding_key: str) -> dict:
    for collection in ("perturbations", "uns_embeddings"):
        values = adata.uns.get(collection, {})
        if isinstance(values, dict) and embedding_key in values:
            payload = values[embedding_key]
            if isinstance(payload, dict):
                return payload

    embpy = adata.uns.get("embpy", {})
    if isinstance(embpy, dict):
        for collection in ("perturbations", "uns_embeddings"):
            values = embpy.get(collection, {})
            if isinstance(values, dict) and embedding_key in values:
                payload = values[embedding_key]
                if isinstance(payload, dict):
                    return payload
    payload = adata.uns.get(embedding_key)
    if isinstance(payload, dict):
        return payload
    raise KeyError(
        f"Could not find uns embedding {embedding_key!r}. Expected "
        f"adata.uns['perturbations'][{embedding_key!r}], "
        f"adata.uns['embpy']['perturbations'][{embedding_key!r}] for legacy files, "
        f"or adata.uns[{embedding_key!r}]."
    )


__all__ = [
    "materialize_perturbation_obsm",
    "route_output",
    "to_anndata",
    "to_anndata_many",
    "to_payload",
    "to_payloads",
    "to_table",
    "to_tables",
]
