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
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .result import EmbeddingResult

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


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
        result.matrix, index=index, columns=result.dim_names,
    )
    if alias_frame is None:
        return dim_frame
    return pd.concat([alias_frame, dim_frame], axis=1)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _write_sidecar(path: Path, result: EmbeddingResult) -> Path:
    meta = {
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
    fmt: Literal["parquet", "csv"] = "parquet",
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
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}.")

    frame = _build_frame(result)

    if path is None:
        return frame

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        frame.to_parquet(out, engine="pyarrow")
    else:
        frame.to_csv(out)
    side = _write_sidecar(out, result)
    logger.info(
        "Wrote %s embedding table (%d x %d) to %s; provenance -> %s",
        result.entity_type, result.n_entities, result.n_dims, out, side.name,
    )
    return frame


def _default_key(result: EmbeddingResult) -> str:
    return f"X_emb_{result.provenance.model}"


def _uns_block(result: EmbeddingResult) -> dict:
    return {
        "entity_type": result.entity_type,
        "id_scheme": result.id_scheme,
        "provenance": result.provenance.to_dict(),
        "alias_schemes": _alias_schemes(result),
    }


def _standalone_anndata(result: EmbeddingResult) -> AnnData:
    from anndata import AnnData

    obs = pd.DataFrame(index=pd.Index(result.entity_ids, name=result.id_scheme))
    # Display cross-refs go into .obs so the canonical id stays the index.
    for scheme in _alias_schemes(result):
        obs[scheme] = [
            (result.aliases.get(eid, {}) or {}).get(scheme)  # type: ignore[union-attr]
            for eid in result.entity_ids
        ]
    var = pd.DataFrame(index=pd.Index(result.dim_names, name="embedding_dim"))
    adata = AnnData(X=result.matrix, obs=obs, var=var)
    adata.uns["embpy"] = _uns_block(result)
    logger.info(
        "Built standalone AnnData (%d obs x %d dims) for %s [%s].",
        result.n_entities, result.n_dims, result.entity_type, result.id_scheme,
    )
    return adata


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
        result.entity_type, "obs" if chosen == "obs" else "var",
        chosen_hits, n, chosen,
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
            len(missing_names), axis,
        )
    out = np.full((len(axis_names), result.n_dims), np.nan, dtype=np.float32)
    for i, n in enumerate(axis_names):
        j = pos.get(n)
        if j is not None:
            out[i] = result.matrix[j]
    return out


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

    Standalone (``target=None``): a new AnnData with ``obs_names =
    entity_ids``, ``X = matrix``, ``var_names = dim_0...``, alias
    cross-refs in ``.obs`` and metadata in ``uns["embpy"]``.

    Attach (``target`` given): place the (re-indexed) matrix into
    ``target.obsm[key]`` or ``target.varm[key]`` depending on which axis
    the entity ids align to. ``attach_to="auto"`` decides by overlap and
    **raises** (never guesses) when both or neither axis clears
    ``min_overlap``. The matrix is re-indexed to the chosen axis order;
    entities missing from the target are an error (``missing="error"``)
    or NaN-filled (``missing="nan"``).
    """
    if target is None:
        return _standalone_anndata(result)

    out_key = key or _default_key(result)
    axis = _resolve_axis(result, target, attach_to, min_overlap)
    if axis == "obs":
        mat = _reindex_matrix(result, list(target.obs_names), missing, "obs")
        target.obsm[out_key] = mat
    else:
        mat = _reindex_matrix(result, list(target.var_names), missing, "var")
        target.varm[out_key] = mat
    target.uns.setdefault("embpy", {})[out_key] = _uns_block(result)
    logger.info(
        "Attached %s embedding into target.%sm[%r].",
        result.entity_type, axis, out_key,
    )
    return target


def route_output(
    result: EmbeddingResult,
    *,
    output: Literal["anndata", "table"] = "anndata",
    target: AnnData | None = None,
    attach_to: Literal["auto", "obs", "var"] = "auto",
    harmonize_dim: int | None = None,
    path: str | Path | None = None,
    fmt: Literal["parquet", "csv"] = "parquet",
    missing: Literal["error", "nan"] = "error",
    key: str | None = None,
    min_overlap: float = 1.0,
    random_state: int = 0,
) -> AnnData | pd.DataFrame:
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

    if harmonize_dim is not None:
        from .harmonize import harmonize  # noqa: PLC0415

        result = harmonize(result, harmonize_dim, random_state=random_state)

    if output == "table":
        if target is not None:
            logger.warning(
                "output='table' ignores the provided target AnnData; "
                "returning/writing a table instead."
            )
        return to_table(result, path=path, fmt=fmt)

    return to_anndata(
        result, target=target, attach_to=attach_to, key=key,
        missing=missing, min_overlap=min_overlap,
    )


__all__ = ["to_table", "to_anndata", "route_output"]
