"""JUMP Cell Painting metadata, gene mapping, and FOV fetching.

``jump-portrait`` 0.1.0 calls ``FROM meta_wells`` in DuckDB while ``broad_babel.data.get_table``
returns a filesystem path string, which triggers a DuckDB replacement-scan error. This module
reimplements the metadata join using ``read_csv_auto`` on that path so image lookup works with
current ``broad-babel`` and ``jump-portrait`` releases.

It also provides:

* :func:`get_jump_gene_mapper` -- thin wrapper around ``broad_babel`` to obtain a
  ``{JCP2022: gene_symbol}`` dictionary for a given plate type.
* :func:`fetch_jump_fov` -- download all 5 Cell Painting channels for a single
  well/site and return them as a stacked ``(5, H, W)`` array in the standard
  ``(DNA, ER, RNA, AGP, Mito)`` order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def get_jump_gene_mapper(
    plate_type: str = "crispr",
) -> dict[str, str]:
    """Return a ``{JCP2022_id: gene_symbol}`` mapping for a JUMP plate type.

    Thin wrapper around :func:`broad_babel.query.get_mapper` that fixes the
    column arguments to the JUMP convention so callers do not need to
    remember the ``broad-babel`` API.

    Parameters
    ----------
    plate_type
        JUMP plate type to query, e.g. ``"crispr"`` (default), ``"orf"``,
        or ``"compound"``.

    Returns
    -------
    dict[str, str]
        Mapping from JCP2022 identifiers to standard gene symbols
        (or compound names for ``"compound"`` plates).

    Examples
    --------
    >>> mapper = get_jump_gene_mapper()          # CRISPR plates
    >>> mapper["JCP2022_807021"]
    'TERT'
    >>> mapper_orf = get_jump_gene_mapper("orf")  # ORF overexpression plates
    """
    from broad_babel.query import get_mapper

    return get_mapper(
        query=plate_type,
        input_column="plate_type",
        output_columns="JCP2022,standard_key",
    )


def get_jump_item_location_metadata(
    item_name: str,
    operator: str | None = None,
    input_column: str = "standard_key",
) -> list[dict[str, Any]]:
    """Return well-level rows joined to the JUMP image index for a gene or compound.

    Parameters match ``jump_portrait.fetch.get_item_location_metadata``. Rows use
    ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well``,
    ``Metadata_Site``, etc.

    Requires ``broad-babel``, ``duckdb``, ``pyarrow``, and ``jump-portrait`` (for
    ``get_index_file`` cache only).
    """
    from broad_babel import query
    from broad_babel.data import get_table

    from jump_portrait.fetch import get_index_file

    if input_column not in ("standard_key", "JCP2022"):
        raise ValueError(
            'input_column must be "standard_key" or "JCP2022", '
            f"got {input_column!r}"
        )

    jcp_ids = query.run_query(
        query=item_name,
        input_column=input_column,
        output_columns="JCP2022,standard_key",
        operator=operator,
    )
    jcp_item = dict(jcp_ids)
    if not jcp_item:
        return []

    meta_wells_path = _sql_path(get_table("well"))
    index_path = _sql_path(str(get_index_file()))
    safe_item = item_name.replace("'", "''")
    in_list = ",".join(_sql_quote(k) for k in jcp_item)

    sql = f"""
    WITH found_rows AS (
        SELECT *, '{safe_item}' AS standard_key
        FROM read_csv_auto('{meta_wells_path}')
        WHERE Metadata_JCP2022 IN ({in_list})
    )
    SELECT * FROM found_rows
    JOIN read_parquet('{index_path}')
    USING (Metadata_Source, Metadata_Plate, Metadata_Well)
    """
    import duckdb

    with duckdb.connect(database=":memory:") as con:
        table = con.execute(sql).fetch_arrow_table()
    return table.to_pylist()


def fetch_jump_fov(well_meta: dict[str, Any]) -> np.ndarray:
    """Fetch all 5 Cell Painting channels for one well/site from the Cell Painting Gallery.

    Returns a ``(5, H, W)`` float32 array in the standard Cell Painting channel
    order: ``(DNA, ER, RNA, AGP, Mito)`` -- matching
    :data:`embpy.pp.CELL_PAINTING_CHANNELS`.

    Parameters
    ----------
    well_meta
        A row dict as returned by :func:`get_jump_item_location_metadata`.
        Must contain ``source`` / ``Metadata_Source``, ``batch`` / ``Metadata_Batch``,
        ``plate`` / ``Metadata_Plate``, ``well`` / ``Metadata_Well``, and optionally
        ``site`` / ``Metadata_Site`` (defaults to ``1``).

    Returns
    -------
    np.ndarray
        ``(5, H, W)`` float32 field-of-view image.

    Examples
    --------
    >>> locs = get_jump_item_location_metadata("PLK1")
    >>> fov = fetch_jump_fov(locs[0])   # (5, H, W)
    """
    import numpy as np
    from jump_portrait.fetch import get_jump_image

    from embpy.pp.morphology_preprocessing import CELL_PAINTING_CHANNELS

    def _get(row: dict, *keys: str):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    src = _get(well_meta, "source", "Metadata_Source")
    batch = _get(well_meta, "batch", "Metadata_Batch")
    plate = _get(well_meta, "plate", "Metadata_Plate")
    well = _get(well_meta, "well", "Metadata_Well")
    site = _get(well_meta, "site", "Metadata_Site")
    if site is None:
        site = 1

    planes = []
    for ch in CELL_PAINTING_CHANNELS:
        img = get_jump_image(
            source=src, batch=batch, plate=plate,
            well=well, channel=ch, site=str(site),
        )
        planes.append(np.asarray(img, dtype=np.float32))
    return np.stack(planes, axis=0)


def _sql_path(path: str) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def _sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"
