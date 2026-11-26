"""Which matrix of an AnnData holds the counts that single-cell preprocessing starts from.

Shared by :func:`embpy.pp.preprocess_counts` (and so ``BioEmbedder.embed_cells``) and the
MCP tools (``inspect_data``, ``preprocess_counts``, ``compute_embeddings``), so that every
one of them reads the same file the same way. It lives outside :mod:`embpy.pp` so that
the MCP tools can use it without importing the whole preprocessing package.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NotRawCountsWarning(UserWarning):
    """Preprocessing was given values that are not raw integer counts.

    Also given when preprocessing keeps an input ``.layers['counts']`` that differs from the
    counts it used, as ``.layers['counts_original']``.
    """


# Kinds of values that are clearly not counts.
NOT_COUNTS = ("log_normalized", "continuous_normalized", "scaled_or_centered")
# Kinds of values used as counts.
COUNT_LIKE = ("raw_counts", "non_integer_counts")
# Log-transformed expression stays below this (log1p of 1e4-normalized counts below ~10,
# log2 of CPM below ~20), while the counts of 50 cells nearly always exceed it.
LOG_RANGE_MAX = 25.0


def describe_values(matrix: Any, log1p_marked: bool = False) -> dict[str, Any]:
    """What a count matrix holds, judged from its first 50 rows.

    ``kind`` is one of

    - ``"raw_counts"``: non-negative integers;
    - ``"non_integer_counts"``: non-negative non-integers with count-like magnitudes
      (above the log range, rows with different totals), e.g. ambient-corrected counts
      (SoupX, DecontX) or EM-assigned counts (alevin-fry, STARsolo, bustools ``--em``);
    - ``"log_normalized"``: non-negative non-integers in the log range (at most 25), or any
      non-integer matrix that ``.uns['log1p']`` marks as log-transformed (``log1p_marked``);
    - ``"continuous_normalized"``: non-negative non-integers whose rows all have the same
      total (``normalize_total`` without a log);
    - ``"scaled_or_centered"``: negative values;
    - ``"unknown"``: nothing to sample.

    The result also holds ``is_integer``, ``has_negative``, ``min`` and ``max`` (of the
    sample) and ``log1p_marked``.
    """
    import scipy.sparse as sp  # here, so importing this module stays as cheap as numpy

    rows = matrix[: min(50, matrix.shape[0])]
    if sp.issparse(rows):
        rows = sp.csr_matrix(rows)
        values = np.asarray(rows.data, dtype=np.float64)
        totals = np.asarray(rows.sum(axis=1), dtype=np.float64).ravel()
    else:
        dense = np.asarray(rows, dtype=np.float64)
        values = dense.ravel()
        totals = dense.sum(axis=1) if dense.ndim == 2 else np.zeros(0)
    values = values[np.isfinite(values)]
    info: dict[str, Any] = {
        "kind": "unknown",
        "is_integer": False,
        "has_negative": False,
        "min": None,
        "max": None,
        "log1p_marked": bool(log1p_marked),
    }
    if values.size == 0:
        return info
    low, high = float(values.min()), float(values.max())
    is_integer = bool(np.all(np.isclose(values, np.round(values), atol=1e-4)))
    info.update(is_integer=is_integer, has_negative=low < -1e-5, min=low, max=high)
    positive = totals[totals > 0]
    if low < -1e-5:
        info["kind"] = "scaled_or_centered"
    elif is_integer:
        info["kind"] = "raw_counts"
    elif log1p_marked:
        info["kind"] = "log_normalized"
    elif positive.size >= 2 and positive.max() - positive.min() <= 1e-3 * positive.mean():
        info["kind"] = "continuous_normalized"
    elif high <= LOG_RANGE_MAX:
        info["kind"] = "log_normalized"
    else:
        info["kind"] = "non_integer_counts"
    return info


def what_x_holds(x: dict[str, Any]) -> str:
    """'<kind> values (<why>)' for a .X that is clearly not counts (see :func:`describe_values`)."""
    if x["kind"] == "scaled_or_centered":
        why = f"negative values, down to {x['min']:.3g}"
    elif x["log1p_marked"]:
        why = ".uns['log1p'] marks it as log-transformed"
    elif x["kind"] == "continuous_normalized":
        why = "every sampled cell has the same total"
    else:
        why = f"non-integers no larger than {x['max']:.3g}, the range of log-transformed data"
    return f"{x['kind']} values ({why})"


def same_values(a: Any, b: Any) -> bool:
    """Whether two matrices (dense or sparse) hold the same values."""
    import scipy.sparse as sp

    if a.shape != b.shape:
        return False
    if sp.issparse(a) or sp.issparse(b):
        return (sp.csr_matrix(a) != sp.csr_matrix(b)).nnz == 0
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def counts_source(adata: Any) -> dict[str, Any]:
    """Which matrix preprocessing takes its counts from, and what to tell the user about it.

    ``.X`` is the source when it holds counts: integers, or non-integers with count-like
    magnitudes (corrected or EM counts). ``.layers['counts']`` is the source only when ``.X``
    is clearly not counts (log-transformed, normalized or scaled) and that layer holds counts.
    Preprocessing stores the counts it used in ``.layers['counts']``; when ``.X`` is used and
    the input's ``.layers['counts']`` holds other values (e.g. the original counts next to
    corrected ones), that layer is kept under ``keep_original_as`` (``counts_original``).

    Returns ``{"source": "X" | "counts", "found": bool, "x": ..., "layer": ...,
    "keep_original_as": str | None, "message": str | None}``: ``found`` is False when neither
    holds counts (``.X`` is then the source), ``x`` and ``layer`` describe ``.X`` and
    ``.layers['counts']`` (see :func:`describe_values`; ``layer`` is None without that layer),
    and ``message`` is the warning to give, if any.
    """
    x = describe_values(adata.X, log1p_marked="log1p" in adata.uns)
    layer = describe_values(adata.layers["counts"]) if "counts" in adata.layers else None
    choice: dict[str, Any] = {
        "source": "X",
        "found": True,
        "x": x,
        "layer": layer,
        "keep_original_as": None,
        "message": None,
    }
    notes: list[str] = []
    if x["kind"] == "non_integer_counts":
        notes.append(
            f".X holds non-integer values with count-like magnitudes (up to {x['max']:.4g}), e.g. "
            "ambient-corrected (SoupX, DecontX) or EM-assigned counts. They are used as the counts."
        )
    elif x["kind"] in NOT_COUNTS and layer is not None and layer["kind"] in COUNT_LIKE:
        choice["source"] = "counts"
        choice["message"] = (
            f".X holds {what_x_holds(x)}, so the counts are taken from .layers['counts'] "
            f"({layer['kind']}) instead of .X."
        )
        return choice
    elif x["kind"] in NOT_COUNTS:
        elsewhere = (
            f" and .layers['counts'] holds {layer['kind']} values"
            if layer is not None
            else " and there is no .layers['counts']"
        )
        choice["found"] = False
        notes.append(
            f"preprocess_counts expects raw counts, but .X holds {what_x_holds(x)}{elsewhere}. They are "
            "treated as raw counts, so normalizing and log-transforming them again distorts the data. "
            "Pass raw counts in .X or .layers['counts'], or skip preprocessing for data that is already normalized."
        )
    # .X is the source; an input .layers['counts'] with other values is kept, not overwritten.
    if layer is not None and not same_values(adata.X, adata.layers["counts"]):
        name, number = "counts_original", 2
        while name in adata.layers:
            name, number = f"counts_original_{number}", number + 1
        choice["keep_original_as"] = name
        notes.append(
            f"The input's .layers['counts'] holds other values than .X, which is used as the counts, "
            f"so that layer is kept as .layers['{name}']."
        )
    choice["message"] = " ".join(notes) or None
    return choice
