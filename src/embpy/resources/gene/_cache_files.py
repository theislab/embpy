"""Settings shared by embpy's gene caches (gene ids and symbol aliases).

Standard library only, so the lightweight alias resolver can use it.
"""

from __future__ import annotations

import logging
import os

SAVE_EVERY_ENV = "EMBPY_GENE_CACHE_SAVE_EVERY"
"""Environment variable: save a gene cache after this many new entries."""

_warned_values: set[str] = set()


def save_every(default: int) -> int:
    """Number of new entries after which a gene cache file is saved.

    ``EMBPY_GENE_CACHE_SAVE_EVERY`` overrides ``default`` when it holds a
    positive whole number (``1`` saves after every new entry); any other value
    is ignored with a warning.
    """
    raw = os.environ.get(SAVE_EVERY_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value >= 1:
        return value
    if raw not in _warned_values:
        _warned_values.add(raw)
        logging.warning(
            "Ignoring %s=%r: expected a positive whole number; saving every %d new entries.",
            SAVE_EVERY_ENV,
            raw,
            default,
        )
    return default
