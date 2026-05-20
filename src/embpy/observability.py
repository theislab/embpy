"""Structured logging + timing primitives (Layer 4).

This module exists so embpy can emit machine-parseable JSON-line
events alongside the human-readable ``logging`` output, without
forcing every call site to know about JSON or threading. The events
are designed to be sliced by ``jq`` after a SLURM run:

    jq -s 'sort_by(.latency_ms) | reverse | .[:10]' logs/*.jsonl

Opt-in
------
Structured logging is **off by default**. Set ``EMBPY_STRUCTURED_LOG=1``
to enable; without it, ``log_event`` is a near-zero-cost no-op (one
dict allocation that is discarded) and ``time_block`` only does the
timing math. This keeps the noise floor in unit tests and notebooks
unchanged.

Optionally set ``EMBPY_STRUCTURED_LOG_FILE=<path>`` to redirect the
JSON-line stream to a separate file; otherwise events go to a dedicated
logger named ``embpy.events`` which can be attached to a handler by
the caller.

Field conventions
-----------------
Every event has at least::

    ts          ISO-8601 UTC timestamp
    event       event kind (string, lowercase, snake_case)
    level       "debug" | "info" | "warn" | "error"

Common optional fields::

    model       embpy model registry key (e.g. "minilm_l6_v2")
    identifier  input identifier (gene symbol, etc.)
    source      data source (e.g. "mygene", "ensembl")
    latency_ms  end-to-end latency for this op
    status      "ok" | "error" | "retry"
    error       short error class name
    request_id  per-batch UUID for correlating multi-line traces

Callers SHOULD prefer adding fields over inventing new event kinds
when extending the schema; ``jq`` slices stay simpler that way.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Dedicated logger so structured events can be routed independently
# of the human-readable embpy logs.
_event_logger = logging.getLogger("embpy.events")


def _structured_enabled() -> bool:
    """Cheap env-var check; called once per event call site."""
    return os.environ.get("EMBPY_STRUCTURED_LOG", "").strip() in ("1", "true", "yes")


def _ensure_handler_attached() -> None:
    """Lazily attach a file handler if ``EMBPY_STRUCTURED_LOG_FILE`` is set.

    We only attach once per process; subsequent calls are no-ops. The
    handler uses a minimal formatter that emits the raw message
    (already a JSON string) verbatim, with no level / timestamp
    prefix -- the message itself is self-describing.
    """
    if getattr(_event_logger, "_embpy_handler_attached", False):
        return
    path = os.environ.get("EMBPY_STRUCTURED_LOG_FILE", "").strip()
    if path:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            _event_logger.addHandler(handler)
            _event_logger.setLevel(logging.DEBUG)
            # Don't propagate to root: the JSON lines are noise in the
            # human log.
            _event_logger.propagate = False
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Could not attach EMBPY_STRUCTURED_LOG_FILE handler at %r: %s",
                path, exc,
            )
    _event_logger._embpy_handler_attached = True  # type: ignore[attr-defined]


def new_request_id() -> str:
    """Short hex token to correlate multi-line traces."""
    return uuid.uuid4().hex[:12]


def log_event(
    event: str,
    *,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit one structured JSON line.

    No-op if ``EMBPY_STRUCTURED_LOG`` is not set. Fields are
    serialised with ``json.dumps(default=str)`` so non-JSON-native
    values (Path, numpy scalars, exceptions) degrade to their string
    form rather than crashing.
    """
    if not _structured_enabled():
        return
    _ensure_handler_attached()
    record = {
        "ts": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "level": level,
        **fields,
    }
    try:
        line = json.dumps(record, default=str)
    except (TypeError, ValueError):
        # Last-resort: stringify and emit so we never lose an event
        # because of a bad field.
        line = json.dumps({
            "ts": record["ts"],
            "event": event,
            "level": "error",
            "_serialization_error": True,
            "_repr": repr(record),
        })
    log_method = getattr(_event_logger, level if level != "warn" else "warning", _event_logger.info)
    log_method(line)


@contextmanager
def time_block(
    event: str,
    *,
    level: str = "info",
    log_on_enter: bool = False,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Context manager that times the block and emits a structured event.

    Usage::

        with time_block("resolver_call", source="mygene", identifier=g) as ctx:
            result = _query_mygene(g)
            ctx["status"] = "ok" if result else "no_hit"

    The yielded dict can be mutated to add fields recorded only after
    the block runs (``status``, error codes, computed sizes). On exit
    the event is emitted with ``latency_ms`` populated automatically.
    If the block raises, the event is still emitted with
    ``status="error"`` and ``error=<exc class name>`` before the
    exception propagates.

    The block always measures latency (cheap), but the emit is gated
    on ``EMBPY_STRUCTURED_LOG`` like :func:`log_event`. The ``ctx``
    dict is yielded regardless so callers can rely on it.
    """
    ctx: dict[str, Any] = dict(fields)
    if log_on_enter:
        log_event(f"{event}_start", level="debug", **ctx)
    started = time.perf_counter()
    try:
        yield ctx
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ctx.setdefault("status", "error")
        ctx.setdefault("error", type(exc).__name__)
        ctx["latency_ms"] = round(elapsed_ms, 3)
        log_event(event, level="error", **ctx)
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ctx.setdefault("status", "ok")
        ctx["latency_ms"] = round(elapsed_ms, 3)
        log_event(event, level=level, **ctx)


__all__ = ["log_event", "time_block", "new_request_id"]
