"""Record failed web-service requests made by the annotators and resolvers.

The annotators turn a failed request into an empty result so one bad gene
does not abort a batch. That makes "the service has nothing for this gene"
and "the service was down" look the same. Code that needs to tell them apart
wraps the call in :func:`record_service_errors`; the HTTP helpers report each
request that failed (timeouts, connection errors, 5xx -- not 404s) through
:func:`report_service_error`. Outside such a block reporting is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_errors: ContextVar[list[dict[str, str]] | None] = ContextVar("embpy_service_errors", default=None)


@contextmanager
def record_service_errors() -> Iterator[list[dict[str, str]]]:
    """Collect the service errors reported while the block runs."""
    errors: list[dict[str, str]] = []
    token = _errors.set(errors)
    try:
        yield errors
    finally:
        _errors.reset(token)


def report_service_error(url: str, exc: BaseException) -> None:
    """Record a failed request if a :func:`record_service_errors` block is active."""
    errors = _errors.get()
    if errors is not None:
        errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"[:300]})
