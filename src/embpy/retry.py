"""Retry + graceful-degradation primitives (Layer 3).

Two patterns live here:

1. ``retry_with_backoff`` -- a decorator for transient external
   failures (HTTP 5xx from MyGene / Ensembl, ``Timeout`` from
   ``requests``, ``ConnectionResetError``). It retries with
   exponential backoff and capped attempts, then re-raises the
   original exception so upstream classification (Layer 1) still
   works on the final failure.

2. ``embed_batch_with_oom_recovery`` -- a wrapper around any
   ``BaseModelWrapper.embed_batch`` that catches CUDA
   out-of-memory at the batch level, halves the batch, and retries
   the halves. Only the *single* input that genuinely cannot fit on
   the GPU bubbles up as a ``ModelOOMError``; the rest of the batch
   completes. This is the difference between losing 16 hours of
   work to one giant intron and losing one row.

Both helpers are intentionally side-effect-light: they do not log
unless ``EMBPY_STRUCTURED_LOG=1`` is set (delegating to Layer 4 for
observability) and they do not record into ``ResolutionReport`` directly
(Layer 2 is wired in at the call sites, not here). This keeps the
retry / recovery logic composable and unit-testable in isolation.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------


def retry_with_backoff(
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable: tuple[type[BaseException], ...] = (Exception,),
    non_retryable: tuple[type[BaseException], ...] = (),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory: retry ``func`` on transient failures.

    Parameters
    ----------
    max_attempts
        Maximum number of attempts including the first. ``3`` means
        the call is tried at ``t=0``, ``t=base_delay``, and
        ``t=base_delay * backoff_factor``.
    base_delay
        Seconds to wait before the *second* attempt. Subsequent
        delays are ``base_delay * backoff_factor**(attempt - 1)``,
        capped at ``max_delay``.
    max_delay
        Hard cap on the inter-attempt wait, regardless of
        ``backoff_factor``.
    backoff_factor
        Multiplier applied to the delay between attempts.
    retryable
        Exception types that trigger a retry. Defaults to ``Exception``
        i.e. retry everything Python; tighten to e.g.
        ``(requests.HTTPError, requests.Timeout,
        requests.ConnectionError)`` for HTTP work.
    non_retryable
        Exception types that always propagate immediately, even when
        they would otherwise match ``retryable``. Use this to short-
        circuit on permanent errors like ``HTTPError`` with a 4xx
        status code (caller can subclass / wrap to express that).
    on_retry
        Optional callback invoked as ``on_retry(attempt, exc, delay)``
        just before sleeping. Useful for emitting structured-log
        events from Layer 4 without coupling the decorator to it.

    Behaviour
    ---------
    On the *final* failed attempt the exception is re-raised verbatim.
    This keeps Layer 1 classification working: a real CUDA OOM after
    3 retries still surfaces as ``torch.OutOfMemoryError``, which
    ``_classify_embedder_exception`` then maps to ``ModelOOMError``
    with the right exit code.
    """

    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            delay = float(base_delay)
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except non_retryable:
                    # Explicit short-circuit -- never retry.
                    raise
                except retryable as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        # Final attempt: re-raise verbatim so callers
                        # can classify the failure normally.
                        raise
                    sleep_for = min(delay, max_delay)
                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc, sleep_for)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "on_retry callback raised; ignoring.",
                                exc_info=True,
                            )
                    logger.debug(
                        "retry_with_backoff: attempt %d/%d failed (%s: %s); "
                        "sleeping %.2fs before retry.",
                        attempt, max_attempts, type(exc).__name__, exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay *= backoff_factor
            # Defensive: loop should always raise or return.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("retry_with_backoff: unreachable")
        return _wrapper

    return _decorator


# ---------------------------------------------------------------------------
# OOM bisection
# ---------------------------------------------------------------------------


def _is_cuda_oom(exc: BaseException) -> bool:
    """Heuristic: is this exception a CUDA OOM we should bisect on?

    We accept both ``torch.OutOfMemoryError`` (added in torch 2.4) and
    the older ``RuntimeError("CUDA out of memory")`` because not every
    pixi env has the new typed exception. We also explicitly catch the
    ``RuntimeError`` raised by Triton / cuBLAS for "Tensor would
    overflow the maximum tensor size" -- those are functionally OOMs
    even when the error message is different.

    We do NOT bisect on generic ``RuntimeError`` -- only on the
    OOM-shaped subset, to avoid swallowing real bugs.
    """
    try:
        import torch  # noqa: PLC0415

        if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
            return True
    except (ImportError, AttributeError):
        pass

    msg = str(exc)
    return any(
        token in msg for token in (
            "CUDA out of memory",
            "out of memory",
            "OOM",
            "CUBLAS_STATUS_NOT_INITIALIZED",
            "CUDNN_STATUS_NOT_INITIALIZED",
        )
    )


def _empty_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache``; silent if torch unavailable."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def embed_batch_with_oom_recovery(
    embed_fn: Callable[[Sequence[Any]], Sequence[Any]],
    inputs: Sequence[Any],
    *,
    min_batch_size: int = 1,
    max_bisections: int = 8,
    on_split: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """Run ``embed_fn(batch)`` with adaptive bisection on CUDA OOM.

    Strategy
    --------
    Try ``embed_fn(inputs)``. If it raises an OOM-shaped exception,
    halve the input list, recurse on each half, and concatenate the
    results. The recursion stops when the batch size hits
    ``min_batch_size``; if a single-item batch still OOMs we re-raise
    so callers can record the input as UNRESOLVED with reason
    ``"model:context_overflow"``.

    Parameters
    ----------
    embed_fn
        Callable that takes a list of inputs and returns a parallel
        list of embeddings (length-preserving). Typically a
        ``functools.partial`` over ``wrapper.embed_batch``.
    inputs
        The full input batch. May be any sequence; treated as a list.
    min_batch_size
        Stop halving once the batch is this small. ``1`` is the right
        choice for the embedder use case (we want to know exactly
        which input is too big).
    max_bisections
        Safety bound on the recursion depth (counted across the whole
        call tree). Mostly defends against pathological input lists.
    on_split
        Optional callback ``(left_size, right_size)`` invoked on each
        bisection. Useful for emitting Layer 4 structured events.

    Returns
    -------
    list of embeddings, one per input, in the original order.
    """
    inputs_list = list(inputs)
    if not inputs_list:
        return []

    counter = {"bisections": 0}

    def _run(batch: list[Any]) -> list[Any]:
        try:
            result = embed_fn(batch)
            # Embed functions return list/sequence; normalise to list.
            return list(result)
        except BaseException as exc:  # noqa: BLE001
            if not _is_cuda_oom(exc):
                # Not OOM-shaped -- never bisect on this; propagate so
                # the caller can decide whether it's a typed embpy
                # error (already handled) or something genuinely new.
                raise
            if len(batch) <= min_batch_size:
                # Genuinely too big for the GPU. Re-raise so the call
                # site can record it as UNRESOLVED (Layer 2) or let
                # the typed-error path (Layer 1) classify it.
                raise
            counter["bisections"] += 1
            if counter["bisections"] > max_bisections:
                logger.warning(
                    "embed_batch_with_oom_recovery: reached max_bisections=%d, "
                    "propagating the last OOM rather than continuing to halve.",
                    max_bisections,
                )
                raise
            _empty_cuda_cache()
            mid = max(1, len(batch) // 2)
            if on_split is not None:
                try:
                    on_split(mid, len(batch) - mid)
                except Exception:  # noqa: BLE001
                    logger.debug("on_split callback raised; ignoring.", exc_info=True)
            logger.warning(
                "CUDA OOM at batch_size=%d (%s); halving to %d + %d and retrying.",
                len(batch), type(exc).__name__, mid, len(batch) - mid,
            )
            left = _run(batch[:mid])
            right = _run(batch[mid:])
            return left + right

    return _run(inputs_list)


__all__ = [
    "retry_with_backoff",
    "embed_batch_with_oom_recovery",
    "_is_cuda_oom",
]
