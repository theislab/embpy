"""Tests for the ``retry_after`` hook of ``embpy.retry.retry_with_backoff``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from embpy.retry import retry_with_backoff


class _WaitError(Exception):
    def __init__(self, retry_after):
        super().__init__("wait")
        self.retry_after = retry_after


def _flaky(errors):
    """Callable that raises each of *errors* in turn, then returns ``"ok"``."""
    pending = list(errors)

    def fn():
        if pending:
            raise pending.pop(0)
        return "ok"

    return fn


class TestRetryAfterAttribute:
    def test_waits_at_least_retry_after_even_beyond_max_delay(self):
        wrapped = retry_with_backoff(max_attempts=3, base_delay=1.0, max_delay=2.0)(_flaky([_WaitError(7.5)]))

        with patch("embpy.retry.time.sleep") as sleep:
            assert wrapped() == "ok"
        sleep.assert_called_once_with(7.5)

    def test_backoff_wins_when_larger_than_retry_after(self):
        wrapped = retry_with_backoff(max_attempts=3, base_delay=4.0)(_flaky([_WaitError(0.5)]))

        with patch("embpy.retry.time.sleep") as sleep:
            assert wrapped() == "ok"
        sleep.assert_called_once_with(4.0)

    def test_none_retry_after_is_ignored(self):
        wrapped = retry_with_backoff(max_attempts=3, base_delay=1.0)(_flaky([_WaitError(None)]))

        with patch("embpy.retry.time.sleep") as sleep:
            assert wrapped() == "ok"
        sleep.assert_called_once_with(1.0)

    def test_on_retry_receives_the_actual_delay(self):
        seen = []
        wrapped = retry_with_backoff(
            max_attempts=3, base_delay=1.0, on_retry=lambda attempt, exc, delay: seen.append(delay)
        )(_flaky([_WaitError(5.0)]))

        with patch("embpy.retry.time.sleep"):
            wrapped()
        assert seen == [5.0]

    def test_exceptions_without_the_attribute_are_unchanged(self):
        wrapped = retry_with_backoff(max_attempts=2, base_delay=1.5)(_flaky([ValueError("boom")]))

        with patch("embpy.retry.time.sleep") as sleep:
            assert wrapped() == "ok"
        sleep.assert_called_once_with(1.5)

    def test_final_failure_is_reraised(self):
        wrapped = retry_with_backoff(max_attempts=2, base_delay=1.0)(_flaky([_WaitError(1.0), _WaitError(1.0)]))

        with patch("embpy.retry.time.sleep"), pytest.raises(_WaitError):
            wrapped()
