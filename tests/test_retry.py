"""Tests for the exponential-backoff retry helper."""

from __future__ import annotations

import pytest

from portfolio_optimisation.infra.retry import retry_call


def test_returns_first_success_without_sleeping() -> None:
    delays: list[float] = []
    result = retry_call(lambda: 42, sleep=delays.append)
    assert result == 42
    assert delays == []


def test_succeeds_after_transient_failures() -> None:
    attempts = {"count": 0}
    delays: list[float] = []

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient")
        return "ok"

    result = retry_call(flaky, attempts=3, base_delay=1.0, sleep=delays.append)
    assert result == "ok"
    assert attempts["count"] == 3
    assert delays == [1.0, 2.0]


def test_raises_last_error_after_exhausting_attempts() -> None:
    def always_fail() -> None:
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        retry_call(always_fail, attempts=2, base_delay=0.0, sleep=lambda _: None)


def test_rejects_non_positive_attempts() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        retry_call(lambda: None, attempts=0)
