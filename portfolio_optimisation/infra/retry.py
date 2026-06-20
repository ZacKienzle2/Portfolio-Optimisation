"""Retry helper with exponential backoff for transient failures.

Used to make network-bound market-data acquisition resilient to intermittent
errors without pulling in a third-party retry dependency. The sleep function is
injectable so the backoff logic is testable without real delays.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from logging import Logger


def retry_call[T](
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    logger: Logger | None = None,
) -> T:
    """Call ``func`` with up to ``attempts`` tries and exponential backoff.

    Args:
        func (Callable[[], T]): Zero-argument callable to invoke.
        attempts (int): Maximum number of attempts (must be >= 1).
        base_delay (float): Delay before the first retry; doubles each attempt.
        exceptions (tuple[type[BaseException], ...]): Exception types treated as
            transient and retried.
        sleep (Callable[[float], None]): Sleep function; injectable for tests.
        logger (Logger | None): Optional logger for retry warnings.

    Returns:
        T: The first successful result of ``func``.

    Raises:
        ValueError: If ``attempts`` is less than 1.
        BaseException: The last exception raised once attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as error:
            last_error = error
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    "attempt %d/%d failed: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    error,
                    delay,
                )
            sleep(delay)

    assert last_error is not None
    raise last_error
