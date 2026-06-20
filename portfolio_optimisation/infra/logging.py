"""Structured logging for reproducible, auditable runs.

A single configured logger tree under ``portfolio_optimisation`` replaces ad-hoc
``print`` calls so verbosity is controllable from :class:`Settings` and output
is timestamped and consistently formatted.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "portfolio_optimisation"


def configure_logging(level: str | int = "INFO") -> None:
    """Install a timestamped stream handler on the package logger tree.

    Args:
        level (str | int): Logging level name or numeric level.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger(_ROOT_NAME)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a package logger, configuring the tree once on first use.

    The presence of a handler on the package root acts as the configured flag,
    so repeated calls do not stack duplicate handlers.

    Args:
        name (str): Usually ``__name__`` of the calling module.

    Returns:
        logging.Logger: A logger under the ``portfolio_optimisation`` tree.
    """
    if not logging.getLogger(_ROOT_NAME).handlers:
        configure_logging()
    return logging.getLogger(name)
