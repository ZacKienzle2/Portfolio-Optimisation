"""Command-line interface for portfolio optimisation runs.

Exposes the package through a ``portfolio-opt`` console script so workflows are
scriptable and reproducible outside notebooks. Configuration is resolved via
:func:`portfolio_optimisation.config.load_settings`, so flags, environment
variables and ``portfolio.toml`` all feed the same run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from portfolio_optimisation import __version__
from portfolio_optimisation.config import Settings, load_settings
from portfolio_optimisation.infra.logging import configure_logging, get_logger
from portfolio_optimisation.services import build_pipeline_from_settings

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-opt",
        description="Portfolio optimisation and risk modelling toolkit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a portfolio.toml configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the package version.")
    subparsers.add_parser("config", help="Print the resolved configuration as JSON.")

    run = subparsers.add_parser(
        "run", help="Run the standard HRP allocation and risk pipeline."
    )
    run.add_argument(
        "--tickers", nargs="+", required=True, help="Asset symbols to include."
    )
    run.add_argument(
        "--start", required=True, help="History start date in YYYY-MM-DD form."
    )
    run.add_argument(
        "--no-copula",
        action="store_true",
        help="Use a historical bootstrap instead of the t-copula simulation.",
    )
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the results JSON to this path instead of stdout.",
    )
    return parser


def _run_pipeline(
    settings: Settings, tickers: list[str], start: str, *, use_copula: bool
) -> dict[str, Any]:
    pipeline = build_pipeline_from_settings(settings)
    result = pipeline.run(
        tickers=tickers,
        start_date=start,
        use_copula=use_copula,
        linkage_method=settings.linkage_method,
    )
    return {
        "weights": {str(k): float(v) for k, v in result.weights.items()},
        "risk_metrics": result.risk_metrics,
        "performance_metrics": result.performance_metrics,
        "metadata": result.metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``portfolio-opt`` console script.

    Args:
        argv (Sequence[str] | None): Argument vector; defaults to ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    configure_logging(settings.log_level)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "config":
        print(json.dumps(settings.to_dict(), indent=2))
        return 0
    if args.command == "run":
        payload = _run_pipeline(
            settings, args.tickers, args.start, use_copula=not args.no_copula
        )
        text = json.dumps(payload, indent=2, default=str)
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
            logger.info("Wrote results to %s", args.output)
        else:
            print(text)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
