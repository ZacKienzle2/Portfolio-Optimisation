# 5. Structured logging

- Status: Accepted
- Date: 2026-06-20

## Context

Library modules emitted status and warning messages with `print`. That pollutes
stdout, cannot be silenced or filtered, carries no timestamps, and mixes library
diagnostics with intended program output.

## Decision

A package logger tree under `portfolio_optimisation` is configured once via
`get_logger`/`configure_logging`, with a timestamped formatter and a level taken
from `Settings`. Library `print` calls are replaced with logger calls. The Rich
console is retained only for explicitly user-facing report output.

## Consequences

- Verbosity is controlled centrally and per environment.
- Output is timestamped and consistently formatted.
- Tests can capture or silence diagnostics without touching stdout.
