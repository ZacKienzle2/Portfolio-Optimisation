# 4. Configuration resolution

- Status: Accepted
- Date: 2026-06-20

## Context

Run parameters (risk-free rate, simulation count, tail level, linkage method,
seed, cache paths) were scattered as literals across the pipeline, notebook and
data layer. That makes runs hard to reproduce, hard to vary in continuous
integration, and impossible to audit from one place.

## Decision

A single frozen `Settings` dataclass holds every knob. A loader resolves values
with a fixed precedence:

1. explicit argument,
2. `PORTFOLIO_*` environment variable,
3. `portfolio.toml`,
4. built-in default.

Each field has a typed coercer so environment and file values are parsed
consistently, and the result is immutable.

## Consequences

- One auditable source of run configuration; no hidden module-level globals.
- Environment and file overrides make CI and deployment reproducible.
- The precedence is explicit, so the origin of any value is predictable.
