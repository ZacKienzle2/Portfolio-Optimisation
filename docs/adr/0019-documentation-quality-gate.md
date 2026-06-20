# 19. Documentation quality gate

- Status: Accepted
- Date: 2026-06-21

## Context

The coding standards require a docstring on every public function, method and
class, but nothing enforced it, so coverage could erode silently as the surface
grew. The generated API reference and the changelog are only as good as the
docstrings behind them.

## Decision

Public-API docstring coverage is measured by interrogate and checked in
continuous integration alongside the linters and the type checker. Nested
closures, private helpers, and dunder and init methods are excluded from the
measure, since the standards do not require docstrings on them. The threshold is
set at full coverage, since every public symbol now carries a docstring, so any
new undocumented public API fails the build. The auto-generated changelog is
also surfaced as a page in the documentation site so the published reference
shows the release history.

## Consequences

- New public APIs cannot ship undocumented without failing the build.
- The measure tracks the documented public surface rather than internal helpers.
- The published site carries the changelog without a separate maintenance step.
- The full-coverage threshold keeps the entire public API documented.
