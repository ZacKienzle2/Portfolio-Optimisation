# 11. Testing strategy

- Status: Accepted
- Date: 2026-06-20

## Context

Numerical research code can drift silently: a refactor changes results within
tolerance, or an edge case violates a mathematical invariant without raising.

## Decision

Testing combines several layers: a golden-master regression that pins the
default pipeline outputs on the committed data snapshot; Hypothesis property
tests for invariants (weights on the simplex, positive semi-definite
covariances, Conditional VaR not above VaR); seeded reproducibility tests on
every stochastic path; and a `pytest-benchmark` harness that runs once in the
default suite and is timed on demand. Warnings are errors in the test suite.

## Consequences

- Silent numerical drift and invariant violations are caught.
- Invariants are documented executably, not just in prose.
- Performance is tracked without slowing the default suite.
