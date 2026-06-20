# 15. Mean-risk optimisation and a shared constraint framework

- Status: Accepted
- Date: 2026-06-20

## Context

The hierarchical and risk-parity allocators size positions from the covariance
structure alone. A complementary family minimises a coherent tail measure of
the portfolio loss subject to an expected-return floor, tracing a mean-risk
efficient frontier. Two coherent measures are wanted. Conditional Value-at-Risk
is the established convex tail measure, and Entropic Value-at-Risk is its
tightest coherent upper bound. Both measures, alongside the existing drawdown
programme,
need the same real-world constraints (sign, box, sector, leverage, turnover),
which should be expressed once rather than re-encoded per optimiser.

## Decision

Conditional Value-at-Risk is minimised through the Rockafellar-Uryasev linear
programme, which introduces an auxiliary Value-at-Risk variable and per-scenario
slacks. Entropic Value-at-Risk is minimised through its exponential-cone
formulation; because the measure is positively homogeneous in the loss, the cone
terms are scaled by a fixed constant to improve conditioning without changing the
optimal weights. Both consume a `PortfolioConstraints` value object whose
`build` method emits the shared linear constraints for a supplied weight
variable. All conditions in that object are linear, so they preserve convexity
and global optimality. The convex solvers live behind the `[optim]` extra.
Cardinality limits, which require mixed-integer programming, are deliberately
excluded from this convex framework and tracked separately.

## Consequences

- CVaR and EVaR optimisation share one constraint vocabulary with the drawdown
  programme, and new measures can reuse it without duplicating the encoding.
- Box, sector, leverage, turnover and return-floor mandates are available to the
  mean-risk optimisers from a single, tested value object.
- The solutions are globally optimal because every constraint is linear.
- Cardinality-constrained portfolios remain out of scope until a mixed-integer
  path is added.
