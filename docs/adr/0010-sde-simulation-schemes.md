# 10. SDE simulation schemes

- Status: Accepted
- Date: 2026-06-20

## Context

Process simulation must stay numerically valid (for example, prices and
variances that cannot go negative) while remaining flexible enough to add new
models.

## Decision

Generic Euler-Maruyama and Milstein engines drive the simulators. Geometric
Brownian Motion uses an exact log-Euler scheme so paths stay positive; the
Cox-Ingersoll-Ross process and the Heston variance use full-truncation Euler to
keep the square-root diffusion well defined. Antithetic variates are available
for variance reduction, and every simulator takes an explicit seed.

## Consequences

- Positivity and non-negativity are preserved where the models require them.
- Simulations are reproducible and support variance reduction.
- New processes slot onto the shared engines with little new code.
