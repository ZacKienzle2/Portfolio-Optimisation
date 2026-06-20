# 8. Covariance estimation strategy

- Status: Accepted
- Date: 2026-06-20

## Context

The sample covariance is noisy and ill-conditioned for portfolio construction,
especially as the number of assets approaches the number of observations, which
destabilises every allocator that consumes it.

## Decision

Ledoit-Wolf shrinkage is the default estimator. Two alternative estimators are
offered and are interchangeable through each allocator's `cov_matrix` argument:
Marchenko-Pastur denoising (with optional market-mode detoning) and factor-model
covariance (principal-component statistical factors or explicit-factor
regression).

## Consequences

- Allocators receive better-conditioned covariances by default.
- Estimators are pluggable and composable with the allocators.
- The estimation choice is explicit and documented per run.
