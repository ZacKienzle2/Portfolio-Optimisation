# 16. Covariance shrinkage estimators

- Status: Accepted
- Date: 2026-06-21

## Context

The sample covariance is a poor estimator when the number of assets is not
small relative to the sample length. Its extreme eigenvalues are biased away
from the truth, which destabilises every allocator that inverts or factorises a
covariance. The Marchenko-Pastur denoising in decision record 0008 addresses
the bulk eigenvalues, but a complementary shrinkage family is wanted that gives
an explicit optimal estimator for each eigenvalue.

## Decision

Three shrinkage estimators are added. Linear Ledoit-Wolf and
Oracle-Approximating Shrinkage pull the sample covariance toward a
scaled-identity target with a single optimal intensity, and both wrap the
trusted scikit-learn implementations. Analytical nonlinear shrinkage of Ledoit
and Wolf (2020) keeps the sample eigenvectors and applies a separate optimal
shrinkage to every eigenvalue through an Epanechnikov-kernel estimate of the
limiting spectral density and its Hilbert transform. The nonlinear estimator is
implemented in NumPy and validated numerically against two oracles, recovery of
the sample spectrum as the sample grows and a lower Frobenius error than the
sample covariance on a known population. Every estimator returns a covariance
frame indexed by ticker so the allocators consume it unchanged.

## Consequences

- Allocators can swap in a well-conditioned covariance without code changes.
- The nonlinear estimator handles the case of more assets than observations.
- The linear estimators stay dependency-light through scikit-learn, while the
  nonlinear estimator carries its own validated implementation.
- The numerical validation guards against silent regressions in the kernel
  formula.
