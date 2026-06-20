# Roadmap

This document tracks high-level direction. For granular work see the [issue tracker](https://github.com/ZacKienzle2/Portfolio-Optimisation/issues) and [project boards](https://github.com/ZacKienzle2/Portfolio-Optimisation/projects).

The roadmap is intentionally aspirational. Items are not commitments. Priorities shift as the project learns from users and contributors. Dates are quarters of the calendar year.

## Vision

A research-grade, fully reproducible toolkit for portfolio construction and risk
modelling: rigorous numerics with literature references, statistical validation
of every model, high-performance simulation, and documentation an external
reviewer can audit end to end.

## Recently delivered (Q2 2026)

- Request-validated market-data cache (no stale snapshots) and a deduplicated, retrying data layer.
- Deterministic seeding across every Monte Carlo path with reproducibility and golden-master tests.
- `einsum` co-moment tensors with an O(N^4) universe guard; vectorised RMT eigen-reconstruction.
- Typed `Settings` (env / TOML / defaults), structured logging, typed errors.
- `portfolio-opt` command-line interface.
- Cohesive, colour-blind-safe figure-style system; plotting decoupled from the optimisation layer.
- Code-derived architecture diagrams (Mermaid + Graphviz) with a CI layer-violation guard.
- MkDocs documentation site with API reference and architecture decision records.
- VaR/ES backtests (Kupiec, Christoffersen, Acerbi-Szekely) and risk-contribution decomposition.
- GARCH-family conditional-volatility forecasting feeding the VaR/ES backtests.
- Factor-model covariance (statistical PCA and explicit-factor regression) for all allocators.
- Property-based tests and a benchmark harness.

## Now

In active development.

- TTL-aware market-data cache refresh policy on top of the validated snapshot.
- Walk-forward backtesting engine with transaction costs and turnover.
- Expanded methodology pages with full mathematical derivations per method.

## Next

Planned for the next milestone (H2 2026).

- Mean-CVaR and mean-EVaR optimisation (Rockafellar-Uryasev) and a unified constraint framework.
- Extreme Value Theory tail fitting (peaks-over-threshold, generalised Pareto, Hill estimator).
- Additional SDEs with Euler-Maruyama and Milstein simulation engines (CIR, Vasicek, Heston, Merton jumps).

## Later

Under consideration. Open issues to discuss.

- GPU / compiled-kernel acceleration for large-universe simulation.
- Live and alternative data adapters behind the existing repository protocol.
- Interactive dashboard over the pipeline outputs.
- Bridge to the sibling deep-hedging project for path-dependent allocation.

## Capability gaps (research agenda)

The thematic backlog of what a frontier version of this project would add.

### Allocation and optimisation

- Risk parity and equal-risk-contribution (convex Spinu / Newton formulations).
- Mean-CVaR, mean-EVaR and mean-drawdown optimisation under a shared constraint API.
- Robust optimisation with uncertainty sets; resampled efficient frontier (Michaud).
- Cardinality, sector, box, turnover and leverage constraints across every allocator.
- Critical Line Algorithm and full mean-variance frontier tracing.
- Black-Litterman with entropy pooling and view-uncertainty calibration (Meucci).
- Hierarchical risk parity variants (HRP with alternative linkage and distance metrics).

### Risk modelling

- Conditional volatility models: GARCH, EGARCH, GJR-GARCH; DCC-GARCH for dynamic correlation.
- Extreme Value Theory: peaks-over-threshold, generalised Pareto tails, Hill estimator.
- Drawdown analytics: Calmar, Ulcer index, pain index, MAR ratio, time-under-water.
- Component and incremental VaR / CVaR alongside the volatility decomposition.
- Wider copula families (Clayton, Gumbel, Frank, vine copulas) with goodness-of-fit tests.
- Spectral and distortion risk budgeting.

### Stochastic processes and simulation

- SDE library: CIR, Vasicek, CKLS, Heston, Merton and Kou jump-diffusions.
- Euler-Maruyama, Milstein and exact / almost-exact (Broadie-Kaya, Andersen QE) schemes.
- Variance reduction: antithetic variates, control variates, quasi-Monte-Carlo (Sobol).
- Regime-switching (Hamilton / hidden Markov) and Kalman-filtered latent states.

### Estimation and machine learning

- Nonlinear shrinkage (Ledoit-Wolf 2020) and OAS covariance estimators.
- Cluster-count selection (gap statistic, silhouette) for the hierarchical methods.
- Combinatorial purged cross-validation and deflated performance metrics.
- Bayesian estimation (PyMC) and Gaussian-process return models.

### Backtesting and validation

- Walk-forward and combinatorial backtests with transaction-cost models.
- Multiple-testing controls: White's reality check, Hansen's SPA, deflated Sharpe across strategies.
- Mutation testing and a coverage threshold gate in continuous integration.
- Numerical-accuracy tests against closed-form analytic benchmarks.

### Performance and high-performance computing

- Numba kernels for the copula and Monte-Carlo inner loops behind the `perf` extra.
- Optional compiled (Cython / pybind11) and CUDA simulation backends.
- Parallel and vectorised backtest execution; Polars data path.
- Performance-regression tracking in continuous integration.

### Data and infrastructure

- Multiple data-source adapters behind the repository protocol; point-in-time, survivorship-bias-free data.
- Partitioned on-disk store and asynchronous data acquisition.
- Experiment registry: config hashing, run provenance and artefact tracking.
- Container and dev-container images for reproducible environments.

### Documentation and academic rigour

- Per-method methodology notes with derivations, assumptions and references.
- Worked case studies and benchmark comparisons against PyPortfolioOpt and Riskfolio.
- A rendered, executed gallery of the worked example in the documentation site.

## Out of Scope

Explicitly not on the roadmap. Open an issue to challenge if you disagree.

- Live order routing, execution or brokerage integration.
- Real-money trading or portfolio custody.
- Rewrites of the numerical core outside the Python scientific stack.

## How to Influence the Roadmap

- Comment on items above with use cases.
- File feature requests using the [feature template](.github/ISSUE_TEMPLATE/feature_request.yml).
- Join discussions in [GitHub Discussions](https://github.com/ZacKienzle2/Portfolio-Optimisation/discussions).
- Submit a pull request demonstrating the idea.
