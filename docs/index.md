# Portfolio Optimisation

Quant infrastructure for portfolio construction, risk modelling and time-series
diagnostics, built on the Python scientific stack (NumPy, SciPy, pandas,
statsmodels, arch, scikit-learn, PyPortfolioOpt and pymle).

## Capabilities

### Allocation

- Hierarchical Risk Parity (HRP) with Ledoit-Wolf shrinkage and
  stationary-bootstrap robustness checks.
- Nested Clustered Optimisation (NCO) and Hierarchical Equal Risk Contribution
  (HERC), with variance- or CVaR-driven cluster splits.
- Marchenko-Pastur correlation denoising and market-mode detoning.
- Black-Litterman Bayesian view blending against an HRP equilibrium prior.
- Minimum Conditional Drawdown-at-Risk (Chekhlov-Uryasev) linear programme.
- Second-order Stochastic Dominance constrained linear programme.
- Polynomial Goal Programming over the first four portfolio moments.

### Risk

- Value-at-Risk and Conditional VaR (empirical and parametric).
- Entropic VaR, spectral risk, and Wang-transform distortion risk.
- Student t-copula simulation with Kendall-tau correlation and MLE degrees of
  freedom.
- Probabilistic and Deflated Sharpe Ratio with stationary-bootstrap confidence
  intervals.

### Econometrics and processes

- Diagnostic battery: Jarque-Bera, ADF, Ljung-Box, Breusch-Pagan, ARCH-LM,
  CUSUM.
- Maximum-likelihood SDE fitting for Geometric Brownian Motion and
  Ornstein-Uhlenbeck processes.

## Design

The codebase follows a domain-driven layering with explicit dependency
inversion: a pure domain layer, infrastructure adapters behind protocols, and a
service layer that orchestrates the workflow. See
[Architecture](architecture.md) for the generated dependency graphs and the
[Decision Records](adr/0001-ddd-layering.md) for the rationale.

## Reproducibility

Every Monte Carlo entry point accepts an explicit seed and is covered by
determinism tests, and the default pipeline is pinned by a golden-master
regression test. Configuration is resolved deterministically from explicit
arguments, environment variables, an optional `portfolio.toml`, and built-in
defaults.
