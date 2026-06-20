# Portfolio Optimisation

Quant infrastructure for portfolio construction, risk modelling and time
series diagnostics. Python 3.12+ over numpy, scipy, pandas, statsmodels,
arch, sklearn, pypfopt and pymle.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe5196.svg)](https://www.conventionalcommits.org/en/v1.0.0/)
[![SemVer](https://img.shields.io/badge/SemVer-2.0.0-blue.svg)](https://semver.org/spec/v2.0.0.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- CI, CodeQL and last-commit badges return on public visibility. -->

## Scope

### Allocation
- Hierarchical Risk Parity (HRP) with Ledoit-Wolf shrinkage and
  stationary-bootstrap robustness checks
- Nested Clustered Optimisation (NCO) and Hierarchical Equal Risk
  Contribution (HERC), with variance- or CVaR-driven cluster splits
- Marchenko-Pastur correlation denoising plus market-mode detoning
- Black-Litterman Bayesian view blending against an HRP equilibrium prior
- Minimum Conditional Drawdown-at-Risk (Chekhlov-Uryasev) LP
- Second-order Stochastic Dominance constrained LP
- Polynomial Goal Programming over Mean-Variance-Skewness-Kurtosis with
  empirical co-skewness M3 and co-kurtosis M4 tensors

### Risk
- Value-at-Risk and Conditional VaR (empirical and parametric)
- Entropic VaR (coherent, Chernoff-bound formulation)
- Spectral risk via exponential or power admissible spectra
- Wang-transform distortion risk
- Student t-copula simulation with Kendall-tau correlation and MLE
  degrees-of-freedom estimation
- Probabilistic and Deflated Sharpe Ratio (Bailey-Lopez de Prado), plus
  Politis-Romano stationary-bootstrap CIs on the Sharpe ratio

### Econometrics + processes
- Test battery: Jarque-Bera, ADF, Ljung-Box, Breusch-Pagan, ARCH-LM, CUSUM
- Maximum-likelihood SDE fitting: Geometric Brownian Motion,
  Ornstein-Uhlenbeck

### Visualisation
- Plotly and matplotlib for efficient frontier, weights, dendrogram,
  correlation heatmap

## Install

```bash
uv sync --frozen --all-extras
```

Requires `uv >= 0.5`. Python 3.12 auto-selected via `.python-version`.

## Usage

```python
from portfolio_optimisation.optim import HRPModel
from portfolio_optimisation.risk import calculate_risk_metrics, CopulaRiskAnalyser
from portfolio_optimisation.econometrics import Econometrics
from portfolio_optimisation.sde import SDEFitter
```

See `main.ipynb` for an end-to-end worked example.

### Command line

```bash
portfolio-opt version
portfolio-opt config
portfolio-opt run --tickers IYW VGT IYF --start 2018-01-01 --output result.json
```

Configuration resolves with the precedence explicit flag > `PORTFOLIO_*`
environment variable > `portfolio.toml` > built-in default. Every Monte Carlo
path accepts a `seed` for reproducible results.

### Architecture diagrams

Module and DDD-layer dependency graphs are generated from the source and kept in
sync by CI:

```bash
python tools/gen_diagrams.py          # regenerate docs/diagrams/*
python tools/gen_diagrams.py --check  # verify they match the source
```

## Maintainers

See [CODEOWNERS](.github/CODEOWNERS).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Conventional Commits 1.0.0 and DCO
sign-off required.

## License

[MIT](LICENSE).

## Related

[SECURITY](SECURITY.md) | [SUPPORT](SUPPORT.md) | [GOVERNANCE](GOVERNANCE.md) | [CHANGELOG](CHANGELOG.md) | [ROADMAP](ROADMAP.md) | [CITATION](CITATION.cff)
