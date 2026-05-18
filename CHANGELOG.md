# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Each release section uses the following subsections, omitting any that do not apply:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** for vulnerability fixes.

## [Unreleased]

### Added

- Initial repository scaffolding: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY.
- GitHub workflows: ci, codeql, dependency-review, release, stale, commitlint, pr-labeler.
- Repo metadata: CODEOWNERS, PR template, issue templates, FUNDING, CITATION.
- Tooling configuration: .editorconfig, .yamllint.yaml, .markdownlint.yaml, _typos.toml, commitlint.config.cjs.
- Baseline files: LICENSE (MIT), .gitignore, .gitattributes, CHANGELOG, SUPPORT, GOVERNANCE, AUTHORS, ROADMAP.
- Python stack scaffold: pyproject.toml (uv-managed), .python-version (3.12), `portfolio_optimisation/` package layered as risk / optim / econometrics / sde / viz / infra / domain, smoke tests, `python` CI workflow.
- LaTeX scaffold: neutral report-class `main.tex`, shared preamble at `preamble/packages.tex`, chapters/preliminary/appendix shells with TODO placeholders.
- `optim.denoise`: Marchenko-Pastur eigenvalue denoising and market-mode detoning of sample correlations (Lopez de Prado, 2020).
- `optim.nco`: Nested Clustered Optimisation (Lopez de Prado, 2016).
- `optim.herc`: Hierarchical Equal Risk Contribution with variance or CVaR cluster splits (Raffinot, 2017).
- `risk.coherent`: Entropic Value-at-Risk (Ahmadi-Javid, 2012), spectral risk measures (Acerbi, 2002), and Wang-transform distortion risk (Wang, 2000).
- `optim.cdar`: Minimum Conditional Drawdown-at-Risk linear programme via cvxpy (Chekhlov-Uryasev-Zabarankin, 2005).
- `optim.stochastic_dominance`: Second-order stochastic dominance constrained linear programme (Dentcheva-Ruszczynski, 2003).
- `optim.black_litterman`: Bayesian view blending with reverse-optimised equilibrium prior, Idzorek diagonal Omega, posterior dataclass (Black-Litterman, 1992).
- `risk.sharpe`: Probabilistic and Deflated Sharpe Ratio with skew and kurtosis adjustments, plus Politis-Romano stationary-bootstrap confidence intervals (Bailey-Lopez de Prado, 2014).
- `optim.higher_moments`: Polynomial Goal Programming over Mean-Variance-Skewness-Kurtosis using empirical co-skewness and co-kurtosis tensors (Lai, 1991).
- `[optim]` optional dependency extra wiring cvxpy + ecos + scs solvers.

[Unreleased]: https://github.com/ZacKienzle2/Portfolio-Optimisation/compare/HEAD
