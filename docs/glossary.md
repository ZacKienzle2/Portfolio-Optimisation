# Glossary

The shared vocabulary of the project. The terms below name the same concepts in
the code, the documentation and the writeup.

## Architecture

| Term | Meaning |
| --- | --- |
| Domain | Pure layer of protocols and value objects with no IO or framework coupling. |
| Repository | Port that returns aggregates from a data source while hiding the storage. |
| Unit of Work | Port that groups a set of operations into one atomic commit or rollback. |
| Adapter | Infrastructure class that implements a domain port against a concrete technology. |
| Service layer | Orchestration layer that wires repositories, analytics and reporting into a workflow. |
| Allocator | Component that maps a return history to portfolio weights. |
| Port | Abstract interface the domain defines and infrastructure implements. |

## Allocation

| Term | Meaning |
| --- | --- |
| HRP | Hierarchical Risk Parity, a tree-based allocator that avoids inverting the covariance. |
| HERC | Hierarchical Equal Risk Contribution, an HRP variant that equalises risk across clusters. |
| NCO | Nested Clustered Optimisation, which optimises within clusters then across them. |
| Risk parity (ERC) | Allocation that equalises each asset's contribution to total volatility. |
| Black-Litterman | Bayesian blend of an equilibrium prior with investor views. |
| Mean-CVaR | Allocation that minimises Conditional Value-at-Risk subject to a return floor. |
| Mean-EVaR | Allocation that minimises Entropic Value-at-Risk subject to a return floor. |
| Resampled (Michaud) | Allocation averaged over bootstrap resamples to reduce estimation noise. |

## Risk

| Term | Meaning |
| --- | --- |
| VaR | Value-at-Risk, the loss quantile at a given tail level. |
| CVaR | Conditional Value-at-Risk, the mean loss beyond the VaR threshold. |
| EVaR | Entropic Value-at-Risk, the tightest coherent upper bound on CVaR. |
| CDaR | Conditional Drawdown-at-Risk, the mean drawdown beyond its tail threshold. |
| Spectral measure | Coherent risk built from a weighting function over loss quantiles. |
| EVT | Extreme Value Theory, a tail model fitted to threshold exceedances. |
| PSR and DSR | Probabilistic and Deflated Sharpe Ratio, significance-adjusted performance measures. |
| Risk contribution | The share of total portfolio risk attributable to one asset. |

## Estimation

| Term | Meaning |
| --- | --- |
| Covariance shrinkage | Pulling the sample covariance toward a structured target to reduce error. |
| Nonlinear shrinkage | Shrinking each sample eigenvalue by its own analytically optimal amount. |
| Denoising | Replacing noise-bulk eigenvalues with their average to stabilise the covariance. |
| Factor model | Covariance built from a small set of common factors plus idiosyncratic risk. |
| Copula | A model of the dependence structure separated from the marginal distributions. |

## Stochastic processes

| Term | Meaning |
| --- | --- |
| SDE | Stochastic Differential Equation governing a price or rate path. |
| GBM | Geometric Brownian Motion, the lognormal price process. |
| OU | Ornstein-Uhlenbeck, a mean-reverting process. |
| CIR | Cox-Ingersoll-Ross, a mean-reverting non-negative process. |
| Euler-Maruyama | First-order discretisation scheme for an SDE. |
| Antithetic variates | A variance-reduction technique that pairs each path with its mirror. |
