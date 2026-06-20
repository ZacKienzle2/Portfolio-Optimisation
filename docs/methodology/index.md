# Methodology

These pages derive the methods implemented in the package from first
principles. Each derivation states the model, the assumptions, the optimisation
problem solved, and the properties that justify the implementation choices. The
notation is shared across the pages and is collected below.

## Shared notation

A universe of \(N\) assets is observed over \(T\) periods. The return matrix is
\(R \in \mathbb{R}^{T \times N}\) with row \(r_t\) the cross-section at time
\(t\) and column \(R_{\cdot i}\) the history of asset \(i\). Portfolio weights
are \(w \in \mathbb{R}^N\); the fully-invested long-only simplex is
\(\Delta = \{ w : w \ge 0,\ \mathbf{1}^\top w = 1 \}\). The sample mean is
\(\hat{\mu} = \tfrac{1}{T} \sum_t r_t\) and the sample covariance is
\(\Sigma \in \mathbb{R}^{N \times N}\). The portfolio return is \(r_t^\top w\)
and the portfolio loss is \(L_t = -r_t^\top w\). A tail level \(\alpha \in (0,1)\)
denotes the probability mass in the loss tail, so \(\alpha = 0.05\) is the worst
five percent.

## Areas

- [Allocation](allocation.md) covers the hierarchical, risk-parity, mean-risk,
  stochastic-dominance, higher-moment, view-based and robust allocators.
- Risk, estimation, stochastic processes and econometrics follow the same
  structure and are derived on their own pages.
