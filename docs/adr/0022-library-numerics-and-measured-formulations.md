# 22. Library numerics and measured formulations

- Status: Accepted
- Date: 2026-09-22

## Context

A review of the implementation against its dependencies found routines written
by hand that a dependency already provides, formulations whose cost grows
faster than the problem requires, and one numerical defect. Each finding was
timed with pyperf before and after the change, and each replacement was checked
against the version it replaced. The accompanying writeup reviews the
literature behind every item.

## Decision

Where a dependency implements a routine, the package calls it.

- The HRP seriation is SciPy's `leaves_list`, replacing a quadratic list
  splice, and HRP, HERC and NCO share one clustering module.
- Correlation from covariance is statsmodels' `cov2corr`, the clipped
  correlation is `corr_clipped`, and the copula correlation is the Kendall-tau
  inversion of statsmodels' `StudentTCopula.fit_corr_param`.
- The Entropic Value-at-Risk uses `scipy.special.logsumexp`, the Sharpe
  moments `scipy.stats.skew` and `kurtosis`, and the bootstrap loop arch's
  `StationaryBootstrap.apply`.
- The CVaR and CDaR programmes state their tail with cvxpy's `cvar` atom.
- Whole-share allocation is a mixed-integer programme for
  `scipy.optimize.milp`, and the efficient frontier a cvxpy parameterised
  programme, which removes PyPortfolioOpt.
- The Marchenko-Pastur fit reads a kernel density that statsmodels computes
  once by Silverman's binned FFT, rather than summing every kernel at every
  grid point for each candidate variance.
- yfinance retries transient errors itself through its `retries` setting,
  which removes the retry module.

Where the formulation was the cost, it is replaced by an equivalent one.

- Second-order stochastic dominance is solved by cutting planes over the
  tail-sum characterisation, on HiGHS through `scipy.optimize.linprog`, rather
  than the programme with one variable per pair of scenarios.
- Black-Litterman solves one `k x k` system in the Woodbury form.
- The four-moment objective evaluates skewness and kurtosis from the portfolio
  return series, never forming the co-moment tensors, and the tensors
  themselves are one matrix product each.
- Resampled efficiency draws the sample mean and a scaled Wishart covariance
  directly and solves the resamples in one batched call.
- Whole-share allocation minimises the shortfall below target, which equals
  half the deviation plus leftover cash for weights summing to one, over steps
  from rounding down, in units of the dearest share. The deviation form in
  units of the budget stopped short of the optimum and let cents fall below the
  solver's feasibility tolerance.
- The violation transitions are the joint count and two margins of a 2 x 2
  table, each a `count_nonzero` over one byte per day.
- The Ornstein-Uhlenbeck simulator samples the exact Gaussian transition
  through `scipy.signal.lfilter`, superseding the Euler step recorded in ADR 10
  for that process, and the Merton simulator draws every increment at once.

The Merton jump term multiplied one normal draw by the jump count. The sum of
`N` log jumps then had a standard deviation of `N` times the jump volatility
rather than `sqrt(N)` times. It now draws the sum from its exact law.

## Consequences

- Stochastic dominance on 250 scenarios takes 16 ms against 4.3 s, and 2000
  scenarios of 30 assets, which the pairwise programme could not hold, take
  1.6 s.
- Two routines are slower. EVaR takes 1.7 ms against 0.5 ms, all of it the
  per-call dispatch of SciPy 1.18's `logsumexp`. Whole-share allocation takes
  42 ms against 4.7 ms, spent in HiGHS's root heuristics, and in exchange is
  certified optimal.
- PyPortfolioOpt, numba, numexpr, ECOS and the direct SCS pin leave the
  dependency set, and cvxpy is needed only for the convex programmes.
- The replaced implementations remain in `portfolio_optimisation.baselines`,
  where equivalence tests compare them with the rewrites.
- Merton paths with more than one jump per step now have the correct
  dispersion; at one step with three expected jumps the terminal mean returns to
  its analytic value.
