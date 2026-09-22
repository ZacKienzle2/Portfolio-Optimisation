# 22. Library numerics and measured formulations

- Status: Accepted
- Date: 2026-09-22

## Context

A review of the implementation against its dependencies found routines written
by hand that a dependency already provides, formulations whose cost grows faster
than the problem requires, and one numerical defect. Each finding was timed with
pyperf before and after the change, and each replacement was checked against the
version it replaced. The accompanying writeup reviews the literature behind
every item.

## Decision

Where a dependency implements a routine, the package calls it.

- The HRP seriation is SciPy's `leaves_list`, replacing a quadratic list splice,
  and HRP, HERC and NCO share one clustering module.
- Correlation from covariance is statsmodels' `cov2corr`, the clipped
  correlation is `corr_clipped`, and the copula correlation is the Kendall-tau
  inversion of statsmodels' `StudentTCopula.fit_corr_param`.
- The Entropic Value-at-Risk uses `scipy.special.logsumexp`, the Sharpe moments
  `scipy.stats.skew` and `kurtosis`, and the bootstrap loop arch's
  `StationaryBootstrap.apply`.
- The CVaR and CDaR programmes state their tail with cvxpy's `cvar` atom.
- Whole-share allocation is a mixed-integer programme for `scipy.optimize.milp`,
  and the efficient frontier a cvxpy parameterised programme, which removes
  PyPortfolioOpt.
- The Marchenko-Pastur fit reads a kernel density that statsmodels computes once
  by Silverman's binned FFT, rather than summing every kernel at every grid
  point for each candidate variance.
- yfinance retries transient errors itself through its `retries` setting, which
  removes the retry module.
- HERC cuts the dendrogram with SciPy's `fcluster` at the number of clusters the
  gap statistic chooses, following Raffinot (2018) in place of a bisection of
  the seriated order.
- The Ornstein-Uhlenbeck fit is statsmodels' `AutoReg`, whose least-squares
  estimates give the closed-form maximum likelihood estimators of Tang and Chen
  (2009), and the geometric Brownian motion fit is the sample moments of the log
  returns. This removes pymle.
- Every cvxpy programme names Clarabel, so cvxpy no longer retries imports of
  absent solvers on each solve or hands quadratic programmes to OSQP.

Where the formulation was the cost, it is replaced by an equivalent one.

- Second-order stochastic dominance is solved by cutting planes over the
  tail-sum characterisation, on HiGHS through `scipy.optimize.linprog`, rather
  than the programme with one variable per pair of scenarios.
- CDaR follows the drawdown recursion of Chekhlov, Uryasev and Zabarankin (2005,
  Proposition 4.1), two constraint rows per date rather than three.
- Minimum EVaR is the smooth programme of Ahmadi-Javid and Fallah-Tafti (2019)
  in the weights and the EVaR parameter, solved by SLSQP, in place of one
  exponential cone per scenario. When `alpha T <= 1` it is the minimax rule and
  is solved as the CVaR programme at `alpha = 1/T`.
- The constraint set is encoded once as SciPy's `LinearConstraint` and `Bounds`,
  each L1 budget split into a slack block, which `build` renders for cvxpy and
  SLSQP reads directly.
- EVaR evaluation searches a bracket derived from Ahmadi-Javid (2012,
  Proposition 3.2) on centred and scaled losses, where a fixed bound on the
  parameter overstated the measure for small losses.
- Black-Litterman solves one `k x k` system in the Woodbury form.
- The four-moment objective evaluates skewness and kurtosis from the portfolio
  return series, never forming the co-moment tensors, and the tensors themselves
  are one matrix product each.
- Resampled efficiency draws the sample mean and a scaled Wishart covariance
  directly and solves the resamples in one batched call.
- Whole-share allocation minimises the shortfall below target, which equals half
  the deviation plus leftover cash for weights summing to one, over steps from
  rounding down, in units of the dearest share. The deviation form in units of
  the budget stopped short of the optimum and let cents fall below the solver's
  feasibility tolerance.
- The violation transitions are the joint count and two margins of a 2 x 2
  table, each a `count_nonzero` over one byte per day.
- The Ornstein-Uhlenbeck simulator samples the exact Gaussian transition through
  `scipy.signal.lfilter`, superseding the Euler step recorded in ADR 10 for that
  process, and the Merton simulator draws every increment at once.

The Merton jump term multiplied one normal draw by the jump count. The sum of
`N` log jumps then had a standard deviation of `N` times the jump volatility
rather than `sqrt(N)` times. It now draws the sum from its exact law.

## Consequences

- Stochastic dominance on 250 scenarios takes 16 ms against 4.3 s, and 2000
  scenarios of 30 assets, which the pairwise programme could not hold, take 1.6
  s.
- CDaR on 1000 dates and 30 assets takes 64 ms against 191 ms.
- Minimum EVaR on 30 assets takes 12.6 ms against 64.5 ms at 1000 scenarios and
  47.9 ms against 2.65 s at 20000. Clarabel stalled on about one cone programme
  in six hundred at its default step.
- The tests of these changes are written by `nox -s generate` against the
  formulation each paper states, kept in `baselines.py`, and edited only to
  narrow the strategies and set a tolerance. The mean-semideviation test
  compares objectives, since a linear programme's optimal weights need not be
  unique.
- cvxpy 1.9.3 is the lowest version allowed, since earlier versions took the
  shape of a sum from an uninitialised array and warned at random.
- Three routines are slower. EVaR takes 1.4 ms against 0.5 ms, all of it the
  per-call dispatch of SciPy 1.18's `logsumexp`. Whole-share allocation takes 42
  ms against 4.7 ms, spent in HiGHS's root heuristics, and in exchange is
  certified optimal. HERC on 200 assets takes 58 ms against 10 ms, the cost of
  the gap statistic's twenty reference clusterings, most of it SciPy validating
  the linkage matrix on every cut.
- Three measured alternatives were not adopted. The L-shaped decomposition of
  the CVaR programme left a six per cent gap after 380 rounds. HiGHS solved the
  scenario programmes more slowly than Clarabel. Limiting OpenBLAS to one thread
  around the gap statistic's reference loop made no difference outside the
  run-to-run spread.
- PyPortfolioOpt, numba, numexpr, ECOS and the direct SCS pin leave the
  dependency set, and cvxpy is needed only for the convex programmes.
- The replaced implementations remain in `portfolio_optimisation.baselines`,
  where equivalence tests compare them with the rewrites.
- Merton paths with more than one jump per step now have the correct dispersion;
  at one step with three expected jumps the terminal mean returns to its
  analytic value.
