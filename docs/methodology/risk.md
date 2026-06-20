# Risk

Derivations for the measures and tests in `portfolio_optimisation.risk`. The
shared notation is defined on the [methodology index](index.md). Losses are
\(L = -r^\top w\), and \(\alpha \in (0,1)\) is the tail mass.

## Coherence

A risk measure \(\rho\) is coherent when it is monotone, sub-additive, positively
homogeneous and translation invariant. Sub-additivity,
\(\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)\), is the property that rewards
diversification, and it is the one Value-at-Risk lacks. The measures below are
constructed to be coherent, which is what makes them suitable as optimisation
objectives.

## Value-at-Risk and Conditional Value-at-Risk

Value-at-Risk is the loss quantile \(\mathrm{VaR}_\alpha(L) = \inf\{ x :
\mathbb{P}(L > x) \le \alpha \}\). It is not sub-additive, so a portfolio can
appear less risky than the sum of its parts. Conditional Value-at-Risk repairs
this by averaging the tail,

\[
\mathrm{CVaR}_\alpha(L)
= \frac{1}{\alpha} \int_0^{\alpha} \mathrm{VaR}_u(L)\, du
= \mathbb{E}\big[ L \mid L \ge \mathrm{VaR}_\alpha(L) \big],
\]

the second equality holding for continuous \(L\). Conditional Value-at-Risk is
coherent and is the objective derived for the
[mean-CVaR allocator](allocation.md#mean-cvar).

## Entropic Value-at-Risk

Entropic Value-at-Risk follows from the Chernoff bound. For any \(z > 0\),
\(\mathbb{P}(L \ge x) \le e^{-zx} \mathbb{E}[e^{zL}]\), and inverting the bound at
level \(\alpha\) and optimising over \(z\) gives

\[
\mathrm{EVaR}_\alpha(L)
= \inf_{z > 0}\ z \ln\!\left( \frac{M_L(z)}{\alpha} \right),
\qquad
M_L(z) = \mathbb{E}\big[ e^{zL} \big].
\]

It dominates Conditional Value-at-Risk,
\(\mathrm{CVaR}_\alpha \le \mathrm{EVaR}_\alpha\), and is the tightest bound
expressible through the moment generating function. The convex programme that
minimises it is derived for the
[mean-EVaR allocator](allocation.md#mean-evar).

## Spectral and distortion measures

A spectral risk measure weights the loss quantiles by an admissible function
\(\phi : [0,1] \to \mathbb{R}_+\) that is non-decreasing and integrates to one,

\[
M_\phi(L) = \int_0^1 \phi(p)\, \mathrm{VaR}_p(L)\, dp .
\]

Putting more weight on the upper quantiles encodes risk aversion, and the
monotonicity of \(\phi\) is exactly the condition that makes \(M_\phi\) coherent.
Conditional Value-at-Risk is the special case
\(\phi(p) = \tfrac{1}{\alpha} \mathbf{1}\{ p \ge 1 - \alpha \}\). A distortion
measure instead reweights the loss distribution through a concave distortion
\(g\), and the Wang transform \(g(u) = \Phi(\Phi^{-1}(u) - \lambda)\) shifts
probability mass into the tail by \(\lambda\) standard normal units, recovering
the mean loss at \(\lambda = 0\).

## Extreme value theory

The historical sample is thin in the tail, so quantiles far out are unreliable.
The Pickands-Balkema-de Haan theorem states that, for a broad class of
distributions, the excess over a high threshold \(u\) converges to a generalised
Pareto distribution as \(u\) grows,

\[
\mathbb{P}(L - u \le y \mid L > u) \to
G_{\xi, \beta}(y) = 1 - \left( 1 + \frac{\xi y}{\beta} \right)^{-1/\xi},
\qquad \xi \neq 0 .
\]

Fitting \((\xi, \beta)\) to the \(N_u\) exceedances out of \(n\) observations
gives the peaks-over-threshold tail estimator
\(\mathbb{P}(L > x) = \tfrac{N_u}{n} \big( 1 + \xi (x - u)/\beta \big)^{-1/\xi}\),
and inverting it yields the tail risk

\[
\mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}
\left[ \left( \frac{\alpha\, n}{N_u} \right)^{-\xi} - 1 \right],
\qquad
\mathrm{ES}_\alpha = \frac{\mathrm{VaR}_\alpha}{1 - \xi} +
\frac{\beta - \xi u}{1 - \xi}, \quad \xi < 1 .
\]

When only the tail index is needed, the Hill estimator gives it directly from
the \(k\) largest order statistics,

\[
\hat{\xi} = \frac{1}{k} \sum_{i=1}^{k}
\ln \frac{L_{(i)}}{L_{(k+1)}},
\]

for \(\xi > 0\), where \(L_{(1)} \ge \dots \ge L_{(n)}\) are the ordered losses.

## Conditional volatility

Risk is not constant in time. A GARCH model lets the conditional variance evolve
with past shocks. The GARCH(1,1) recursion is

\[
\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2,
\]

stationary when \(\alpha + \beta < 1\). Two asymmetric variants capture the
leverage effect, by which negative shocks raise volatility more than positive
ones. The GJR-GARCH adds a sign term,
\(\sigma_t^2 = \omega + (\alpha + \gamma\, \mathbf{1}\{\varepsilon_{t-1} < 0\})
\varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2\), and the EGARCH models the log
variance,
\(\ln \sigma_t^2 = \omega + \alpha (|z_{t-1}| - \mathbb{E}|z|) +
\gamma\, z_{t-1} + \beta \ln \sigma_{t-1}^2\) with standardised residual
\(z_t = \varepsilon_t / \sigma_t\). The one-step forecasts give time-varying risk
\(\mathrm{VaR}_{\alpha,t} = \sigma_t\, q_\alpha\) and the matching Expected
Shortfall, where \(q_\alpha\) is the lower-tail quantile of the chosen
innovation law, Gaussian or Student-t.

## Risk contributions

Total volatility decomposes exactly by Euler's theorem, since
\(\sigma(w) = \sqrt{w^\top \Sigma w}\) is homogeneous of degree one,

\[
\sigma(w) = \sum_i \underbrace{w_i\, \frac{(\Sigma w)_i}{\sqrt{w^\top \Sigma w}}}
_{\text{component contribution } \mathrm{RC}_i},
\qquad
\frac{\partial \sigma}{\partial w_i}
= \frac{(\Sigma w)_i}{\sqrt{w^\top \Sigma w}}
\ \text{(marginal contribution)} .
\]

The percentage contributions \(\mathrm{RC}_i / \sigma(w)\) sum to one, and their
concentration, measured by the Herfindahl index
\(\sum_i (\mathrm{RC}_i / \sigma(w))^2\), is minimised by the
[risk-parity allocator](allocation.md#risk-parity).

## Probabilistic and deflated Sharpe ratio

A point estimate of the Sharpe ratio ignores its sampling error, which grows
with skew and heavy tails. The probabilistic Sharpe ratio reports the
probability that the true ratio exceeds a benchmark \(SR^\star\),

\[
\widehat{\mathrm{PSR}}(SR^\star)
= \Phi\!\left(
\frac{(\widehat{SR} - SR^\star)\sqrt{T - 1}}
{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} +
\tfrac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2}}
\right),
\]

with skewness \(\hat{\gamma}_3\) and kurtosis \(\hat{\gamma}_4\) of the returns.
The deflated Sharpe ratio sets \(SR^\star\) to the expected maximum of \(N\)
trials under the null, which corrects for the selection bias of choosing the
best strategy from many.

## Backtesting Value-at-Risk

A VaR model is validated on the violation sequence
\(I_t = \mathbf{1}\{ L_t > \mathrm{VaR}_{\alpha,t} \}\). Three tests cover the two
properties a good model must have, correct frequency and independence of
violations.

**Unconditional coverage.** Kupiec's proportion-of-failures test compares the
observed violation rate \(\hat{\pi} = x / n\) with \(\alpha\) through the
likelihood ratio

\[
\mathrm{LR}_{\mathrm{uc}}
= -2 \ln \frac{(1 - \alpha)^{n - x} \alpha^{x}}
{(1 - \hat{\pi})^{n - x} \hat{\pi}^{x}}
\ \sim\ \chi^2_1 .
\]

**Independence.** Christoffersen's test models the violations as a two-state
Markov chain with transition counts \(n_{ij}\) and rejects when a violation
predicts the next, again through a likelihood ratio that is \(\chi^2_1\) under the
null of independence. The conditional-coverage test adds the two statistics,
\(\mathrm{LR}_{\mathrm{cc}} = \mathrm{LR}_{\mathrm{uc}} + \mathrm{LR}_{\mathrm{ind}}
\sim \chi^2_2\), testing both properties jointly.

**Expected Shortfall.** Value-at-Risk frequency says nothing about the size of
the tail losses, which Expected Shortfall forecasts. The Acerbi-Szekely test
averages the realised tail loss relative to its forecast,

\[
Z_2 = 1 + \frac{1}{n \alpha} \sum_{t=1}^{n}
\frac{L_t}{\mathrm{ES}_{\alpha,t}}\, I_t,
\]

which has expectation zero when the Expected Shortfall forecast is correct. A
materially positive value signals that the model underestimates the tail, and
significance is assessed by simulation under the forecasting model.
