# Allocation

Derivations for the allocators in `portfolio_optimisation.optim`. The shared
notation is defined on the [methodology index](index.md).

## Hierarchical risk parity

Hierarchical risk parity avoids inverting the covariance, which is the source of
the instability in mean-variance weights when $\Sigma$ is near singular. It
proceeds in three stages.

**Tree.** Convert the correlation $\rho_{ij}$ into the distance
$d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})}$, a proper metric on the unit
sphere, then cluster with agglomerative linkage to obtain a binary tree.

**Quasi-diagonalisation.** Reorder the assets by the tree leaves so that similar
assets sit adjacent. The reordered covariance concentrates its mass near the
diagonal, which makes the recursive split below behave like a sequence of
independent sub-problems.

**Recursive bisection.** Split each cluster $C$ into halves $C_1, C_2$. For a
cluster, the inverse-variance weights $v_i = \sigma_i^{-2} / \sum_{j \in C}
\sigma_j^{-2}$ give the cluster variance
$\tilde{V}(C) = v^\top \Sigma_C\, v$. Allocate between the two halves by the
inverse of their variances,

$$
\beta = \frac{\tilde{V}(C_1)}{\tilde{V}(C_1) + \tilde{V}(C_2)},
\qquad
w_{C_1} \leftarrow (1 - \beta)\, w_{C_1},
\quad
w_{C_2} \leftarrow \beta\, w_{C_2},
$$

so the half with the larger variance receives the smaller multiplier. Recursing
to singletons yields weights on the simplex without a single matrix inversion.

## Hierarchical equal risk contribution

The hierarchical equal risk contribution allocator keeps the tree of the
previous method but replaces the inverse-variance split with an equal-risk
split under a chosen risk measure $\mathcal{R}$, either variance or
Conditional Value-at-Risk. For a node with children $C_1, C_2$ the split is

$$
\alpha_{\text{split}}
= \frac{\mathcal{R}(C_1)}{\mathcal{R}(C_1) + \mathcal{R}(C_2)},
$$

and the child risks are measured under the same $\mathcal{R}$, so the ratio is
scale-consistent. Using Conditional Value-at-Risk makes the allocation sensitive
to tail co-movement that variance ignores.

## Nested clustered optimisation

Nested clustered optimisation reduces the conditioning of the mean-variance
solution by solving it twice on smaller, better-conditioned problems. Cluster
the assets into $K$ groups. Within cluster $k$ solve the minimum-variance
problem to obtain intra-cluster weights $w^{(k)}$. Form the reduced covariance
across clusters,

$$
\Sigma^{\text{red}}_{kl} = w^{(k)\top} \Sigma_{kl}\, w^{(l)},
$$

solve the minimum-variance problem on $\Sigma^{\text{red}}$ for the inter-
cluster weights $\omega$, and combine as
$w_i = \omega_k\, w^{(k)}_i$ for asset $i$ in cluster $k$. Because each
sub-problem inverts a smaller and better-conditioned matrix, the estimation
error in $\Sigma$ is amplified far less than in the full inversion.

## Risk parity

A risk-parity portfolio equalises the contribution of each asset to total
volatility. With portfolio volatility $\sigma(w) = \sqrt{w^\top \Sigma w}$,
Euler's theorem on the homogeneous-degree-one function $\sigma$ gives the
exact decomposition

$$
\sigma(w) = \sum_i w_i \frac{\partial \sigma}{\partial w_i}
= \sum_i \frac{w_i (\Sigma w)_i}{\sqrt{w^\top \Sigma w}},
$$

so the risk contribution of asset $i$ is
$\mathrm{RC}_i = w_i (\Sigma w)_i / \sqrt{w^\top \Sigma w}$. The budgeted
risk-parity condition is $\mathrm{RC}_i \propto b_i$ for a risk budget $b$ on
the simplex, equivalently $w_i (\Sigma w)_i = b_i\, w^\top \Sigma w$.

Solving the nonlinear system directly is awkward. The convex reformulation
removes the budget constraint. Consider

$$
\min_{y > 0}\ \tfrac{1}{2} y^\top \Sigma y - \sum_i b_i \ln y_i .
$$

The objective is strictly convex, since $\Sigma$ is positive definite and
$-\ln$ is convex, so the stationary point is the unique global minimum. Setting
the gradient to zero,

$$
\Sigma y - b \oslash y = 0
\quad\Longleftrightarrow\quad
(\Sigma y)_i = \frac{b_i}{y_i}
\quad\Longleftrightarrow\quad
y_i (\Sigma y)_i = b_i .
$$

Normalising $w = y / \mathbf{1}^\top y$ scales every risk contribution by the
same constant, so $w_i (\Sigma w)_i \propto b_i$ holds and the
equal-risk-contribution condition is met. Uniqueness of the minimiser makes the
solution independent of the solver.

## Mean-variance baseline

The mean-variance frontier solves
$\min_w \tfrac{1}{2} w^\top \Sigma w$ subject to $\hat{\mu}^\top w \ge \mu_0$
and $\mathbf{1}^\top w = 1$. The Lagrangian is quadratic and the solution is
affine in $\mu_0$, which traces the frontier. The dependence on
$\Sigma^{-1} \hat{\mu}$ is what makes the weights sensitive to estimation
error and motivates the shrinkage, robust and hierarchical alternatives.

## Mean-CVaR

Conditional Value-at-Risk at tail level $\alpha$ is the mean loss in the worst
$\alpha$ fraction of outcomes. The Rockafellar-Uryasev identity expresses it as
a minimisation,

$$
\mathrm{CVaR}_\alpha(L)
= \min_{\zeta \in \mathbb{R}}
\left\{ \zeta + \frac{1}{\alpha}\, \mathbb{E}\big[(L - \zeta)^+\big] \right\},
$$

where the minimiser $\zeta^\star$ equals the Value-at-Risk. On the empirical
sample with $L_t = -r_t^\top w$ and uniform weights $1/T$, introduce slacks
$u_t \ge (L_t - \zeta)^+$ to linearise the positive part, giving the programme

$$
\min_{w, \zeta, u}\ \zeta + \frac{1}{\alpha T} \sum_{t=1}^{T} u_t
\quad\text{s.t.}\quad
u_t \ge -r_t^\top w - \zeta,\ \ u_t \ge 0,
$$

with $w$ in the shared constraint set. The objective and constraints are
linear, so this is a linear programme and the solution is a global optimum. The
divisor is $\alpha T$, the mass of the tail being averaged.

## Mean-EVaR

Entropic Value-at-Risk is the tightest coherent upper bound on Conditional
Value-at-Risk obtainable from the Chernoff bound on the moment generating
function,

$$
\mathrm{EVaR}_\alpha(L)
= \inf_{z > 0}\ z \ln\!\left( \frac{1}{\alpha}\,
\frac{1}{T} \sum_{t=1}^{T} e^{L_t / z} \right).
$$

The joint problem over $w$ and $z$ is convex and admits an
exponential-cone form. Introduce $t$ and per-sample $u_t$ with the
perspective constraints $u_t \ge z\, e^{(L_t - t)/z}$, the exponential cone, and
$\sum_t u_t \le z$. Then

$$
\sum_t e^{(L_t - t)/z} \le 1
\quad\Longleftrightarrow\quad
t \ge z \ln \sum_t e^{L_t / z},
$$

so minimising $t - z \ln(\alpha T)$ reproduces
$z \ln\!\big( (\alpha T)^{-1} \sum_t e^{L_t/z} \big)$, the empirical
Entropic Value-at-Risk. Because the measure is positively homogeneous in the
loss, scaling the per-sample losses by a constant rescales the objective without
moving the optimal $w$, which the implementation exploits to condition the
cone solver.

## Conditional drawdown at risk

For the cumulative return path $P_t = \sum_{s \le t} r_s^\top w$ and running
maximum $M_t = \max_{s \le t} P_s$, the drawdown is $D_t = M_t - P_t \ge 0$.
Conditional Drawdown-at-Risk is the Rockafellar-Uryasev average of the drawdown
beyond its tail threshold,

$$
\min_{w, \zeta, u, m}\
\zeta + \frac{1}{\alpha T} \sum_{t=1}^{T} u_t
\quad\text{s.t.}\quad
u_t \ge (m_t - P_t) - \zeta,\ \ u_t \ge 0,
$$

with the running maximum linearised by a non-decreasing auxiliary variable
$m_t \ge m_{t-1}$, $m_t \ge P_t$, $m_0 \ge 0$. The averaging divisor is
$\alpha T$, consistent with the worst-$\alpha$ drawdown that the evaluation
metric reports. The programme is linear and shares the constraint set with the
other mean-risk allocators.

## Second-order stochastic dominance

A portfolio return $X$ second-order stochastically dominates a benchmark
$Y$ when $\mathbb{E}[U(X)] \ge \mathbb{E}[U(Y)]$ for every increasing concave
utility $U$. Equivalently, for every threshold $\eta$,

$$
\mathbb{E}\big[(\eta - X)^+\big] \le \mathbb{E}\big[(\eta - Y)^+\big].
$$

On a discrete panel the continuum of thresholds collapses to the benchmark
realisations $\eta_i = Y_i$. Maximising the expected return subject to
dominance is the linear programme

$$
\max_{w}\ \hat{\mu}^\top w
\quad\text{s.t.}\quad
\frac{1}{T} \sum_t u_{t,i} \le s_i(Y),\ \
u_{t,i} \ge \eta_i - r_t^\top w,\ \
u_{t,i} \ge 0,
$$

where $s_i(Y) = \tfrac{1}{T} \sum_t (\eta_i - Y_t)^+$ is the benchmark lower
partial moment at $\eta_i$. The dominance constraints are linear in $w$, so
the problem stays a linear programme.

## Polynomial goal programming over four moments

When returns are skewed and fat-tailed, variance is an incomplete risk summary.
The third and fourth central co-moments are the tensors

$$
M_3 = \mathbb{E}\big[(r - \mu)(r - \mu)^\top \otimes (r - \mu)^\top\big],
\qquad
M_4 = \mathbb{E}\big[(r - \mu)(r - \mu)^\top \otimes (r - \mu)^\top \otimes
(r - \mu)^\top\big],
$$

estimated by the sample averages of the outer products, which the
implementation forms with `einsum`. The portfolio moments are the contractions
$s(w) = w^\top \hat{\mu}$, $v(w) = w^\top \Sigma w$,
$\text{sk}(w) = w^\top M_3 (w \otimes w)$ and
$\text{ku}(w) = w^\top M_4 (w \otimes w \otimes w)$. Polynomial goal
programming maximises mean and skewness while minimising variance and kurtosis
by minimising the weighted relative deviations from each moment's aspiration
level $g$,

$$
\min_{w \in \Delta}\
\sum_{k} \lambda_k
\left| \frac{g_k - f_k(w)}{g_k} \right|^{p},
$$

where $f_k$ ranges over the four moments and $\lambda_k$ encodes the investor
preference. The objective is nonlinear, so a general nonlinear solver is used.

## Black-Litterman

Black-Litterman blends a market-equilibrium prior with subjective views in a
Bayesian update. Reverse optimisation recovers the prior mean from the market
weights $w_{\text{mkt}}$ and a risk-aversion $\delta$,

$$
\pi = \delta\, \Sigma\, w_{\text{mkt}}.
$$

The prior is $\mu \sim \mathcal{N}(\pi, \tau \Sigma)$. A set of views is
$P \mu = Q + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \Omega)$.
Conjugacy gives the posterior mean

$$
\mu_{\text{BL}}
= \big[ (\tau \Sigma)^{-1} + P^\top \Omega^{-1} P \big]^{-1}
\big[ (\tau \Sigma)^{-1} \pi + P^\top \Omega^{-1} Q \big],
$$

a precision-weighted average of prior and views, with the posterior parameter
covariance $\big[ (\tau \Sigma)^{-1} + P^\top \Omega^{-1} P \big]^{-1}$ added to
$\Sigma$ for the posterior return covariance. Feeding $\mu_{\text{BL}}$ into
the mean-variance step replaces noisy sample means with a shrunk, view-adjusted
estimate.

## Resampled efficiency

Sample estimates are noisy, so a single optimisation overfits one draw. Michaud
resampling averages over the sampling distribution. For $b = 1, \dots, B$,
draw a bootstrap or parametric resample of the returns, estimate
$(\hat{\mu}^{(b)}, \Sigma^{(b)})$, and solve the chosen optimisation to get
$w^{(b)}$. The resampled portfolio is the average

$$
\bar{w} = \frac{1}{B} \sum_{b=1}^{B} w^{(b)},
$$

which remains on the simplex as a convex combination of feasible points. The
averaging shrinks the weights toward the centre of the efficient region and
reduces turnover relative to the single-shot solution.

## Robust mean-variance

Robust optimisation guards against estimation error in $\hat{\mu}$ by
optimising the worst case over an uncertainty set. With a box set
$\mathcal{U} = \{ \mu : |\mu_i - \hat{\mu}_i| \le \delta_i \}$, the inner
worst case has a closed form,

$$
\min_{\mu \in \mathcal{U}} \mu^\top w
= \hat{\mu}^\top w - \delta^\top |w|,
$$

since each coordinate is minimised independently at the box edge. The robust
problem is therefore

$$
\max_{w}\ \hat{\mu}^\top w - \delta^\top |w| - \frac{\gamma}{2} w^\top \Sigma w,
$$

a concave programme. The penalty $\delta^\top |w|$ discourages large positions
in assets whose mean is poorly estimated, which is the intended robustness.
