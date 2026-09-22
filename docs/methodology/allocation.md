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
independent sub-problems. The order is the left-to-right leaf order of the
dendrogram, which SciPy's `leaves_list` returns in linear time.

**Recursive bisection.** Split each cluster $C$ into halves $C_1, C_2$. For a
cluster, the inverse-variance weights
$v_i = \sigma_i^{-2} / \sum_{j \in C}
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

Raffinot's (2018) allocator keeps the correlation-distance dendrogram of the
previous method and changes three things. It cuts the tree into $K$ clusters,
with $K$ chosen by the gap statistic of Tibshirani, Walther and Hastie (2001).
It divides capital along the dendrogram's own splits rather than at the midpoint
of the seriated order. And it holds each cluster by naive risk parity, weights
proportional to $1 / \mathcal{R}_i$ for the asset risk $\mathcal{R}_i$,
volatility or expected shortfall.

The gap statistic embeds the assets by classical scaling of the correlation
matrix $C = Q \Lambda Q^\top$ as the rows of $X = Q \Lambda^{1/2}$, whose
squared distances are $2 (1 - \rho_{ij})$, so clustering $X$ reproduces the
correlation-distance dendrogram. With $W_k$ the pooled within-cluster sum of
squares of the cut into $k$ clusters,

$$
\operatorname{Gap}(k) = \mathbb{E}^\ast [\log W_k] - \log W_k ,
$$

where the expectation is over uniform samples in the box aligned with the
principal components of $X$. The estimate is the smallest $k$ with
$\operatorname{Gap}(k) \ge \operatorname{Gap}(k+1) - s_{k+1}$, where $s_k$ is
the standard deviation of the reference $\log W_k$ times $\sqrt{1 + 1/B}$.

Each of the $K - 1$ splits above the cut divides a node's capital between its
children $C_1, C_2$ so that both contribute equally,

$$
\alpha_{\text{split}}
= \frac{\mathcal{R}(C_2)}{\mathcal{R}(C_1) + \mathcal{R}(C_2)},
$$

with $\mathcal{R}(C)$ the risk of the naive risk parity portfolio of $C$.
Raffinot prints the ratio with $\mathcal{R}(C_1)$ in the numerator, which would
give the riskier child more capital, and the implementation follows the equal
contribution condition instead. Using expected shortfall makes the allocation
sensitive to tail co-movement that variance ignores.

On planted block structures the one-standard-error rule recovers one, two, three
and five blocks of six assets under average linkage. It stops early when the gap
rises in steps smaller than $s_k$, returning two for eight blocks whose gap
curve peaks at eight, and under Ward linkage it returns one for three or more
blocks.

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
cluster weights $\omega$, and combine as $w_i = \omega_k\, w^{(k)}_i$ for asset
$i$ in cluster $k$. Because each sub-problem inverts a smaller and
better-conditioned matrix, the estimation error in $\Sigma$ is amplified far
less than in the full inversion.

## Risk parity

A risk-parity portfolio equalises the contribution of each asset to total
volatility. With portfolio volatility $\sigma(w) = \sqrt{w^\top \Sigma w}$,
Euler's theorem on the homogeneous-degree-one function $\sigma$ gives the exact
decomposition

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

The objective is strictly convex, since $\Sigma$ is positive definite and $-\ln$
is convex, so the stationary point is the unique global minimum. Setting the
gradient to zero,

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

## Maximum diversification

The diversification ratio of a long-only portfolio is its weighted average
volatility over its volatility,

$$
\mathrm{DR}(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}},
$$

with $\sigma$ the vector of asset volatilities. It is at least one, and equals
one only when the holdings are perfectly correlated. Choueifaty and Coignard
(2008) maximise it over the long-only simplex. The ratio does not change when
$w$ is scaled, so Choueifaty, Froidure and Reynier (2013) solve the quadratic
programme

$$
\min_{y \ge 0}\ y^\top \Sigma y
\quad\text{s.t.}\quad
\sigma^\top y = 1, \qquad w = y / \mathbf{1}^\top y,
$$

whose solution is unique when $\Sigma$ is definite. Their core property
characterises the optimum. Every asset held has the same correlation with the
portfolio, and every asset left out is at least as correlated with it. A
generated test compares the programme with SLSQP maximising the ratio of
Choueifaty and Coignard directly. On two hundred assets with the Ledoit-Wolf
covariance the programme solves in 21 ms.

## Mean-variance baseline

The mean-variance frontier solves $\min_w \tfrac{1}{2} w^\top \Sigma w$ subject
to $\hat{\mu}^\top w \ge \mu_0$ and $\mathbf{1}^\top w = 1$. The Lagrangian is
quadratic and the solution is affine in $\mu_0$, which traces the frontier. The
dependence on $\Sigma^{-1} \hat{\mu}$ is what makes the weights sensitive to
estimation error and motivates the shrinkage, robust and hierarchical
alternatives.

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

with $w$ in the shared constraint set. The objective and constraints are linear,
so this is a linear programme and the solution is a global optimum. The divisor
is $\alpha T$, the mass of the tail being averaged. The implementation states
the objective with cvxpy's `cvar` atom, which canonicalises to this programme.
At $\alpha = 1/T$ the average covers the single worst scenario, which is the
minimax model of Young (1998), so that model needs no programme of its own.

## Mean-EVaR

Entropic Value-at-Risk is the tightest coherent upper bound on Conditional
Value-at-Risk obtainable from the Chernoff bound on the moment generating
function,

$$
\mathrm{EVaR}_\alpha(L)
= \inf_{z > 0}\ z \ln\!\left( \frac{1}{\alpha}\,
\frac{1}{T} \sum_{t=1}^{T} e^{L_t / z} \right).
$$

The objective is the perspective of a log-sum-exp composed with the linear
losses, so it is jointly convex in $w$ and $z$. Ahmadi-Javid and Fallah-Tafti
(2019, problem 3.7) observe that it is also differentiable and has $N + 1$
variables whatever the sample size. With $s_t = L_t / z$ and $p$ the softmax of
$s$, the gradient is

$$
\nabla_w = -R^\top p, \qquad
\partial_z = \ln \sum_t e^{s_t} - \ln(\alpha T) - p^\top s,
$$

and SLSQP minimises it over the linear rows and bounds of the shared constraint
set, the L1 budgets split into slack variables. The exponential-cone form it
replaced carries one cone per scenario. On thirty assets it took 64.5 ms at a
thousand scenarios and 2.65 s at twenty thousand, against 12.6 ms and 47.9 ms.
The returns are divided by their standard deviation, which positive homogeneity
allows, so $z$ is of order one.

When $\alpha T \le 1$ every scenario carries probability at least $\alpha$, so
the measure of every portfolio is its largest loss, the limit of Ahmadi-Javid
(2012, Proposition 3.2). The infimum over $z > 0$ is then not attained, and the
programme is the minimax rule of Young (1998), which the implementation solves
as the Mean-CVaR programme at $\alpha = 1/T$.

## Conditional drawdown at risk

For the cumulative return path $P_t = \sum_{s \le t} r_s^\top w$ and running
maximum $M_t = \max(0, \max_{s \le t} P_s)$, the drawdown is
$D_t = M_t - P_t \ge 0$. Conditional Drawdown-at-Risk is the Rockafellar-Uryasev
average of the drawdown beyond its tail threshold. The drawdown obeys the
recursion $D_t = \max(D_{t-1} - r_t^\top w, 0)$ with $D_0 = 0$, which
Proposition 4.1 of Chekhlov, Uryasev and Zabarankin (2005) turns into the linear
programme

$$
\min_{w, \zeta, u, z}\
\zeta + \frac{1}{\alpha T} \sum_{t=1}^{T} z_t
\quad\text{s.t.}\quad
z_t \ge u_t - \zeta,\ \ u_t \ge u_{t-1} - r_t^\top w,\ \
z_t, u_t \ge 0,\ \ u_0 = 0 .
$$

Each row carries one period's return, where the formulation with a running-peak
variable $m_t \ge P_t$ carries the cumulative one and needs a third row per
date. At a thousand dates and thirty assets the recursion solves in 64 ms
against 173 ms, to the same optimum. The averaging divisor is $\alpha T$,
consistent with the worst-$\alpha$ drawdown that the evaluation metric reports,
and the programme shares the constraint set with the other mean-risk allocators.

## Second-order stochastic dominance

A portfolio return $X$ second-order stochastically dominates a benchmark $Y$
when $\mathbb{E}[U(X)] \ge \mathbb{E}[U(Y)]$ for every increasing concave
utility $U$. Equivalently, for every threshold $\eta$,

$$
\mathbb{E}\big[(\eta - X)^+\big] \le \mathbb{E}\big[(\eta - Y)^+\big].
$$

On a panel of $T$ equally likely scenarios the condition is equivalent to
dominance of the tail sums: writing $x_{(1)} \le \dots \le x_{(T)}$ for the
ordered outcomes,

$$
\sum_{i \le k} x_{(i)} \ge \sum_{i \le k} y_{(i)}, \qquad k = 1, \dots, T.
$$

The tail sum of the portfolio is the minimum of $\sum_{t \in J} r_t^\top w$ over
subsets $J$ of size $k$, so maximising the expected return subject to dominance
is the linear programme

$$
\max_{w}\ \hat{\mu}^\top w
\quad\text{s.t.}\quad
\sum_{t \in J} r_t^\top w \ge \sum_{i \le k} y_{(i)}
\ \ \text{for every } J \text{ with } |J| = k,
$$

with exponentially many constraints of which few bind. The implementation adds
them by cutting planes. Each round solves the master problem with HiGHS, sorts
the current portfolio's scenarios, and for the most violated tail sums adds the
cut whose subset is the $k$ worst scenarios, computed for every $k$ at once from
one cumulative sum. The master problem has $N$ variables, where the formulation
with one slack per pair of scenario and threshold has $T^2$, and on 250
scenarios the cutting planes reach the same optimum in 25 ms against 4.3 s.

## Mean-semideviation

Variance penalises gains and losses alike, so a mean-variance choice can be
dominated in the second order. Ogryczak and Ruszczynski (1999) measure risk by
the semideviations below the mean,

$$
\bar{\delta}_X = \mathbb{E}\big[(\mu_X - X)^+\big], \qquad
\bar{\sigma}_X = \Big(\mathbb{E}\big[\big((\mu_X - X)^+\big)^2\big]\Big)^{1/2},
$$

and show that a maximiser of $\mu_X - \lambda \bar{\delta}_X$ (their
Corollary 4) or of $\mu_X - \lambda \bar{\sigma}_X$ (Corollary 8) is efficient
under second-order stochastic dominance when $0 < \lambda \le 1$, apart from
ties in mean and semideviation. Neither bound can be raised for general
distributions. The deviations above and below the mean have equal expectation,
so the absolute semideviation is half the mean absolute deviation, and the first
model is the Konno-Yamazaki programme with the trade-off halved (Mansini,
Ogryczak and Speranza, 2003). On $T$ equally likely scenarios it is a linear
programme with $T$ shortfall variables. The standard semideviation replaces
their mean with a Euclidean norm, a second-order cone. A generated test compares
both with the deviations written as variables, the absolute model as the linear
programme of Mansini, Ogryczak and Speranza on HiGHS and the standard model by
SLSQP. On a thousand scenarios and thirty assets the two programmes solve in 36
ms and 47 ms.

## Polynomial goal programming over four moments

When returns are skewed and fat-tailed, variance is an incomplete risk summary.
The third and fourth central co-moments are the tensors

$$
M_3 = \mathbb{E}\big[(r - \mu)(r - \mu)^\top \otimes (r - \mu)^\top\big],
\qquad
M_4 = \mathbb{E}\big[(r - \mu)(r - \mu)^\top \otimes (r - \mu)^\top \otimes
(r - \mu)^\top\big],
$$

estimated by the sample averages of the outer products, which the implementation
forms as one matrix product with the row-wise Kronecker square of the centred
returns. The portfolio moments are the contractions $s(w) = w^\top \hat{\mu}$,
$v(w) = w^\top \Sigma w$, $\text{sk}(w) = w^\top M_3 (w \otimes w)$ and
$\text{ku}(w) = w^\top M_4 (w \otimes w \otimes w)$. Polynomial goal programming
maximises mean and skewness while minimising variance and kurtosis by minimising
the weighted relative deviations from each moment's aspiration level $g$,

$$
\min_{w \in \Delta}\
\sum_{k} \lambda_k
\left| \frac{g_k - f_k(w)}{g_k} \right|^{p},
$$

where $f_k$ ranges over the four moments and $\lambda_k$ encodes the investor
preference. The objective is nonlinear, so a general nonlinear solver is used.
The optimiser never forms the tensors. With $p = (R - \bar{r}) w$ the centred
portfolio return, $w^\top M_3 (w \otimes w)$ is the mean of $p^3$ and
$w^\top M_4 (w \otimes w \otimes w)$ the mean of $p^4$, so every objective
evaluation costs $O(TN)$ rather than $O(N^4)$.

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
$\Sigma$ for the posterior return covariance. By the Woodbury identity the same
quantities are

$$
\mu_{\text{BL}} = \pi + \tau \Sigma P^\top A^{-1} (Q - P \pi),
\qquad
M = \tau \Sigma - \tau \Sigma P^\top A^{-1} P \tau \Sigma,
\qquad
A = P \tau \Sigma P^\top + \Omega,
$$

which the implementation evaluates with one Cholesky solve of the $k \times k$
system $A$ instead of three $N \times N$ inversions, and which stays defined for
a view held with certainty. Feeding $\mu_{\text{BL}}$ into the mean-variance
step replaces noisy sample means with a shrunk, view-adjusted estimate.

## Resampled efficiency

Sample estimates are noisy, so a single optimisation overfits one draw. Michaud
resampling averages over the sampling distribution. For $b = 1, \dots, B$, draw
a bootstrap or parametric resample of the returns, estimate
$(\hat{\mu}^{(b)}, \Sigma^{(b)})$, and solve the chosen optimisation to get
$w^{(b)}$. The resampled portfolio is the average

$$
\bar{w} = \frac{1}{B} \sum_{b=1}^{B} w^{(b)},
$$

which remains on the simplex as a convex combination of feasible points. The
averaging shrinks the weights toward the centre of the efficient region and
reduces turnover relative to the single-shot solution. For a Gaussian resample
of length $T$ the estimates are sufficient statistics with known laws, the mean
$\hat{\mu}^{(b)} \sim \mathcal{N}(\hat{\mu}, \Sigma / T)$ independent of
$(T - 1)\, \Sigma^{(b)} \sim W_N(T - 1, \Sigma)$. The implementation draws them
directly, the Wishart through SciPy's Bartlett decomposition, and solves all $B$
systems in one batched call.

## Robust mean-variance

Robust optimisation guards against estimation error in $\hat{\mu}$ by optimising
the worst case over an uncertainty set. With a box set
$\mathcal{U} = \{ \mu : |\mu_i - \hat{\mu}_i| \le \delta_i \}$, the inner worst
case has a closed form,

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
