# Estimation

Derivations for the covariance and dependence estimators in
`portfolio_optimisation.optim` and `portfolio_optimisation.risk`. The shared
notation is defined on the [methodology index](index.md). The concentration
ratio is $q = N / T$.

## Why the sample covariance fails

When $q$ is not small, the sample covariance $\Sigma$ is a poor estimator.
Its eigenvalues are spread wider than the population's, the largest biased up and
the smallest biased down, so inverting it amplifies noise. The estimators below
correct the spectrum in three different ways.

## Linear shrinkage

Ledoit-Wolf shrinkage pulls the sample covariance toward a structured target
$F$ by a constant intensity $\delta \in [0, 1]$,

$$
\hat{\Sigma} = \delta\, F + (1 - \delta)\, \Sigma .
$$

The optimal intensity minimises the expected Frobenius loss
$\mathbb{E}\|\hat{\Sigma} - \Sigma_{\text{pop}}\|_F^2$, which has the closed form
$\delta^\star = \pi / \gamma$ clipped to $[0, 1]$, where $\pi$ estimates the
summed variance of the sample covariance entries and $\gamma$ the squared
distance $\|\Sigma - F\|_F^2$ to the target. With a scaled-identity target the
intensity grows with the noise and shrinks toward zero as $T$ grows. The
Oracle-Approximating Shrinkage uses the same form with an intensity tuned for
the Gaussian case, which converges faster when the returns are close to normal.

## Nonlinear shrinkage

Linear shrinkage applies one intensity to every eigenvalue. Nonlinear shrinkage
applies a separate optimal correction to each. Eigendecompose
$\Sigma = U \operatorname{diag}(\lambda) U^\top$. The oracle that minimises the
Frobenius loss while keeping the sample eigenvectors replaces each $\lambda_i$
by $d_i = u_i^\top \Sigma_{\text{pop}} u_i$, which the analytical estimator of
Ledoit and Wolf (2020) approximates from the sample spectral density. With the
Epanechnikov kernel of bandwidth $h = T^{-1/3}$, the density and its Hilbert
transform are estimated at each eigenvalue,

$$
\tilde{f}(\lambda_i),\quad \mathcal{H}\tilde{f}(\lambda_i),
\qquad
d_i = \frac{\lambda_i}
{\big( \pi q \lambda_i \tilde{f}(\lambda_i) \big)^2
+ \big( 1 - q - \pi q \lambda_i\, \mathcal{H}\tilde{f}(\lambda_i) \big)^2 } .
$$

Rebuilding $\hat{\Sigma} = U \operatorname{diag}(d) U^\top$ gives a covariance
that recovers the sample spectrum as $T \to \infty$ and improves the
conditioning sharply when $q$ is moderate, the two properties the
implementation verifies numerically.

## Marchenko-Pastur denoising

Random-matrix theory gives the limiting spectrum of a pure-noise correlation
matrix. For independent returns with $q = N/T < 1$, the eigenvalues fall in
$[\lambda_-, \lambda_+]$ with edges

$$
\lambda_\pm = \sigma^2 \big( 1 \pm \sqrt{q} \big)^2,
$$

the Marchenko-Pastur support. Eigenvalues inside the band are indistinguishable
from noise, so denoising fits $\sigma^2$ to the empirical bulk, replaces every
eigenvalue below $\lambda_+$ by their common average while preserving the
trace, and rebuilds the correlation. Detoning additionally removes the largest
eigenvalue, the market mode, so the clustering used by the hierarchical
allocators sees the residual structure rather than the dominant common factor.

## Factor-model covariance

A factor model imposes structure by writing returns as a small number of common
drivers plus idiosyncratic noise, $r = B f + \varepsilon$, with factor
covariance $\Sigma_f$ and diagonal idiosyncratic covariance $D$. The implied
covariance is

$$
\hat{\Sigma} = B \Sigma_f B^\top + D ,
$$

low rank plus diagonal, which is well conditioned even when $N > T$. The
statistical variant takes the loadings $B$ from the leading principal
components of the returns, so the factors are the directions of greatest
variance. The explicit variant regresses the returns on observed factors and
reads $B$ from the slopes, with $D$ the variance of the residuals.

## Copula dependence

A copula separates the dependence structure from the marginal behaviour.
Sklar's theorem states that any joint distribution factors as
$H(x) = C\big( F_1(x_1), \dots, F_N(x_N) \big)$ for a unique copula $C$ on
uniform margins. Fitting the margins and the copula separately lets each asset
keep its own fat-tailed marginal while a Student-t copula models the joint tail
dependence that a Gaussian dependence would miss. The copula correlation is
calibrated from Kendall's tau through $\rho = \sin(\pi \tau / 2)$, which is
robust to the marginal shapes, and the fitted model is sampled to generate the
joint scenarios consumed by the risk metrics.
