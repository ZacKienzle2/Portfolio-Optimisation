# Stochastic processes

Derivations for the simulation engines and estimators in
`portfolio_optimisation.sde`. The shared notation is defined on the
[methodology index](index.md). A process $X_t$ follows the stochastic
differential equation $dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t$ for a
Brownian motion $W_t$, with step $\Delta t$.

## Discretisation schemes

The Euler-Maruyama scheme is the first-order discretisation,

$$
X_{t+\Delta t} = X_t + \mu(X_t)\, \Delta t
+ \sigma(X_t)\, \sqrt{\Delta t}\, Z, \qquad Z \sim \mathcal{N}(0, 1),
$$

with strong order one half. The Milstein scheme adds the first Ito correction
from the expansion of $\sigma$,

$$
X_{t+\Delta t} = X_t + \mu(X_t)\, \Delta t
+ \sigma(X_t)\, \sqrt{\Delta t}\, Z
+ \tfrac{1}{2} \sigma(X_t)\, \sigma'(X_t)\, \Delta t\, (Z^2 - 1),
$$

raising the strong order to one when $\sigma$ depends on the state. The extra
term vanishes for constant diffusion, where the two schemes coincide.

## Geometric Brownian motion

Geometric Brownian motion $dS_t = \mu S_t\, dt + \sigma S_t\, dW_t$ has a
closed-form solution from Ito's lemma applied to $\ln S_t$,

$$
S_{t+\Delta t} = S_t \exp\!\left(
\big( \mu - \tfrac{1}{2}\sigma^2 \big)\Delta t
+ \sigma \sqrt{\Delta t}\, Z \right).
$$

Simulating the exact log-increment avoids the discretisation error entirely, so
the implementation uses this form rather than the Euler step.

## Mean-reverting processes

The Ornstein-Uhlenbeck process $dX_t = \kappa(\theta - X_t)\, dt +
\sigma\, dW_t$ is Gaussian with an exact transition, mean
$\theta + (X_t - \theta) e^{-\kappa \Delta t}$ and variance
$\tfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa \Delta t})$, which the simulator
samples directly. The Cox-Ingersoll-Ross process
$dX_t = \kappa(\theta - X_t)\, dt + \sigma \sqrt{X_t}\, dW_t$ stays
non-negative when the Feller condition $2 \kappa \theta \ge \sigma^2$ holds. A
plain Euler step can go negative through the square root, so the implementation
applies full truncation, evaluating the diffusion at $\max(X_t, 0)$, which
keeps the path well defined.

## Jumps and stochastic volatility

The Merton model adds a compound-Poisson jump to geometric Brownian motion,
$dS_t / S_t = \mu\, dt + \sigma\, dW_t + dJ_t$, where jumps arrive at rate
$\lambda$ with lognormal sizes. Over a step the number of jumps is Poisson with
mean $\lambda \Delta t$ and the log return gains the sum of the jump sizes. The
Heston model gives the volatility its own mean-reverting square-root process,

$$
dS_t = \mu S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^S, \qquad
dv_t = \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^v,
$$

with correlated drivers $\mathrm{corr}(dW^S, dW^v) = \rho$. The variance path
is simulated with the same full-truncation guard as the Cox-Ingersoll-Ross
process, and the correlated normals are drawn from the Cholesky factor of the
two-by-two correlation.

## Variance reduction

Antithetic variates halve the simulation variance at no extra cost for the
symmetric drivers. For each normal draw $Z$ the mirror $-Z$ is also used, so
the estimator averages $f(Z)$ and $f(-Z)$. When $f$ is monotone the two are
negatively correlated, which lowers the variance of their mean below that of two
independent draws. Every path is seeded explicitly so the simulation is
reproducible.

## Parameter estimation

The drift and diffusion parameters are fitted by maximum likelihood on the exact
transition densities. For geometric Brownian motion the log-increments are
independent and normal, so the estimates are the sample mean and variance of the
log returns rescaled by $\Delta t$. For the Ornstein-Uhlenbeck process the
Gaussian transition gives a likelihood whose maximiser is available in closed
form, and the fitter maximises it per asset, in parallel across the universe.
