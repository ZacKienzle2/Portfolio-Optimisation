"""Stochastic-process simulation engines and common SDE models.

Provides generic discretisation schemes (Euler-Maruyama and Milstein) and
concrete one-dimensional processes built on them, plus the two-factor Heston
stochastic-volatility model and the Merton jump-diffusion. All simulators take
an explicit seed and return an array of shape ``(n_paths, n_steps + 1)`` that
includes the initial value.

Paths are built time-major, one contiguous row per step, so each step of a
recursion reads and writes adjacent memory, and the result is returned as the
transposed view. Antithetic variates pair each Brownian increment with its
negation for variance reduction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

Array = NDArray[np.float64]
Coefficient = Callable[[Array, float], Array]


def _brownian_increments(
    rng: np.random.Generator, n_paths: int, n_steps: int, *, antithetic: bool
) -> Array:
    """Standard normal increments of shape ``(n_steps, n_paths)``."""
    if not antithetic:
        return rng.standard_normal((n_steps, n_paths))
    base = rng.standard_normal((n_steps, (n_paths + 1) // 2))
    return np.concatenate([base, -base], axis=1)[:, :n_paths]


def _start(x0: float, n_steps: int, n_paths: int) -> Array:
    """Time-major path buffer with the initial value in the first row."""
    paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = x0
    return paths


def simulate_euler_maruyama(
    drift: Coefficient,
    diffusion: Coefficient,
    *,
    x0: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> Array:
    """Simulate ``dX = drift(X, t) dt + diffusion(X, t) dW`` via Euler-Maruyama.

    Args:
        drift: Drift ``a(x, t)`` acting on the path array.
        diffusion: Diffusion ``b(x, t)`` acting on the path array.
        x0: Initial value.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Use antithetic Brownian increments.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    shocks = np.sqrt(dt) * _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    paths = _start(x0, n_steps, n_paths)
    for i in range(n_steps):
        x = paths[i]
        time = i * dt
        paths[i + 1] = x + drift(x, time) * dt + diffusion(x, time) * shocks[i]
    return paths.T


def simulate_milstein(
    drift: Coefficient,
    diffusion: Coefficient,
    diffusion_prime: Coefficient,
    *,
    x0: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> Array:
    """Simulate a scalar SDE with the Milstein scheme, of strong order one.

    Adds the ``0.5 b b' (dW^2 - dt)`` correction to Euler-Maruyama, where ``b'``
    is the derivative of the diffusion in the state.

    Args:
        drift: Drift ``a(x, t)``.
        diffusion: Diffusion ``b(x, t)``.
        diffusion_prime: State derivative ``db/dx (x, t)``.
        x0: Initial value.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Use antithetic Brownian increments.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    shocks = np.sqrt(dt) * _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    paths = _start(x0, n_steps, n_paths)
    for i in range(n_steps):
        x = paths[i]
        time = i * dt
        dw = shocks[i]
        b = diffusion(x, time)
        paths[i + 1] = (
            x + drift(x, time) * dt + b * dw + 0.5 * b * diffusion_prime(x, time) * (dw**2 - dt)
        )
    return paths.T


def simulate_gbm(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> Array:
    """Geometric Brownian Motion via the exact log-Euler scheme, which stays positive.

    Args:
        s0: Initial price.
        mu: Drift.
        sigma: Volatility.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Use antithetic Brownian increments.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * _brownian_increments(
        rng, n_paths, n_steps, antithetic=antithetic
    )
    log_paths = np.zeros((n_steps + 1, n_paths))
    np.cumsum(increments, axis=0, out=log_paths[1:])
    return (s0 * np.exp(log_paths)).T


def simulate_ornstein_uhlenbeck(
    *,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> Array:
    """Ornstein-Uhlenbeck / Vasicek mean-reverting process, sampled exactly.

    The transition over a step ``dt`` is Gaussian with mean
    ``theta + (x - theta) exp(-kappa dt)`` and variance
    ``sigma^2 (1 - exp(-2 kappa dt)) / (2 kappa)``, so the paths are an AR(1)
    recursion with no discretisation bias. The recursion is a first-order linear
    filter, which ``scipy.signal.lfilter`` runs along every path at once.

    Unlike the stepping schemes, the filter walks along each path, so the paths
    are stored path-major and each one is contiguous. The normals are drawn into
    the output buffer and scaled in place, and ``x0`` sits in the first column,
    where the filter passes it through unchanged and so needs no initial state.
    Filtering along the time axis of a time-major buffer, with two scaled
    copies of the normals, ran slower than the Euler loop it replaced.

    Args:
        x0: Initial value.
        kappa: Mean-reversion speed, non-negative.
        theta: Long-run mean.
        sigma: Diffusion coefficient.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Use antithetic Gaussian innovations.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    decay = np.exp(-kappa * dt)
    variance = -np.expm1(-2.0 * kappa * dt) / (2.0 * kappa) if kappa > 0 else dt
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    if antithetic:
        half = (n_paths + 1) // 2
        rng.standard_normal(out=paths[:half])
        np.negative(paths[: n_paths - half], out=paths[half:])
    else:
        rng.standard_normal(out=paths)
    shocks = paths[:, 1:]
    shocks *= sigma * np.sqrt(variance)
    shocks += theta * (1.0 - decay)
    paths[:, 0] = x0
    return lfilter([1.0], [1.0, -decay], paths, axis=1)


def simulate_cir(
    *,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> Array:
    """Cox-Ingersoll-Ross square-root process with full-truncation Euler.

    The state is floored at zero each step so the square-root diffusion stays
    well defined even when the Feller condition is violated.

    Args:
        x0: Initial value, non-negative.
        kappa: Mean-reversion speed.
        theta: Long-run mean.
        sigma: Volatility of the square-root diffusion.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Use antithetic Brownian increments.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    shocks = (
        sigma * np.sqrt(dt) * _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    )
    paths = _start(x0, n_steps, n_paths)
    for i in range(n_steps):
        x = np.maximum(paths[i], 0.0)
        np.maximum(
            paths[i] + kappa * (theta - x) * dt + np.sqrt(x) * shocks[i], 0.0, out=paths[i + 1]
        )
    return paths.T


def simulate_merton_jump_diffusion(
    *,
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> Array:
    """Merton jump-diffusion, GBM plus a compound-Poisson log-normal jump term.

    Over a step the number of jumps ``N`` is Poisson with mean
    ``jump_intensity * dt`` and, given ``N``, the sum of the log jump sizes is
    normal with mean ``N * jump_mean`` and variance ``N * jump_std^2``. The drift
    carries the compensator, so ``E[S_t] = s0 exp(mu t)``. Jump sizes are drawn
    only for the steps that have a jump, and the log price is the cumulative sum
    of every increment.

    Args:
        s0: Initial price.
        mu: Expected rate of return.
        sigma: Diffusion volatility.
        jump_intensity: Expected number of jumps per unit time.
        jump_mean: Mean of the log jump size.
        jump_std: Standard deviation of the log jump size.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.

    Returns:
        Price paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    compensator = jump_intensity * np.expm1(jump_mean + 0.5 * jump_std**2)
    shape = (n_steps, n_paths)
    increments = (mu - 0.5 * sigma**2 - compensator) * dt + sigma * np.sqrt(
        dt
    ) * rng.standard_normal(shape)
    counts = rng.poisson(jump_intensity * dt, size=shape)
    jumped = np.flatnonzero(counts)
    hits = counts.flat[jumped]
    increments.flat[jumped] += hits * jump_mean + np.sqrt(hits) * jump_std * rng.standard_normal(
        jumped.size
    )
    log_paths = np.zeros((n_steps + 1, n_paths))
    np.cumsum(increments, axis=0, out=log_paths[1:])
    return (s0 * np.exp(log_paths)).T


def simulate_heston(
    *,
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Heston stochastic-volatility model with full-truncation Euler on variance.

    Each step draws its two normals and advances the variance and the log price
    together, so the working set is one row of paths, which stays in cache.
    Drawing the whole normal panel up front and summing the log price afterwards
    moved several full-size temporaries through memory and ran slower.

    Args:
        s0: Initial price.
        v0: Initial variance.
        mu: Price drift.
        kappa: Variance mean-reversion speed.
        theta: Long-run variance.
        xi: Volatility of variance.
        rho: Correlation between the price and variance Brownians.
        t: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.

    Returns:
        Price paths and variance paths, each of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    orthogonal = np.sqrt(1.0 - rho**2)
    variances = _start(v0, n_steps, n_paths)
    log_paths = _start(0.0, n_steps, n_paths)
    for i in range(n_steps):
        price_shock = rng.standard_normal(n_paths)
        variance_shock = rho * price_shock + orthogonal * rng.standard_normal(n_paths)
        v = np.maximum(variances[i], 0.0)
        root = np.sqrt(v) * sqrt_dt
        np.maximum(
            variances[i] + kappa * (theta - v) * dt + xi * root * variance_shock,
            0.0,
            out=variances[i + 1],
        )
        np.add(log_paths[i], (mu - 0.5 * v) * dt + root * price_shock, out=log_paths[i + 1])
    return (s0 * np.exp(log_paths)).T, variances.T
