"""Stochastic-process simulation engines and common SDE models.

Provides generic discretisation schemes (Euler-Maruyama and Milstein) and
concrete one-dimensional processes built on them, plus the two-factor Heston
stochastic-volatility model and the Merton jump-diffusion. All simulators take
an explicit seed and return an array of shape ``(n_paths, n_steps + 1)`` that
includes the initial value.

Antithetic variates are available on the generic engines for variance
reduction: each Brownian increment is paired with its negation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Coefficient = Callable[[Array, float], Array]


def _brownian_increments(
    rng: np.random.Generator, n_paths: int, n_steps: int, *, antithetic: bool
) -> Array:
    if not antithetic:
        return rng.standard_normal((n_paths, n_steps))
    half = (n_paths + 1) // 2
    base = rng.standard_normal((half, n_steps))
    return np.concatenate([base, -base], axis=0)[:n_paths]


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
        drift (Coefficient): Drift ``a(x, t)`` acting on the path array.
        diffusion (Coefficient): Diffusion ``b(x, t)`` acting on the path array.
        x0 (float): Initial value.
        t (float): Time horizon.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulated paths.
        seed (int | None): Seed for reproducibility.
        antithetic (bool): Use antithetic Brownian increments.

    Returns:
        Array: Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    increments = _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0
    for i in range(n_steps):
        x = paths[:, i]
        time = i * dt
        paths[:, i + 1] = x + drift(x, time) * dt + diffusion(x, time) * sqrt_dt * increments[:, i]
    return paths


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
    """Simulate a scalar SDE with the Milstein scheme (higher strong order).

    Adds the ``0.5 b b' (dW^2 - dt)`` correction to Euler-Maruyama, where ``b'``
    is the derivative of the diffusion in the state. The ``x0``, ``t``,
    ``n_steps``, ``n_paths``, ``seed`` and ``antithetic`` arguments are as in
    :func:`simulate_euler_maruyama`.

    Args:
        drift (Coefficient): Drift ``a(x, t)``.
        diffusion (Coefficient): Diffusion ``b(x, t)``.
        diffusion_prime (Coefficient): State derivative ``db/dx (x, t)``.

    Returns:
        Array: Paths of shape ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    increments = _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0
    for i in range(n_steps):
        x = paths[:, i]
        time = i * dt
        dw = sqrt_dt * increments[:, i]
        b = diffusion(x, time)
        paths[:, i + 1] = (
            x + drift(x, time) * dt + b * dw + 0.5 * b * diffusion_prime(x, time) * (dw**2 - dt)
        )
    return paths


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
    """Geometric Brownian Motion via the exact log-Euler scheme (stays positive)."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    increments = _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    drift = (mu - 0.5 * sigma**2) * dt
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(drift + sigma * np.sqrt(dt) * increments, axis=1)],
        axis=1,
    )
    return s0 * np.exp(log_paths)


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
    """Ornstein-Uhlenbeck / Vasicek mean-reverting process via Euler-Maruyama."""
    return simulate_euler_maruyama(
        lambda x, _: kappa * (theta - x),
        lambda x, _: np.full_like(x, sigma),
        x0=x0,
        t=t,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )


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
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    increments = _brownian_increments(rng, n_paths, n_steps, antithetic=antithetic)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0
    for i in range(n_steps):
        x = np.maximum(paths[:, i], 0.0)
        paths[:, i + 1] = np.maximum(
            paths[:, i]
            + kappa * (theta - x) * dt
            + sigma * np.sqrt(x) * sqrt_dt * increments[:, i],
            0.0,
        )
    return paths


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
    """Merton jump-diffusion: GBM plus a compound-Poisson log-normal jump term."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    compensator = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    drift = (mu - 0.5 * sigma**2 - compensator) * dt

    log_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(s0)
    for i in range(n_steps):
        diffusion = drift + sigma * sqrt_dt * rng.standard_normal(n_paths)
        counts = rng.poisson(jump_intensity * dt, size=n_paths)
        jumps = rng.normal(jump_mean, jump_std, size=n_paths) * counts
        log_paths[:, i + 1] = log_paths[:, i] + diffusion + jumps
    return np.exp(log_paths)


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

    Args:
        s0 (float): Initial price.
        v0 (float): Initial variance.
        mu (float): Price drift.
        kappa (float): Variance mean-reversion speed.
        theta (float): Long-run variance.
        xi (float): Volatility of variance.
        rho (float): Correlation between the price and variance Brownians.
        t (float): Time horizon.
        n_steps (int): Number of time steps.
        n_paths (int): Number of simulated paths.
        seed (int | None): Seed for reproducibility.

    Returns:
        tuple[Array, Array]: Price paths and variance paths, each of shape
        ``(n_paths, n_steps + 1)``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = np.sqrt(dt)
    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    variances[:, 0] = v0
    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n_paths)
        v = np.maximum(variances[:, i], 0.0)
        variances[:, i + 1] = np.maximum(
            variances[:, i] + kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * z2,
            0.0,
        )
        prices[:, i + 1] = prices[:, i] * np.exp((mu - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * z1)
    return prices, variances
