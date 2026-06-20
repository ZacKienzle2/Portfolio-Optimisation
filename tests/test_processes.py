"""Tests for the SDE simulation engines and process models."""

from __future__ import annotations

import numpy as np

from portfolio_optimisation.sde import (
    simulate_cir,
    simulate_gbm,
    simulate_heston,
    simulate_merton_jump_diffusion,
    simulate_ornstein_uhlenbeck,
)


def test_gbm_is_positive_and_correctly_shaped() -> None:
    paths = simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, t=1.0, n_steps=252, n_paths=500, seed=1)
    assert paths.shape == (500, 253)
    assert bool((paths > 0).all())
    assert np.allclose(paths[:, 0], 100.0)


def test_gbm_mean_matches_analytic_expectation() -> None:
    paths = simulate_gbm(
        s0=100.0, mu=0.10, sigma=0.2, t=1.0, n_steps=64, n_paths=40000, seed=7, antithetic=True
    )
    terminal_mean = float(paths[:, -1].mean())
    assert abs(terminal_mean - 100.0 * np.exp(0.10)) / (100.0 * np.exp(0.10)) < 0.03


def test_gbm_is_seed_reproducible() -> None:
    kwargs = {"s0": 100.0, "mu": 0.05, "sigma": 0.2, "t": 1.0, "n_steps": 50, "n_paths": 100}
    np.testing.assert_array_equal(simulate_gbm(seed=11, **kwargs), simulate_gbm(seed=11, **kwargs))


def test_ornstein_uhlenbeck_reverts_to_mean() -> None:
    paths = simulate_ornstein_uhlenbeck(
        x0=0.0, kappa=5.0, theta=0.05, sigma=0.02, t=3.0, n_steps=300, n_paths=20000, seed=3
    )
    assert abs(float(paths[:, -1].mean()) - 0.05) < 0.01


def test_cir_stays_non_negative() -> None:
    paths = simulate_cir(
        x0=0.04, kappa=3.0, theta=0.04, sigma=0.5, t=2.0, n_steps=400, n_paths=2000, seed=5
    )
    assert bool((paths >= 0.0).all())


def test_merton_prices_are_positive() -> None:
    paths = simulate_merton_jump_diffusion(
        s0=100.0,
        mu=0.05,
        sigma=0.2,
        jump_intensity=1.0,
        jump_mean=-0.1,
        jump_std=0.15,
        t=1.0,
        n_steps=252,
        n_paths=1000,
        seed=9,
    )
    assert paths.shape == (1000, 253)
    assert bool((paths > 0).all())


def test_heston_variance_non_negative_and_prices_positive() -> None:
    prices, variances = simulate_heston(
        s0=100.0,
        v0=0.04,
        mu=0.03,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        t=1.0,
        n_steps=252,
        n_paths=1000,
        seed=13,
    )
    assert prices.shape == variances.shape == (1000, 253)
    assert bool((variances >= 0.0).all())
    assert bool((prices > 0.0).all())
