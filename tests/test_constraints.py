"""Tests for the reusable portfolio constraint framework."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from portfolio_optimisation.optim import PortfolioConstraints


def test_build_emits_budget_and_long_only() -> None:
    w = cp.Variable(4)
    built = PortfolioConstraints().build(cp, w, n_assets=4)
    assert len(built) == 2


def test_min_weight_overrides_long_only() -> None:
    w = cp.Variable(3)
    spec = PortfolioConstraints(long_only=True, min_weight=-0.1)
    built = spec.build(cp, w, n_assets=3)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - np.array([0.9, 0.9, -0.8]))), built)
    problem.solve()
    solution = np.asarray(w.value, dtype=np.float64)
    assert (solution >= -0.1 - 1e-6).all()


def test_max_leverage_must_be_positive() -> None:
    w = cp.Variable(3)
    with pytest.raises(ValueError, match="max_leverage"):
        PortfolioConstraints(long_only=False, max_leverage=0.0).build(cp, w, n_assets=3)


def test_group_matrix_shape_validated() -> None:
    w = cp.Variable(4)
    spec = PortfolioConstraints(group_matrix=np.ones((2, 3)))
    with pytest.raises(ValueError, match="group_matrix"):
        spec.build(cp, w, n_assets=4)


def test_turnover_requires_previous_weights() -> None:
    w = cp.Variable(3)
    with pytest.raises(ValueError, match="previous_weights"):
        PortfolioConstraints(max_turnover=0.3).build(cp, w, n_assets=3)


def test_target_return_requires_expected_returns() -> None:
    w = cp.Variable(3)
    with pytest.raises(ValueError, match="expected_returns"):
        PortfolioConstraints(target_return=0.01).build(cp, w, n_assets=3)


def test_group_exposure_bounds_enforced() -> None:
    w = cp.Variable(4)
    group = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    spec = PortfolioConstraints(
        group_matrix=group,
        group_max=np.array([0.6, 1.0]),
        group_min=np.array([0.0, 0.4]),
    )
    built = spec.build(cp, w, n_assets=4)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - np.array([1.0, 0.0, 0.0, 0.0]))), built)
    problem.solve()
    solution = np.asarray(w.value, dtype=np.float64)
    assert float(group[0] @ solution) <= 0.6 + 1e-6
    assert float(group[1] @ solution) >= 0.4 - 1e-6
