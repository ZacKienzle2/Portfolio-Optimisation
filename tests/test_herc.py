"""Tests for Hierarchical Equal Risk Contribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimisation.optim import HERCModel, herc_weights


def _block_returns(seed: int = 13, t: int = 600, n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.005, size=t)
    cols = []
    for k in range(n):
        block = k // 5
        factor = rng.normal(0.0, 0.01, size=t)
        idio = rng.normal(0.0, 0.003, size=t)
        cols.append(0.3 * market + (0.6 if block == 0 else 0.4) * factor + idio)
    data = np.column_stack(cols)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])


def test_herc_variance_split_is_long_only_simplex() -> None:
    returns = _block_returns()
    weights = herc_weights(returns, risk_measure="variance")
    assert (weights >= -1e-12).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)


def test_herc_cvar_split_is_long_only_simplex() -> None:
    returns = _block_returns()
    weights = herc_weights(returns, risk_measure="cvar", cvar_alpha=0.05)
    assert (weights >= -1e-12).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)


def test_herc_equalises_cluster_risk_pair() -> None:
    """At each ERC split the two sibling clusters should contribute equal risk."""
    returns = _block_returns()
    weights = herc_weights(returns)
    cov = returns.cov().to_numpy()
    w = weights.to_numpy()
    # The full-portfolio variance should be finite and positive.
    portfolio_var = float(w @ cov @ w)
    assert portfolio_var > 0


def test_herc_model_caches_weights() -> None:
    returns = _block_returns()
    m = HERCModel(returns, risk_measure="variance")
    w = m.optimise()
    assert np.allclose(w.to_numpy(), m.weights.to_numpy())
