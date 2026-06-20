"""Lock the einsum co-moment tensors to their brute-force definitions.

Guards the vectorised co-skewness and co-kurtosis builders against any future
reshape or index-order regression by comparing against explicit loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimisation.optim.higher_moments import (
    cokurtosis_tensor,
    coskewness_tensor,
    pgp_higher_moment_weights,
)


def _sample(seed: int = 0, t: int = 60, n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = [f"A{i}" for i in range(n)]
    return pd.DataFrame(rng.normal(0.0, 0.02, size=(t, n)), columns=columns)


def test_coskewness_matches_definition() -> None:
    df = _sample()
    centred = (df - df.mean()).to_numpy()
    _, n = centred.shape
    ref = np.zeros((n, n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ref[i, j * n + k] = np.mean(centred[:, i] * centred[:, j] * centred[:, k])
    np.testing.assert_allclose(coskewness_tensor(df), ref, atol=1e-12)


def test_cokurtosis_matches_definition() -> None:
    df = _sample()
    centred = (df - df.mean()).to_numpy()
    _, n = centred.shape
    ref = np.zeros((n, n * n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    ref[i, j * n * n + k * n + ell] = np.mean(
                        centred[:, i] * centred[:, j] * centred[:, k] * centred[:, ell]
                    )
    np.testing.assert_allclose(cokurtosis_tensor(df), ref, atol=1e-12)


def test_pgp_rejects_oversized_universe() -> None:
    df = _sample(n=6)
    with pytest.raises(ValueError, match="O\\(N\\^4\\)"):
        pgp_higher_moment_weights(df, max_assets=5)
