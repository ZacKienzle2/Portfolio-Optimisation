# This test code was written by the `hypothesis.extra.ghostwriter` module

# and is provided under the Creative Commons Zero public domain dedication.


import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

import portfolio_optimisation.baselines
import portfolio_optimisation.optim


@settings(deadline=None, max_examples=25)
@given(
    cov_matrix=st.none(),
    returns=st.tuples(st.integers(20, 250), st.integers(2, 12), st.integers(0, 2**32 - 1)).map(
        lambda s: pd.DataFrame(np.random.default_rng(s[2]).normal(0.0, 0.01, size=s[:2]))
    ),
    solver=st.just("CLARABEL"),
)
def test_equivalent_max_diversification_ratio_weights_max_diversification_weights(
    cov_matrix: pd.DataFrame | None,
    returns: pd.DataFrame,
    solver: str,
) -> None:

    result_max_diversification_ratio_weights = (
        portfolio_optimisation.baselines.max_diversification_ratio_weights(
            returns=returns, cov_matrix=cov_matrix
        )
    )

    result_max_diversification_weights = portfolio_optimisation.optim.max_diversification_weights(
        returns=returns, cov_matrix=cov_matrix, solver=solver
    )

    np.testing.assert_allclose(
        result_max_diversification_ratio_weights, result_max_diversification_weights, atol=1e-5
    )
