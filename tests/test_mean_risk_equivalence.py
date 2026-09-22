# This test code was written by the `hypothesis.extra.ghostwriter` module

# and is provided under the Creative Commons Zero public domain dedication.


import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import portfolio_optimisation.baselines
import portfolio_optimisation.optim


@settings(deadline=None, max_examples=25)
@given(
    alpha=st.floats(0.01, 0.5),
    constraints=st.none(),
    returns=st.tuples(st.integers(20, 500), st.integers(2, 8), st.integers(0, 2**32 - 1)).map(
        lambda s: pd.DataFrame(np.random.default_rng(s[2]).standard_t(4, size=s[:2]) / 100)
    ),
)
def test_equivalent_min_evar_cone_weights_min_evar_weights(
    alpha: float,
    constraints: portfolio_optimisation.optim.PortfolioConstraints | None,
    returns: pd.DataFrame,
) -> None:
    assume(alpha * len(returns) > 2.0)

    result_min_evar_cone_weights = portfolio_optimisation.baselines.min_evar_cone_weights(
        returns=returns, alpha=alpha
    )

    result_min_evar_weights = portfolio_optimisation.optim.min_evar_weights(
        returns=returns, alpha=alpha, constraints=constraints
    )

    np.testing.assert_allclose(result_min_evar_cone_weights, result_min_evar_weights, atol=1e-4)
