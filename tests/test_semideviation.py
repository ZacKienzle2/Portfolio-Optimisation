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
    constraints=st.none(),
    measure=st.sampled_from(["standard", "absolute"]),
    returns=st.tuples(st.integers(20, 250), st.integers(2, 8), st.integers(0, 2**32 - 1)).map(
        lambda s: pd.DataFrame(np.random.default_rng(s[2]).normal(0.0003, 0.01, size=s[:2]))
    ),
    risk_aversion=st.floats(0.05, 1.0),
    solver=st.just("CLARABEL"),
)
def test_equivalent_mean_semideviation_lp_weights_mean_semideviation_weights(
    constraints: portfolio_optimisation.optim.PortfolioConstraints | None,
    measure: str,
    returns: pd.DataFrame,
    risk_aversion: float,
    solver: str,
) -> None:

    result_mean_semideviation_lp_weights = (
        portfolio_optimisation.baselines.mean_semideviation_lp_weights(
            returns=returns, risk_aversion=risk_aversion, measure=measure
        )
    )

    result_mean_semideviation_weights = portfolio_optimisation.optim.mean_semideviation_weights(
        returns=returns,
        risk_aversion=risk_aversion,
        measure=measure,
        constraints=constraints,
        solver=solver,
    )

    np.testing.assert_allclose(
        *(
            portfolio_optimisation.baselines.mean_semideviation_objective(
                returns, result, risk_aversion=risk_aversion, measure=measure
            )
            for result in (result_mean_semideviation_lp_weights, result_mean_semideviation_weights)
        ),
        rtol=1e-6,
        atol=1e-9,
    )
