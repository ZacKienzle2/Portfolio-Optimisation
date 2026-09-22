# This test code was written by the `hypothesis.extra.ghostwriter` module

# and is provided under the Creative Commons Zero public domain dedication.


import typing

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import portfolio_optimisation.baselines
import portfolio_optimisation.risk


@settings(deadline=None)
@given(
    alpha=st.floats(0.005, 0.5),
    kind=st.sampled_from(["loss", "return"]),
    values=st.tuples(st.integers(2, 500), st.floats(1e-4, 1e-1), st.integers(0, 2**32 - 1)).map(
        lambda s: np.random.default_rng(s[2]).standard_t(4, size=s[0]) * s[1]
    ),
)
def test_equivalent_evar_dual_representation_entropic_value_at_risk(
    alpha: float,
    kind: str,
    values: np.ndarray[tuple[typing.Any, Ellipsis], np.dtype[np.float64]],
) -> None:

    result_evar_dual_representation = portfolio_optimisation.baselines.evar_dual_representation(
        values=values, alpha=alpha, kind=kind
    )

    result_entropic_value_at_risk = portfolio_optimisation.risk.entropic_value_at_risk(
        values=values, alpha=alpha, kind=kind
    )

    np.testing.assert_allclose(
        result_evar_dual_representation, result_entropic_value_at_risk, rtol=1e-5, atol=1e-7
    )
