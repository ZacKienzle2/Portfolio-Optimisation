"""Implementations that a library call or a faster formulation replaced.

Each function is the version the package ran before its rewrite, or for an
allocator with no predecessor the formulation its paper states, kept so an
equivalence test can compare the package against it on generated inputs. None
of them is called by the package itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize, minimize_scalar

from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance


def quasi_diagonal(linkage_matrix: NDArray[np.float64]) -> NDArray[np.intp]:
    """Seriate dendrogram leaves by splicing each cluster into its children.

    Replaced by ``scipy.cluster.hierarchy.leaves_list``.

    Args:
        linkage_matrix: SciPy linkage matrix.

    Returns:
        Leaf indices in dendrogram order.
    """
    num_items = linkage_matrix.shape[0] + 1
    sorted_index: list[float] = [linkage_matrix[-1, 0], linkage_matrix[-1, 1]]
    while len(sorted_index) < num_items:
        position = next(i for i, item in enumerate(sorted_index) if item >= num_items)
        row = int(sorted_index[position] - num_items)
        sorted_index = [
            *sorted_index[:position],
            linkage_matrix[row, 0],
            linkage_matrix[row, 1],
            *sorted_index[position + 1 :],
        ]
    return np.asarray(sorted_index, dtype=np.intp)


def clipped_correlation(corr: NDArray[np.float64], threshold: float = 1e-8) -> NDArray[np.float64]:
    """Floor the eigenvalues at ``threshold`` and rescale to a unit diagonal.

    Replaced by ``statsmodels.stats.correlation_tools.corr_clipped``.

    Args:
        corr: Symmetric correlation estimate.
        threshold: Smallest eigenvalue kept.

    Returns:
        Positive definite correlation matrix.
    """
    epsilon = threshold
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues[eigenvalues < epsilon] = epsilon
    rebuilt = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    scale = np.where(scale < epsilon, 1.0, scale)
    return np.clip(rebuilt / np.outer(scale, scale), -1.0, 1.0)


def black_litterman_posterior(
    sigma: NDArray[np.float64],
    pi: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    omega: NDArray[np.float64],
    tau: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Black-Litterman posterior mean and covariance in the precision form.

    Inverts ``tau Sigma``, ``Omega`` and the posterior precision, three
    ``O(N^3)`` inversions, where the rewrite solves one ``k x k`` system.

    Args:
        sigma: Asset covariance.
        pi: Equilibrium returns.
        p: Pick matrix.
        q: View returns.
        omega: View uncertainty.
        tau: Prior covariance scale.

    Returns:
        Posterior mean and covariance.
    """
    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega)
    posterior = np.linalg.inv(tau_sigma_inv + p.T @ omega_inv @ p)
    return posterior @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q), sigma + posterior


def portfolio_third_and_fourth_moments(
    weights: NDArray[np.float64], centred: NDArray[np.float64]
) -> tuple[float, float]:
    """Portfolio third and fourth central moments through the co-moment tensors.

    Contracts ``w' M3 (w kron w)`` and ``w' M4 (w kron w kron w)``, which costs
    ``O(N^4)`` per call, where the rewrite takes the moments of ``centred @ w``.

    Args:
        weights: Portfolio weights.
        centred: Demeaned returns, one column per asset.

    Returns:
        Third and fourth central moments of the portfolio return.
    """
    t, n = centred.shape
    m3 = np.einsum("ti,tj,tk->ijk", centred, centred, centred).reshape(n, n * n) / t
    m4 = np.einsum("ti,tj,tk,tl->ijkl", centred, centred, centred, centred).reshape(n, n**3) / t
    pair = np.kron(weights, weights)
    return float(weights @ m3 @ pair), float(weights @ m4 @ np.kron(weights, pair))


def lower_partial_moments(
    sample: NDArray[np.float64], thresholds: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``E[max(eta - X, 0)]`` at every threshold through an ``O(T^2)`` matrix.

    Args:
        sample: Scenario outcomes.
        thresholds: Thresholds ``eta``.

    Returns:
        Lower partial moment at each threshold.
    """
    return np.maximum(thresholds[:, None] - sample[None, :], 0.0).mean(axis=1)


def ssd_expected_return(returns: NDArray[np.float64], benchmark: NDArray[np.float64]) -> float:
    """Optimal expected return under SSD dominance from the pairwise programme.

    One slack variable per pair of scenario and benchmark threshold, ``T^2`` in
    all, where the rewrite adds tail-sum cuts to an ``N``-variable programme.
    Solved by HiGHS, whose simplex vertex is exact, since an interior-point
    solution sits a tolerance outside a feasible set as small as one point.
    Needs the ``[optim]`` extra.

    Args:
        returns: Scenario returns, one column per asset.
        benchmark: Benchmark scenario returns.

    Returns:
        The largest long-only expected return that dominates the benchmark.
    """
    import cvxpy as cp

    t_steps, n_assets = returns.shape
    threshold_lpm = np.maximum(benchmark[:, None] - benchmark[None, :], 0.0).mean(axis=1)
    w = cp.Variable(n_assets, nonneg=True)
    slack = cp.Variable((t_steps, t_steps), nonneg=True)
    problem = cp.Problem(
        cp.Maximize(returns.mean(axis=0) @ w),
        [
            cp.sum(w) == 1,
            slack >= benchmark[:, None] - (returns @ w)[None, :],
            slack @ np.full(t_steps, 1.0 / t_steps) <= threshold_lpm,
        ],
    )
    problem.solve(solver=cp.HIGHS)
    return float(problem.value)


def min_cdar_objective(returns: NDArray[np.float64], alpha: float) -> float:
    """Minimum CDaR with a running-peak variable per date.

    Rows state ``m_t >= P_t`` on the cumulative return, ``m_t >= m_{t-1}`` and
    ``m_0 >= 0``, three rows per date with the cumulative returns dense in the
    weights, where the rewrite uses the drawdown recursion. Long-only and
    fully invested. Needs the ``[optim]`` extra.

    Args:
        returns: Asset returns, one row per date.
        alpha: Tail level.

    Returns:
        The minimum CDaR.
    """
    import cvxpy as cp

    t_steps, n_assets = returns.shape
    w = cp.Variable(n_assets, nonneg=True)
    peak = cp.Variable(t_steps)
    cumulative = cp.cumsum(returns @ w)
    problem = cp.Problem(
        cp.Minimize(cp.cvar(peak - cumulative, 1.0 - alpha)),
        [cp.sum(w) == 1, peak >= cumulative, peak[1:] >= peak[:-1], peak[0] >= 0],
    )
    problem.solve()
    return float(problem.value)


def marchenko_pastur_variance(
    eigenvalues: NDArray[np.float64], q: float, bandwidth: float = 0.01
) -> float:
    """Noise variance fitting the Marchenko-Pastur density to an exact kernel sum.

    Evaluates every Gaussian kernel at every grid point on each candidate grid,
    ``N x 1000`` exponentials per evaluation, where the rewrite bins the data
    once, convolves by FFT and interpolates.

    Args:
        eigenvalues: Sample correlation eigenvalues.
        q: Ratio of observations to variables.
        bandwidth: Kernel bandwidth.

    Returns:
        The fitted noise variance.
    """

    def loss(var: float) -> float:
        lam_minus = var * (1.0 - np.sqrt(1.0 / q)) ** 2
        lam_plus = var * (1.0 + np.sqrt(1.0 / q)) ** 2
        grid = np.linspace(lam_minus, lam_plus, 1000)
        pdf_mp = (q / (2.0 * np.pi * var * grid)) * np.sqrt(
            np.clip((lam_plus - grid) * (grid - lam_minus), 0.0, None)
        )
        kernels = np.exp(-0.5 * ((grid[None, :] - eigenvalues[:, None]) / bandwidth) ** 2)
        pdf_emp = kernels.mean(axis=0) / (bandwidth * np.sqrt(2.0 * np.pi))
        return float(np.sum((pdf_emp - pdf_mp) ** 2))

    return float(minimize_scalar(loss, bounds=(1e-5, 1.0 - 1e-5), method="bounded").x)


def whole_shares(targets: NDArray[np.float64], prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Share counts minimising L1 deviation plus leftover cash, in budget units.

    Linearises ``|w_i - p_i x_i|`` with two rows per asset and leaves the counts
    unbounded. The objective sits near minus one, so the relative gap has to be
    closed to zero, where the rewrite minimises shortfall alone and stops at the
    default absolute gap.

    Args:
        targets: Target weights, summing to one.
        prices: Share prices divided by the budget.

    Returns:
        Share count per asset.
    """
    n = prices.size
    price, ident = np.diag(prices), np.eye(n)
    rows = np.block([[-price, -ident], [price, -ident], [prices[None, :], np.zeros((1, n))]])
    result = milp(
        c=np.concatenate([-prices, np.ones(n)]),
        constraints=LinearConstraint(rows, -np.inf, np.concatenate([-targets, targets, [1.0]])),
        integrality=np.concatenate([np.ones(n), np.zeros(n)]),
        bounds=Bounds(0.0, np.inf),
        options={"mip_rel_gap": 0.0},
    )
    return np.rint(result.x[:n])


def transition_counts(violations: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Count violation transitions with four boolean passes.

    Args:
        violations: Violation indicator per period.

    Returns:
        ``(n00, n01, n10, n11)``.
    """
    previous = violations[:-1].astype(np.int64)
    current = violations[1:].astype(np.int64)
    return (
        int(np.sum((previous == 0) & (current == 0))),
        int(np.sum((previous == 0) & (current == 1))),
        int(np.sum((previous == 1) & (current == 0))),
        int(np.sum((previous == 1) & (current == 1))),
    )


def evar_dual_representation(
    values: NDArray[np.float64], *, alpha: float = 0.05, kind: str = "return"
) -> float:
    """EVaR of a sample by the dual representation of Ahmadi-Javid (2012).

    Theorem 3.3 writes ``EVaR = sup E_Q[L]`` over the measures ``Q`` whose
    relative entropy to the empirical measure is at most ``log(1/alpha)``. On
    ``T`` scenarios that is a maximum of ``q'L`` over the simplex with
    ``sum q log(q T) <= log(1/alpha)``, attained on a compact set, where the
    primal infimum over the EVaR parameter is not attained once the largest
    loss carries probability ``alpha``. The scalar search this replaced bounded
    the parameter ``1/t`` by 1000, which excluded the optimum of a sample whose
    losses are of the order of a tenth of a per cent. The losses are scaled to
    unit magnitude, and Clarabel's step is held to nine tenths of the distance
    to the cone boundary, since its default of 0.99 stalled on about one sample
    in sixty. Needs the ``[optim]`` extra.

    Args:
        values: Loss or return sample.
        alpha: Tail level.
        kind: ``"loss"`` if values are positive-loss, ``"return"`` otherwise.

    Returns:
        The entropic value at risk.
    """
    import cvxpy as cp

    losses = np.asarray(values, dtype=np.float64) * (1.0 if kind == "loss" else -1.0)
    scale = float(np.abs(losses).max()) or 1.0
    q = cp.Variable(losses.size, nonneg=True)
    problem = cp.Problem(
        cp.Maximize(q @ (losses / scale)),
        [
            cp.sum(q) == 1,
            cp.sum(cp.rel_entr(q, np.full(losses.size, 1.0 / losses.size))) <= -np.log(alpha),
        ],
    )
    problem.solve(solver=cp.CLARABEL, max_step_fraction=0.9)
    return float(problem.value) * scale


def min_evar_cone_weights(returns: pd.DataFrame, *, alpha: float = 0.05) -> pd.Series:
    """Long-only minimum-EVaR weights from the exponential-cone programme.

    One cone per date, the form the smooth programme of Ahmadi-Javid and
    Fallah-Tafti (2019) replaced. The returns are divided by their standard
    deviation, which positive homogeneity allows, and Clarabel's step is held
    to nine tenths of the distance to the cone boundary, since its default of
    0.99 stalled on about one panel in six hundred. Needs the ``[optim]`` extra.

    Args:
        returns: Asset returns, one row per date.
        alpha: Tail level.

    Returns:
        Weights indexed by ticker.
    """
    import cvxpy as cp

    r = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape
    w = cp.Variable(n_assets, nonneg=True)
    t = cp.Variable()
    z = cp.Variable(nonneg=True)
    u = cp.Variable(t_steps)
    problem = cp.Problem(
        cp.Minimize(t - z * np.log(alpha * t_steps)),
        [
            cp.sum(w) == 1,
            cp.sum(u) <= z,
            cp.constraints.ExpCone(-(r @ w) / r.std() - t, z * np.ones(t_steps), u),
        ],
    )
    problem.solve(solver=cp.CLARABEL, max_step_fraction=0.9)
    return pd.Series(np.asarray(w.value), index=returns.columns)


def max_diversification_ratio_weights(
    returns: pd.DataFrame, *, cov_matrix: pd.DataFrame | None = None
) -> pd.Series:
    """Long-only weights maximising the diversification ratio directly.

    Choueifaty and Coignard (2008) define the most-diversified portfolio as the
    maximiser of ``w' sigma / sqrt(w' Sigma w)`` over the simplex, a
    quasi-concave ratio, solved here by SLSQP with its analytic gradient.

    Args:
        returns: Asset returns; columns are tickers.
        cov_matrix: Covariance to use. Defaults to Ledoit-Wolf shrinkage.

    Returns:
        Weights indexed by ticker.
    """
    cov_df = linear_shrinkage_covariance(returns) if cov_matrix is None else cov_matrix
    covariance = cov_df.to_numpy(dtype=np.float64)
    volatilities = np.sqrt(np.diag(covariance))
    n_assets = volatilities.size

    def negative_ratio(w: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        variance = w @ covariance @ w
        spread = w @ volatilities
        ratio = spread / np.sqrt(variance)
        gradient = volatilities / np.sqrt(variance) - spread * (covariance @ w) / variance**1.5
        return -ratio, -gradient

    result = minimize(
        negative_ratio,
        np.full(n_assets, 1.0 / n_assets),
        jac=True,
        method="SLSQP",
        bounds=Bounds(0.0, 1.0),
        constraints=[LinearConstraint(np.ones((1, n_assets)), 1.0, 1.0)],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return pd.Series(result.x, index=cov_df.columns)


def mean_semideviation_lp_weights(
    returns: pd.DataFrame, *, risk_aversion: float = 1.0, measure: str = "absolute"
) -> pd.Series:
    """Long-only mean-semideviation weights with the deviations as variables.

    The absolute model is the linear programme of Mansini, Ogryczak and Speranza
    (2003, constraints 28 to 31), one deviation ``d_t >= mu'w - r_t'w`` per
    date, solved by HiGHS. The standard model maximises
    ``mu'w - lambda sqrt(mean(d^2))`` by SLSQP, the root being differentiable
    wherever the semideviation is positive.

    Args:
        returns: Asset returns; rows are equally likely scenarios.
        risk_aversion: Trade-off ``lambda``.
        measure: ``"absolute"`` or ``"standard"``.

    Returns:
        Weights indexed by ticker.
    """
    r = returns.to_numpy(dtype=np.float64)
    t_steps, n_assets = r.shape
    mu = r.mean(axis=0)
    below = mu - r
    if measure == "absolute":
        result = linprog(
            np.concatenate([-mu, np.full(t_steps, risk_aversion / t_steps)]),
            A_ub=np.hstack([below, -np.eye(t_steps)]),
            b_ub=np.zeros(t_steps),
            A_eq=np.concatenate([np.ones(n_assets), np.zeros(t_steps)])[None, :],
            b_eq=[1.0],
            bounds=(0.0, None),
            method="highs",
        )
        return pd.Series(result.x[:n_assets], index=returns.columns)

    def negative_objective(w: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        shortfall = np.maximum(below @ w, 0.0)
        deviation = np.sqrt(shortfall @ shortfall / t_steps)
        gradient = below.T @ shortfall / (t_steps * deviation) if deviation > 0.0 else 0.0 * mu
        return -(mu @ w) + risk_aversion * deviation, -mu + risk_aversion * gradient

    result = minimize(
        negative_objective,
        np.full(n_assets, 1.0 / n_assets),
        jac=True,
        method="SLSQP",
        bounds=Bounds(0.0, 1.0),
        constraints=[LinearConstraint(np.ones((1, n_assets)), 1.0, 1.0)],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return pd.Series(result.x, index=returns.columns)


def mean_semideviation_objective(
    returns: pd.DataFrame, weights: pd.Series, *, risk_aversion: float, measure: str
) -> float:
    """Mean return less ``risk_aversion`` times the semideviation, at given weights.

    The absolute model is a linear programme whose optimal weights need not be
    unique, so two solvers are compared on the objective they reach.

    Args:
        returns: Asset returns; rows are equally likely scenarios.
        weights: Portfolio weights.
        risk_aversion: Trade-off ``lambda``.
        measure: ``"absolute"`` or ``"standard"``.

    Returns:
        The objective of Ogryczak and Ruszczynski (1999).
    """
    portfolio = returns.to_numpy(dtype=np.float64) @ weights.to_numpy(dtype=np.float64)
    shortfall = np.maximum(portfolio.mean() - portfolio, 0.0)
    risk = shortfall.mean() if measure == "absolute" else np.sqrt(np.mean(shortfall**2))
    return float(portfolio.mean() - risk_aversion * risk)
