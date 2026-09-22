"""Implementations that a library call or a faster formulation replaced.

Each function is the version the package ran before its rewrite, kept so an
equivalence test can compare the rewrite against it on generated inputs. None
of them is called by the package itself.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp, minimize_scalar


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


def entropic_value_at_risk(losses: NDArray[np.float64], alpha: float) -> float:
    """EVaR with the log-sum-exp shifted by its maximum by hand.

    Replaced by ``scipy.special.logsumexp``, which computes the same sum and is
    slower per call on SciPy 1.18.

    Args:
        losses: Loss sample, positive for a loss.
        alpha: Tail level.

    Returns:
        The entropic value at risk.
    """
    log_scale = np.log(alpha * losses.size)

    def objective(z: float) -> float:
        scaled = z * losses
        peak = scaled.max()
        return float((peak + np.log(np.exp(scaled - peak).sum()) - log_scale) / z)

    return float(minimize_scalar(objective, bounds=(1e-6, 1e3), method="bounded").fun)


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
