"""Bootstrap robustness analysis for hierarchical risk parity allocations."""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from scipy.stats import gmean

from portfolio_optimisation.infra.logging import get_logger
from portfolio_optimisation.optim.hrp import HRPModel
from portfolio_optimisation.optim.shrinkage import linear_shrinkage_covariance

logger = get_logger(__name__)

_PERIODS_PER_YEAR = 252


def _resample_statistics(
    returns: pd.DataFrame,
    block_size: int,
    risk_free_rate: float,
    job: tuple[int, str],
) -> dict[str, float]:
    """Annualised return, volatility and Sharpe ratio of HRP on one resample.

    Args:
        returns: Historical asset returns.
        block_size: Expected stationary-bootstrap block length.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.
        job: Seed of the resample and the linkage method to cluster with.

    Returns:
        The resample's ``exp_return``, ``volatility`` and ``sharpe_ratio``.
    """
    seed, linkage_method = job
    bootstrap = StationaryBootstrap(block_size, returns, seed=np.random.default_rng(seed))
    sample: pd.DataFrame = next(bootstrap.bootstrap(1))[0][0]
    covariance = linear_shrinkage_covariance(sample)
    model = HRPModel(sample, cov_matrix=covariance)
    model.optimize(linkage_method=linkage_method)
    weights = model.clean_weights().reindex(sample.columns).to_numpy()

    growth = gmean(1.0 + sample.to_numpy(dtype=np.float64), axis=0)
    exp_return = float(weights @ (growth**_PERIODS_PER_YEAR - 1.0))
    volatility = float(np.sqrt(weights @ covariance.to_numpy() @ weights * _PERIODS_PER_YEAR))
    sharpe = (exp_return - risk_free_rate) / volatility if volatility > 1e-9 else 0.0
    return {"exp_return": exp_return, "volatility": volatility, "sharpe_ratio": sharpe}


class HRPAnalyser:
    """Stability and performance analysis of HRP under the stationary bootstrap.

    Resamples the returns with the stationary bootstrap and rebuilds the HRP
    portfolio for each linkage method on every resample, so the spread of the
    resulting statistics measures how robust each clustering choice is.

    Args:
        returns: Historical asset returns.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> None:
        self.returns: pd.DataFrame = returns
        self.risk_free_rate: float = risk_free_rate
        self.bootstrap_results: pd.DataFrame | None = None
        self.cache_dir: Path = Path("cache")
        self._block_size: int = int(optimal_block_length(returns**2).iloc[:, 0].mean())

    def _get_cache_path(self, reps: int, linkage_methods: list[str], *, paired: bool) -> Path:
        """Cache path keyed by a digest of everything that changes the result.

        Args:
            reps: Bootstrap repetitions per method.
            linkage_methods: Linkage methods evaluated.
            paired: Whether the methods share resamples.

        Returns:
            Path of the parquet cache for this request.
        """
        key = "|".join(
            [
                ",".join(sorted(self.returns.columns)),
                ",".join(sorted(linkage_methods)),
                str(reps),
                str(self._block_size),
                repr(self.risk_free_rate),
                str(paired),
            ]
        )
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return self.cache_dir / f"bootstrap_{digest}.parquet"

    def run_bootstrap(
        self,
        linkage_methods: list[str] | None = None,
        reps: int = 500,
        verbose: bool = True,
        force_recalculate: bool = False,
        *,
        seed: int | None = None,
        paired: bool = False,
    ) -> None:
        """Run the stationary bootstrap for several HRP linkage methods.

        Results load from the parquet cache unless ``force_recalculate`` is set.
        The resamples run in this process. One costs about a millisecond on six
        assets, while a spawned worker pays several seconds importing the
        numerical stack, so a process pool was slower at every size measured.

        Args:
            linkage_methods: Linkage methods. Defaults to ward, single, complete and
                average.
            reps: Bootstrap repetitions per method.
            verbose: Emit status logs at INFO level.
            force_recalculate: Ignore the cache and recompute.
            seed: Seed for the generator that draws the per-repetition seeds.
            paired: Evaluate every method on the same resamples, which reduces the
                variance of cross-method comparisons.
        """
        if linkage_methods is None:
            linkage_methods = ["ward", "single", "complete", "average"]

        cache_path = self._get_cache_path(reps, linkage_methods, paired=paired)
        if cache_path.exists() and not force_recalculate:
            if verbose:
                logger.info("Loading bootstrap results from cache...")
            self.bootstrap_results = pd.read_parquet(cache_path)
            return

        if verbose:
            logger.info("Running bootstrap with %d reps...", reps)
        rng = np.random.default_rng(seed)
        shared = rng.integers(0, 2**32 - 1, size=reps) if paired else None
        jobs = [
            (int(value), method)
            for method in linkage_methods
            for value in (shared if shared is not None else rng.integers(0, 2**32 - 1, size=reps))
        ]
        task = partial(_resample_statistics, self.returns, self._block_size, self.risk_free_rate)
        results = list(map(task, jobs))

        frame = pd.DataFrame(results)
        frame["linkage_method"] = [method for _, method in jobs]
        self.bootstrap_results = frame
        self.cache_dir.mkdir(exist_ok=True)
        frame.to_parquet(cache_path)
        if verbose:
            logger.info("Bootstrap complete and results saved to cache.")
