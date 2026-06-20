import hashlib
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numpy.typing import NDArray
from pypfopt import expected_returns
from sklearn.covariance import ledoit_wolf

from portfolio_optimisation.infra.logging import get_logger
from portfolio_optimisation.optim.hrp import HRPModel

logger = get_logger(__name__)


class HRPAnalyser:
    """Performs stability and performance analysis on the HRP model using bootstrap.

    Runs a stationary bootstrap simulation to test the robustness of HRP
    portfolios constructed using various clustering linkage methods.
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """Initialise the analyser.

        Args:
            returns (pd.DataFrame): Historical asset returns.
            risk_free_rate (float, optional): Annual risk-free rate for Sharpe
                                            ratio calculation. Defaults to 0.02.
        """
        self.returns: pd.DataFrame = returns
        self.risk_free_rate: float = risk_free_rate
        self.bootstrap_results: pd.DataFrame | None = None
        self.cache_dir: Path = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._block_size: int = self._compute_block_size(returns)

    @staticmethod
    def _compute_block_size(returns: pd.DataFrame) -> int:
        """Optimal stationary-bootstrap block length on squared returns.

        Hoisted out of the per-iteration worker because it depends only on
        the full sample, not on the bootstrap seed.
        """
        block_lengths = optimal_block_length(returns**2)
        return int(block_lengths.iloc[:, 0].mean())

    def _get_cache_path(self, reps: int, linkage_methods: list[str], *, paired: bool) -> Path:
        """Stable cache path derived from the request.

        Keyed on the sorted ticker set, linkage methods, repetition count,
        block size, risk-free rate and pairing mode via blake2b. The builtin
        ``hash()`` is salted per interpreter run (PYTHONHASHSEED), so the
        previous key was non-deterministic across processes and effectively
        never hit the cache between sessions; it also ignored every parameter
        that changes the result.
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
    ):
        """Run the stationary bootstrap analysis for multiple HRP linkage methods.

        Loads results from cache if available and ``force_recalculate`` is
        False. Uses multiprocessing to speed up the numerous HRP optimisations.

        Args:
            linkage_methods (list[str] | None): Linkage methods. Defaults to
                ``["ward", "single", "complete", "average"]``.
            reps (int): Number of bootstrap repetitions. Defaults to 500.
            verbose (bool): Emit status logs at INFO level. Defaults to True.
            force_recalculate (bool): Ignore cache and recompute. Defaults to
                False.
            seed (int | None): Seed for the master generator that draws the
                per-repetition seeds. Pass an int for reproducible runs.
            paired (bool): If True, every linkage method is evaluated on the
                same bootstrap resamples (shared per-rep seeds), reducing the
                variance of cross-method comparisons. Defaults to False.
        """
        if linkage_methods is None:
            linkage_methods = ["ward", "single", "complete", "average"]

        cache_path = self._get_cache_path(reps, linkage_methods, paired=paired)
        if cache_path.exists() and not force_recalculate:
            if verbose:
                logger.info("Loading bootstrap results from cache...")
            self.bootstrap_results = pd.read_parquet(cache_path)
            if verbose:
                logger.info("Bootstrap results loaded.")
            return

        if verbose:
            logger.info("Running bootstrap with %d reps...", reps)

        n_cores = max(1, cpu_count() - 1)
        rng = np.random.default_rng(seed)
        # One job-list spanning all (seed, method) pairs so a single worker
        # pool amortises spawn cost across every linkage method. In paired mode
        # all methods share one set of per-rep seeds so method differences are
        # measured on identical resamples.
        shared_seeds: NDArray[np.int64] | None = (
            rng.integers(0, 2**32 - 1, size=reps, dtype=np.int64) if paired else None
        )
        jobs: list[tuple[int, str]] = []
        for method in linkage_methods:
            method_seeds: NDArray[np.int64] = (
                shared_seeds
                if shared_seeds is not None
                else rng.integers(0, 2**32 - 1, size=reps, dtype=np.int64)
            )
            jobs.extend((int(seed_value), method) for seed_value in method_seeds)

        with Pool(n_cores) as pool:
            results: list[dict[str, float]] = pool.starmap(self._bootstrap_single_method, jobs)

        method_column = [method for method in linkage_methods for _ in range(reps)]
        df = pd.DataFrame(results)
        df["linkage_method"] = method_column
        self.bootstrap_results = df

        self.bootstrap_results.to_parquet(cache_path)
        if verbose:
            logger.info("Bootstrap complete and results saved to cache.")

    def _bootstrap_single_method(self, seed: int, linkage_method: str) -> dict[str, float]:
        """Performs a single HRP optimisation on a bootstrapped sample."""
        rs = np.random.default_rng(seed)
        # _block_size hoisted in __init__; reuse instead of recomputing per iter.
        bs = StationaryBootstrap(self._block_size, self.returns, seed=rs)

        sample_data = next(iter(bs.bootstrap(1)))
        sample_returns: pd.DataFrame = sample_data[0][0]

        if isinstance(sample_returns, np.ndarray):
            sample_returns = pd.DataFrame(sample_returns, columns=self.returns.columns)

        mu = expected_returns.mean_historical_return(
            sample_returns, returns_data=True, frequency=252
        )
        s_matrix = self._calculate_covariance(sample_returns)
        weights_result = self._calculate_hrp_weights(s_matrix, sample_returns, linkage_method)

        exp_return = (weights_result * mu).sum()
        volatility = np.sqrt(weights_result.T @ s_matrix @ weights_result) * np.sqrt(252)
        sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 1e-9 else 0.0

        return {
            "exp_return": float(exp_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
        }

    @staticmethod
    def _calculate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Ledoit-Wolf shrunk covariance matrix from returns."""
        cov_matrix_internal, _ = ledoit_wolf(returns, assume_centered=False)
        return pd.DataFrame(cov_matrix_internal, index=returns.columns, columns=returns.columns)

    @staticmethod
    def _calculate_hrp_weights(
        s_matrix: pd.DataFrame, returns: pd.DataFrame, linkage: str
    ) -> pd.Series:
        """Instantiate and run HRP model given inputs."""
        hrp = HRPModel(returns=returns)
        hrp.cov_matrix = s_matrix
        hrp.optimize(linkage_method=linkage)
        return hrp.clean_weights()
