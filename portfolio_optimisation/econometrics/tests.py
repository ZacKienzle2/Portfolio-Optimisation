from collections.abc import Callable
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
)
from statsmodels.tsa.stattools import adfuller


class Econometrics:
    """Runs a suite of econometric tests on a DataFrame of asset returns."""

    def __init__(self, returns_df: pd.DataFrame):
        """Initialise with a DataFrame of returns.

        Args:
            returns_df (pd.DataFrame): Columns are asset tickers, rows are
                                      returns indexed by date/time. NaN rows dropped.
        """
        self.returns_df: pd.DataFrame = returns_df.dropna()

    @staticmethod
    def _run_normality(series: pd.Series) -> tuple[float, float]:
        """Jarque-Bera test."""
        jb_stat, jb_pval = jarque_bera(series.to_numpy())
        return jb_stat, jb_pval

    @staticmethod
    def _run_stationarity(
        series: pd.Series, regression: str = "c", autolag: str | None = "AIC"
    ) -> tuple[float, float]:
        """Augmented Dickey-Fuller test."""
        adf_stat, p_val, _, _, _, _ = adfuller(
            series.to_numpy(), regression=regression, autolag=autolag
        )
        return adf_stat, p_val

    @staticmethod
    def _run_autocorrelation(series: pd.Series, lags: int) -> tuple[float, float]:
        """Ljung-Box test up to a given lag."""
        res: pd.DataFrame = acorr_ljungbox(series.to_numpy(), lags=lags, return_df=True, model_df=0)
        lb_stat = res["lb_stat"].iloc[-1]
        lb_pval = res["lb_pvalue"].iloc[-1]
        return lb_stat, lb_pval

    @staticmethod
    def _run_heteroskedasticity(series: pd.Series) -> tuple[float, float]:
        """Breusch-Pagan (Koenker) test."""
        exog: NDArray[np.float64] = sm.add_constant(np.arange(len(series)))
        lm_stat, lm_pval, _, _ = het_breuschpagan(series.to_numpy(), exog, robust=True)
        return lm_stat, lm_pval

    @staticmethod
    def _run_arch_effect(series: pd.Series, lags: int) -> tuple[float, float]:
        """ARCH-LM test."""
        lm_stat, lm_pval, _, _ = het_arch(series.to_numpy(), nlags=lags)
        return lm_stat, lm_pval

    @staticmethod
    def _run_structural_break(series: pd.Series) -> tuple[float, float]:
        """CUSUM test on OLS residuals."""
        exog = sm.add_constant(np.ones_like(series))
        model = sm.OLS(series.to_numpy(), exog)
        res = model.fit()
        cusum_stat, p_val, _ = breaks_cusumolsresid(res.resid, ddof=1)
        return cusum_stat, p_val

    def _apply_test_to_all(
        self,
        test_runner: Callable[..., tuple[float, float]],
        columns: list[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Applies a test runner function to all tickers."""
        results: dict[str, tuple[float, float]] = {}
        for ticker in self.returns_df.columns:
            try:
                results[ticker] = test_runner(self.returns_df[ticker], **kwargs)
            except Exception as e:
                print(f"Warning: Test failed for {ticker}: {e}")
                results[ticker] = (np.nan, np.nan)
        return pd.DataFrame.from_dict(results, orient="index", columns=columns)

    def normality_test(self) -> pd.DataFrame:
        """Jarque-Bera test for normality on each return series."""
        return self._apply_test_to_all(self._run_normality, columns=["JB_stat", "JB_pval"])

    def stationarity_test(self, regression: str = "c", autolag: str | None = "AIC") -> pd.DataFrame:
        """Augmented Dickey-Fuller test for stationarity (Unit Root Test)."""
        return self._apply_test_to_all(
            self._run_stationarity,
            columns=["ADF_stat", "ADF_pval"],
            regression=regression,
            autolag=autolag,
        )

    def autocorrelation_test(self, lags: int = 10) -> pd.DataFrame:
        """Ljung-Box test for autocorrelation up to a specified lag."""
        return self._apply_test_to_all(
            self._run_autocorrelation, columns=["LB_stat", "LB_pval"], lags=lags
        )

    def heteroskedasticity_test(self) -> pd.DataFrame:
        """Breusch-Pagan (Koenker) test for heteroskedasticity."""
        return self._apply_test_to_all(
            self._run_heteroskedasticity, columns=["BP_LM_stat", "BP_LM_pval"]
        )

    def arch_effect_test(self, lags: int = 10) -> pd.DataFrame:
        """Engle's ARCH-LM test for volatility clustering."""
        return self._apply_test_to_all(
            self._run_arch_effect, columns=["ARCH_LM_stat", "ARCH_LM_pval"], lags=lags
        )

    def structural_break_test(self) -> pd.DataFrame:
        """CUSUM test for structural breaks in OLS residuals' mean."""
        return self._apply_test_to_all(
            self._run_structural_break, columns=["CUSUM_stat", "CUSUM_pval"]
        )

    def run_all_tests(
        self,
        adf_regression: str = "c",
        ac_lags: int = 10,
        arch_lags: int = 10,
    ) -> pd.DataFrame:
        """Run all econometric tests and join into a single summary."""
        tests = [
            self.normality_test(),
            self.stationarity_test(regression=adf_regression),
            self.autocorrelation_test(lags=ac_lags),
            self.heteroskedasticity_test(),
            self.arch_effect_test(lags=arch_lags),
            self.structural_break_test(),
        ]
        summary_df = reduce(lambda left, right: left.join(right, how="outer"), tests)
        return summary_df
