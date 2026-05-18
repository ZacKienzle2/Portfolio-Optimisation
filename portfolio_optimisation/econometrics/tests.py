from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    def __init__(self, returnsDf: pd.DataFrame):
        """Initialise with a DataFrame of returns.

        Args:
            returnsDf (pd.DataFrame): Columns are asset tickers, rows are
                                      returns indexed by date/time. NaN rows dropped.
        """
        self.returnsDf: pd.DataFrame = returnsDf.dropna()

    @staticmethod
    def _run_normality(series: pd.Series) -> Tuple[float, float]:
        """Jarque-Bera test."""
        jbStat, jbPval = jarque_bera(series.to_numpy())
        return jbStat, jbPval

    @staticmethod
    def _run_stationarity(
        series: pd.Series, regression: str = "c", autolag: Optional[str] = "AIC"
    ) -> Tuple[float, float]:
        """Augmented Dickey-Fuller test."""
        adfStat, pVal, _, _, _, _ = adfuller(
            series.to_numpy(), regression=regression, autolag=autolag
        )
        return adfStat, pVal

    @staticmethod
    def _run_autocorrelation(series: pd.Series, lags: int) -> Tuple[float, float]:
        """Ljung-Box test up to a given lag."""
        res: pd.DataFrame = acorr_ljungbox(
            series.to_numpy(), lags=lags, return_df=True, model_df=0
        )
        lbStat = res["lb_stat"].iloc[-1]
        lbPval = res["lb_pvalue"].iloc[-1]
        return lbStat, lbPval

    @staticmethod
    def _run_heteroskedasticity(series: pd.Series) -> Tuple[float, float]:
        """Breusch-Pagan (Koenker) test."""
        exog: NDArray[np.float64] = sm.add_constant(np.arange(len(series)))
        lmStat, lmPval, _, _ = het_breuschpagan(series.to_numpy(), exog, robust=True)
        return lmStat, lmPval

    @staticmethod
    def _run_arch_effect(series: pd.Series, lags: int) -> Tuple[float, float]:
        """ARCH-LM test."""
        lmStat, lmPval, _, _ = het_arch(series.to_numpy(), nlags=lags)
        return lmStat, lmPval

    @staticmethod
    def _run_structural_break(series: pd.Series) -> Tuple[float, float]:
        """CUSUM test on OLS residuals."""
        exog = sm.add_constant(np.ones_like(series))
        model = sm.OLS(series.to_numpy(), exog)
        res = model.fit()
        cusumStat, pVal, _ = breaks_cusumolsresid(res.resid, ddof=1)
        return cusumStat, pVal

    def _apply_test_to_all(
        self,
        testRunner: Callable[..., Tuple[float, float]],
        columns: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Applies a test runner function to all tickers."""
        results: Dict[str, Tuple[float, float]] = {}
        for ticker in self.returnsDf.columns:
            try:
                results[ticker] = testRunner(self.returnsDf[ticker], **kwargs)
            except Exception as e:
                print(f"Warning: Test failed for {ticker}: {e}")
                results[ticker] = (np.nan, np.nan)
        return pd.DataFrame.from_dict(results, orient="index", columns=columns)

    def normality_test(self) -> pd.DataFrame:
        """Jarque-Bera test for normality on each return series."""
        return self._apply_test_to_all(
            self._run_normality, columns=["JB_stat", "JB_pval"]
        )

    def stationarity_test(
        self, regression: str = "c", autolag: Optional[str] = "AIC"
    ) -> pd.DataFrame:
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
        adfRegression: str = "c",
        acLags: int = 10,
        archLags: int = 10,
    ) -> pd.DataFrame:
        """Run all econometric tests and join into a single summary."""
        tests = [
            self.normality_test(),
            self.stationarity_test(regression=adfRegression),
            self.autocorrelation_test(lags=acLags),
            self.heteroskedasticity_test(),
            self.arch_effect_test(lags=archLags),
            self.structural_break_test(),
        ]
        summaryDf = reduce(lambda left, right: left.join(right, how="outer"), tests)
        return summaryDf
