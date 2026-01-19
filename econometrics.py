# Portfolio_Theory/econometrics.py

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.stats import jarque_bera  # type: ignore

# Ignore specific statsmodels imports if stubs are missing
from statsmodels.tsa.stattools import adfuller  # type: ignore
from statsmodels.stats.diagnostic import (  # type: ignore
    acorr_ljungbox,
    het_arch,
    het_breuschpagan,
    breaks_cusumolsresid,
)
import statsmodels.api as sm  # type: ignore
from functools import reduce
from typing import List, Tuple, Dict, Any, Optional, Callable


class Econometrics:
    """Runs a suite of econometric tests on a DataFrame of asset returns."""

    def __init__(self, returnsDf: pd.DataFrame):
        """Initialise with a DataFrame of returns.

        Args:
            returnsDf (pd.DataFrame): DataFrame where columns are asset tickers
                                      and rows are returns indexed by date/time.
                                      NaN values will be dropped row-wise.
        """
        self.returnsDf: pd.DataFrame = returnsDf.dropna()

    @staticmethod
    def _run_normality(series: pd.Series) -> Tuple[float, float]:
        """Executes the Jarque-Bera test."""
        jbStat, jbPval = jarque_bera(series.to_numpy())
        return jbStat, jbPval

    @staticmethod
    def _run_stationarity(
        series: pd.Series, regression: str = "c", autolag: Optional[str] = "AIC"
    ) -> Tuple[float, float]:
        """Executes the Augmented Dickey-Fuller test."""
        # Returns: adf_stat, p_value, used_lag, nobs, critical_values, ic_best
        adfStat, pVal, _, _, _, _ = adfuller(
            series.to_numpy(), regression=regression, autolag=autolag
        )
        return adfStat, pVal

    @staticmethod
    def _run_autocorrelation(series: pd.Series, lags: int) -> Tuple[float, float]:
        """Executes the Ljung-Box test up to a given lag."""
        # Test autocorrelation up to specified lag; model_df=0 for raw series.
        res: pd.DataFrame = acorr_ljungbox(
            series.to_numpy(), lags=lags, return_df=True, model_df=0
        )
        # Report result for the maximum lag tested.
        lbStat = res["lb_stat"].iloc[-1]
        lbPval = res["lb_pvalue"].iloc[-1]
        return lbStat, lbPval

    @staticmethod
    def _run_heteroskedasticity(series: pd.Series) -> Tuple[float, float]:
        """Executes the Breusch-Pagan test (Koenker version)."""
        # Regress squared residuals on a time trend.
        exog: NDArray[np.float64] = sm.add_constant(np.arange(len(series)))
        # Use robust (Koenker) version against non-normality.
        # Returns: LM stat, LM p-val, F stat, F p-val.
        lmStat, lmPval, _, _ = het_breuschpagan(series.to_numpy(), exog, robust=True)
        return lmStat, lmPval

    @staticmethod
    def _run_arch_effect(series: pd.Series, lags: int) -> Tuple[float, float]:
        """Executes the ARCH-LM test."""
        # Regress squared residuals on lagged squared residuals.
        # Returns: LM stat, LM p-val, F stat, F p-val.
        lmStat, lmPval, _, _ = het_arch(series.to_numpy(), nlags=lags)
        return lmStat, lmPval

    @staticmethod
    def _run_structural_break(series: pd.Series) -> Tuple[float, float]:
        """Executes the CUSUM test on OLS residuals."""
        # Fit model against constant only (mean estimation).
        exog = sm.add_constant(np.ones_like(series))
        model = sm.OLS(series.to_numpy(), exog)
        res = model.fit()
        # Test stability using cumulative residuals; ddof=1 for intercept.
        # Returns: CUSUM stat, p-val, critical values dict.
        cusumStat, pVal, _ = breaks_cusumolsresid(res.resid, ddof=1)
        return cusumStat, pVal

    def _apply_test_to_all(
        self,
        testRunner: Callable[..., Tuple[float, float]],
        columns: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Applies a specific test runner function to all tickers."""
        results: Dict[str, Tuple[float, float]] = {}
        for ticker in self.returnsDf.columns:
            try:
                results[ticker] = testRunner(self.returnsDf[ticker], **kwargs)
            except Exception as e:
                # Store NaN if test fails for a specific series.
                print(f"Warning: Test failed for {ticker}: {e}")
                results[ticker] = (np.nan, np.nan)
        return pd.DataFrame.from_dict(results, orient="index", columns=columns)

    def normality_test(self) -> pd.DataFrame:
        """Perform Jarque-Bera test for normality on each return series.

        Tests the null hypothesis that the sample data has the skewness and
        kurtosis matching a normal distribution.

        Returns:
            pd.DataFrame: DataFrame with Jarque-Bera statistic ('JB_stat') and
                          p-value ('JB_pval') for each asset.
        """
        return self._apply_test_to_all(
            self._run_normality, columns=["JB_stat", "JB_pval"]
        )

    def stationarity_test(
        self, regression: str = "c", autolag: Optional[str] = "AIC"
    ) -> pd.DataFrame:
        """Perform Augmented Dickey-Fuller test for stationarity (Unit Root Test).

        Tests the null hypothesis that a unit root is present (non-stationary).
        The alternative hypothesis is stationarity or trend-stationarity.
        Allows specification of regression trend and lag selection method.

        Args:
            regression (str, optional): Constant/trend order ('c', 'ct', 'ctt', 'n').
                                       Defaults to 'c' (constant only).
            autolag (Optional[str], optional): Method for lag length selection
                                              ('AIC', 'BIC', 't-stat', None).
                                              Defaults to 'AIC'.

        Returns:
            pd.DataFrame: DataFrame with ADF statistic ('ADF_stat') and
                          p-value ('ADF_pval') for each asset. Low p-value rejects null (suggests stationarity).
        """
        return self._apply_test_to_all(
            self._run_stationarity,
            columns=["ADF_stat", "ADF_pval"],
            regression=regression,
            autolag=autolag,
        )

    def autocorrelation_test(self, lags: int = 10) -> pd.DataFrame:
        """Perform Ljung-Box test for autocorrelation up to a specified lag.

        Tests the null hypothesis that autocorrelations *up to* the specified lag
        are jointly zero. Reports the test statistic and p-value for the maximum lag.

        Args:
            lags (int, optional): Maximum number of lags to include in the test statistic.
                                  Defaults to 10. Assumes testing raw returns (model_df=0).

        Returns:
            pd.DataFrame: DataFrame with Ljung-Box statistic ('LB_stat') and
                          p-value ('LB_pval') evaluated at the specified max lag.
        """
        return self._apply_test_to_all(
            self._run_autocorrelation, columns=["LB_stat", "LB_pval"], lags=lags
        )

    def heteroskedasticity_test(self) -> pd.DataFrame:
        """Perform Breusch-Pagan test (Koenker version) for heteroskedasticity.

        Tests the null hypothesis of homoskedasticity (constant variance)
        against the alternative of heteroskedasticity, using a regression
        of squared residuals on exogenous variables (here, just a time trend).
        Uses the studentized (Koenker) version robust to non-normality.

        Returns:
            pd.DataFrame: DataFrame with Breusch-Pagan LM statistic ('BP_LM_stat')
                          and p-value ('BP_LM_pval') for each asset.
        """
        return self._apply_test_to_all(
            self._run_heteroskedasticity, columns=["BP_LM_stat", "BP_LM_pval"]
        )

    def arch_effect_test(self, lags: int = 10) -> pd.DataFrame:
        """Perform Engle's ARCH-LM test for volatility clustering.

        Tests the null hypothesis that there are no ARCH effects (autoregressive
        conditional heteroskedasticity) up to the specified lag in the residuals,
        by regressing squared residuals on lagged squared residuals.

        Args:
            lags (int, optional): Number of lags to include. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame with ARCH LM statistic ('ARCH_LM_stat') and
                          p-value ('ARCH_LM_pval') for each asset.
        """
        return self._apply_test_to_all(
            self._run_arch_effect, columns=["ARCH_LM_stat", "ARCH_LM_pval"], lags=lags
        )

    def structural_break_test(self) -> pd.DataFrame:
        """Perform CUSUM test for structural breaks in OLS residuals' mean.

        Tests the stability of coefficients (here, just the mean) over time
        by examining the cumulative sum of residuals from an OLS regression
        on a constant.

        Returns:
            pd.DataFrame: DataFrame with CUSUM statistic ('CUSUM_stat') and
                          p-value ('CUSUM_pval') for each asset.
        """
        return self._apply_test_to_all(
            self._run_structural_break, columns=["CUSUM_stat", "CUSUM_pval"]
        )

    def run_all_tests(
        self,
        adfRegression: str = "c",
        acLags: int = 10,
        archLags: int = 10,
    ) -> pd.DataFrame:
        """Run all econometric tests and combine results.

        Executes normality, ADF stationarity, autocorrelation, heteroskedasticity,
        ARCH effects, and structural break tests, merging their results into
        a single summary DataFrame. Allows customization of test parameters.

        Args:
            adfRegression (str, optional): Regression type for ADF test ('c','ct','ctt','n'). Defaults to 'c'.
            acLags (int, optional): Lags for autocorrelation test. Defaults to 10.
            archLags (int, optional): Lags for ARCH effect test. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame indexed by asset ticker, containing results
                          from all performed tests.
        """
        tests = [
            self.normality_test(),
            self.stationarity_test(regression=adfRegression),
            self.autocorrelation_test(lags=acLags),
            self.heteroskedasticity_test(),
            self.arch_effect_test(lags=archLags),
            self.structural_break_test(),
        ]
        # Use reduce with DataFrame.join for efficient merging on index
        summaryDf = reduce(lambda left, right: left.join(right, how="outer"), tests)
        return summaryDf
