# Econometrics

Derivations for the diagnostic battery in
`portfolio_optimisation.econometrics`. The shared notation is defined on the
[methodology index](index.md). Each test is applied per asset to the return
series $\{ y_t \}_{t=1}^{T}$ and reports a statistic with its p-value.

## Normality

The Jarque-Bera test checks whether the third and fourth moments match a normal.
With sample skewness $S$ and kurtosis $K$,

$$
\mathrm{JB} = \frac{T}{6}\left( S^2 + \frac{(K - 3)^2}{4} \right),
$$

which is $\chi^2_2$ under normality, since the standardised skewness and excess
kurtosis are asymptotically independent normals. A large statistic rejects
normality, the usual outcome for daily returns, which motivates the fat-tailed
marginals and the tail-risk measures used elsewhere.

## Stationarity

The augmented Dickey-Fuller test asks whether a series has a unit root. It
regresses the first difference on the lagged level and its own lags,

$$
\Delta y_t = \alpha + \gamma\, y_{t-1}
+ \sum_{i=1}^{p} \delta_i\, \Delta y_{t-i} + \varepsilon_t,
$$

and tests $H_0 : \gamma = 0$, a unit root and hence non-stationarity, against
$\gamma < 0$. The lag order $p$ absorbs short-run autocorrelation and is
chosen by the Akaike criterion. Under the null the statistic
$\hat{\gamma} / \operatorname{se}(\hat{\gamma})$ does not follow a normal but
the Dickey-Fuller distribution, whose left-tail critical values are used.

## Autocorrelation

The Ljung-Box test aggregates the first $h$ autocorrelations of the series. From
the sample autocorrelations $\hat{\rho}_k$,

$$
Q = T (T + 2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{T - k},
$$

which is $\chi^2_h$ when the series is white noise. The $(T+2)/(T-k)$
weighting corrects the small-sample bias of the raw Box-Pierce sum. Rejection
indicates predictable linear structure in the returns.

## Heteroskedasticity

The Breusch-Pagan test detects variance that depends on the regressors. It
regresses the squared residuals on the explanatory variables, here a constant and
a time trend, and forms the Lagrange-multiplier statistic

$$
\mathrm{LM} = T R^2 \ \sim\ \chi^2_p,
$$

with $R^2$ from that auxiliary regression and $p$ its number of slopes. The
Koenker studentised form used here divides by the empirical fourth moment, which
keeps the test valid when the residuals are not normal.

## ARCH effects

Engle's test targets volatility clustering, where large moves follow large moves.
It regresses the squared residual on its own $q$ lags,

$$
\varepsilon_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i\, \varepsilon_{t-i}^2
+ u_t,
$$

and tests whether the lag coefficients are jointly zero through
$\mathrm{LM} = T R^2 \sim \chi^2_q$. Rejection is the empirical signature that
justifies the GARCH-family conditional-volatility models on the
[risk page](risk.md#conditional-volatility).

## Structural breaks

The cumulative-sum test checks whether the mean is stable through the sample. It
fits an ordinary least squares model and accumulates the scaled residuals,

$$
W_s = \frac{1}{\hat{\sigma} \sqrt{T}} \sum_{t=1}^{s} \hat{\varepsilon}_t ,
$$

whose maximum deviation from zero is compared with the boundary of a Brownian
bridge. Under the null of a stable mean the path stays within the boundary; a
crossing signals a structural break. This guards against pooling regimes that a
single static estimate would blur.
