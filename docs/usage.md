# Usage

## Python API

The service layer is the entry point for an end-to-end run. It fetches data
through an injected repository, builds an HRP allocation, simulates the return
distribution, and aggregates risk and performance metrics.

```python
from portfolio_optimisation import load_settings
from portfolio_optimisation.services import build_pipeline_from_settings

settings = load_settings(seed=12345)
pipeline = build_pipeline_from_settings(settings)
result = pipeline.run(
    tickers=["IYW", "VGT", "IYF"],
    start_date="2018-01-01",
    use_copula=True,
)

print(result.weights)
print(result.risk_metrics)
print(result.performance_metrics)
```

Individual components can also be used directly:

```python
from portfolio_optimisation.optim import HRPModel
from portfolio_optimisation.risk import CopulaRiskAnalyser, calculate_risk_metrics
from portfolio_optimisation.econometrics import Econometrics
from portfolio_optimisation.sde import SDEFitter
```

## Command line

```bash
portfolio-opt version
portfolio-opt config
portfolio-opt run --tickers IYW VGT IYF --start 2018-01-01 --output result.json
```

The `run` command emits the weights, risk metrics and performance metrics as
JSON, either to stdout or to the path given by `--output`.

## Configuration

Settings resolve with the precedence explicit argument > `PORTFOLIO_*`
environment variable > `portfolio.toml` > built-in default. A `portfolio.toml`
is read from the working directory when present:

```toml
[tool.portfolio]
risk_free_rate = 0.03
n_simulations = 20000
var_method = "parametric"
linkage_method = "single"
seed = 12345
```

Every Monte Carlo path accepts a `seed`; supplying one makes the run fully
reproducible.
