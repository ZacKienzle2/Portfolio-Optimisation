# 9. Risk measures and backtesting

- Status: Accepted
- Date: 2026-06-20

## Context

Value-at-Risk alone is neither coherent nor sufficient for tail risk, and a risk
model is only credible if its forecasts are validated against realised outcomes.

## Decision

The risk package provides Value-at-Risk and Conditional VaR, the coherent
Entropic VaR, spectral and Wang distortion measures, and Extreme Value Theory
tail estimates. Forecasts are validated with coverage backtests (Kupiec,
Christoffersen, and the Acerbi-Szekely Expected Shortfall test), and GARCH
conditional-volatility forecasts feed those backtests directly.

## Consequences

- Risk is measured with coherent and tail-aware measures, not VaR alone.
- Forecasts are validated rather than merely computed.
- The model, forecast and validation steps form one connected workflow.
