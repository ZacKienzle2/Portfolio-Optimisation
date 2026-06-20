# 6. Visualisation style and decoupled plotting

- Status: Accepted
- Date: 2026-06-20

## Context

Figures used ad-hoc sizes, fonts and colour maps, so output was visually
inconsistent and not colour-blind safe. Plotting code also lived inside the
optimisation layer, coupling numerical routines to a plotting backend.

## Decision

A single `viz.style.configure_style` sets shared matplotlib rcParams, a
colour-blind-safe categorical palette and perceptually-uniform colour maps, with
a matching Plotly template. Plotting functions return the `(Figure, Axes)` pair
and save on request rather than calling `show`. All plotting lives in the `viz`
layer; the optimisation layer carries no plotting dependency.

## Consequences

- Figures share one publication-grade visual language and are accessible.
- The optimisation layer is free of plotting dependencies.
- Figures are testable and archivable because they are returned and saved.
