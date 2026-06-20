# 1. Domain-driven layering

- Status: Accepted
- Date: 2026-06-20

## Context

Quantitative research code tends to fuse numerical logic, data acquisition and
plotting into monolithic modules. That coupling makes the numerical core hard to
test in isolation, ties analytics to a specific data vendor, and lets
presentation concerns leak into computation.

## Decision

The package is split into layers with a one-directional dependency rule:

- `domain` holds pure types and protocols with no IO and no third-party
  framework coupling.
- `infra` implements the domain protocols (market-data acquisition, parquet
  caching, reporting) and may depend only on the domain and configuration.
- `optim`, `risk`, `sde` and `econometrics` are analytics layers that depend on
  the domain, configuration and infrastructure, never on visualisation or the
  service layer.
- `viz` consumes analytics outputs and may depend on the analytics layers.
- `services` orchestrates the workflow and is the only layer permitted to wire
  the others together.

The allowed dependencies are encoded in `tools/gen_diagrams.py` and enforced by
`tests/test_architecture.py`, so a violating import fails the test suite.

## Consequences

- The numerical core is testable without network or filesystem access.
- Swapping the data source or the plotting backend is a localised change.
- A small amount of indirection (protocols, a service layer) is added, which is
  justified by the testability and the enforced boundaries.
