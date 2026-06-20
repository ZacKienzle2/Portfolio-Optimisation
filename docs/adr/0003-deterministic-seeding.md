# 3. Deterministic seeding

- Status: Accepted
- Date: 2026-06-20

## Context

Research results must be reproducible. Monte Carlo simulation, bootstrap
resampling and copula sampling are stochastic, and relying on the global NumPy
random state makes results depend on import order and prior calls, which is not
reproducible across runs or machines.

## Decision

Every stochastic entry point accepts an explicit `seed` and constructs its own
`numpy.random.Generator` (or forwards the seed to the underlying sampler) rather
than drawing from the global state. The seed flows from `Settings` through the
service pipeline into the copula, historical and bootstrap paths, and is
recorded in the run metadata for provenance.

Cache keys for bootstrap artefacts use `hashlib.blake2b` over the request rather
than the built-in `hash`, which is salted per interpreter run and therefore not
stable across processes.

## Consequences

- A fixed seed yields identical weights, metrics and simulated distributions,
  which is verified by determinism tests and a golden-master regression.
- Bootstrap caches are stable across sessions and invalidate when any input that
  affects the result changes.
- Callers that want non-deterministic behaviour simply omit the seed.
