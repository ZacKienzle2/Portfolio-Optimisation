# 21. Import latency and serial execution

- Status: Accepted
- Date: 2026-09-22

## Context

Profiling the package with py-spy over a workload that calls every public
routine attributed 45 per cent of the wall time to importing
`portfolio_optimisation.optim`. Each subpackage `__init__` imported every
module it exported, so touching `infra.logging` pulled in PyPortfolioOpt and
through it cvxpy, and touching `optim` pulled in arch. `portfolio-opt version`
took 2.9 s to print a string, measured with hyperfine.

The same imports dominated the two process pools. `HRPAnalyser.run_bootstrap`
and `SDEFitter` spawned workers, and on Windows every worker imports the
numerical stack before its first task. One bootstrap resample takes 1.3 ms on
six assets, so 1000 of them took 8.8 s in the pool and 1.3 s in one process.
The pool also pickled the analyser, returns included, for every chunk of tasks.

## Decision

Every subpackage exports through a stub, `__init__.pyi`, and loads lazily with
`lazy_loader.attach_stub`, the Scientific Python SPEC 1 mechanism. Type checkers
read the stub, which lists each export once with its type. The
root package resolves `__version__` from the installed metadata on first access
rather than at import. The command-line interface binds the services module
and reaches the pipeline through it at call time. Optional heavy imports,
cvxpy and the plotting libraries, are deferred into the function that needs
them, which the ruff configuration records by ignoring PLC0415.

The bootstrap resamples and the per-asset SDE fits run in the calling process.
A pool is reintroduced only when a measurement shows the serial work exceeding
the several seconds a spawned worker spends importing.

## Consequences

- `portfolio-opt version` and `config` start in 0.2 s, the cost of the
  interpreter itself.
- A notebook importing one allocator no longer pays for cvxpy, arch or
  matplotlib.
- The bootstrap and the SDE fits are 5 to 14 times faster at the sizes the
  project runs, and produce identical statistics.
- A new export is added to the stub, and a new heavy dependency is imported
  inside the function that uses it.
