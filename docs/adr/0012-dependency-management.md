# 12. Dependency management

- Status: Accepted
- Date: 2026-06-20

## Context

Some capabilities need heavy dependencies (convex solvers, JIT compilers,
Jupyter, the documentation toolchain) that should not burden a minimal install
or continuous-integration jobs that do not use them.

## Decision

The project is managed with `uv` against a locked dependency set. The core
runtime is kept minimal; heavier capabilities live behind optional extras
(`perf`, `optim`, `notebooks`, `docs`), and developer tooling lives in a
dedicated group. Continuous integration installs from the frozen lock.

## Consequences

- The default install stays lean.
- Builds are reproducible from the locked versions.
- Heavy tooling is opt-in and isolated to the workflows that need it.
