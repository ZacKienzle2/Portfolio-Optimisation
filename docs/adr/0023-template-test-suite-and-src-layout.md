# 23. Template test suite and src layout

- Status: Accepted
- Date: 2026-09-22

## Context

The repository was rendered from repo-template at a commit that predated the
template's Python test suite. The other Python repositories rendered from it run
the same sessions through nox, the same measurement packages and the same
guide-derived ruff selection, and this one ran none of them. The template's
workflow and its mutation session assume the package is under `src`.

## Decision

The repository is updated to the template's current commit with `copier update`.
The package moves to `src/portfolio_optimisation`, the layout the template
renders and the Scientific Python Development Guide recommends. The development
group takes the template's test suite: pytest-randomly, pytest-timeout,
pytest-testmon, pyperf, deptry, nox and the Hypothesis ghostwriter. pytest reads
the native `[tool.pytest]` table with `strict`, and the benchmark fixture no
longer runs disabled by default, since the nox tests session skips it and the
bench session measures it. ruff selects the guide's rules with Google
docstrings, and pydoclint checks the docstrings against the signatures, which
hold the types. interrogate is removed, since ruff's pydocstyle rules now
measure docstring presence. deptry runs in the Python workflow with the package
named as first party.

## Consequences

- `nox` runs lint, tests and typing, `nox -s fast` only the tests a change
  affects, and `nox -s mutants -- <module>` measures what the tests would catch.
- Tests run in random order, so a test that depends on another fails.
- A rewrite keeps its predecessor in `baselines.py` and is checked against it by
  a generated equivalence test.
- Docstrings no longer repeat the annotated types.
