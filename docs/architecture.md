# Architecture

The package is organised into domain-driven layers with strict dependency
inversion. The domain layer is pure (no IO, no framework coupling);
infrastructure adapters implement domain protocols; analytics layers depend only
on the domain, configuration and infrastructure; visualisation consumes
analytics outputs; and the service layer orchestrates the whole flow.

## Layer dependencies

The graph below is generated from the source by `tools/gen_diagrams.py` and kept
in sync by continuous integration. Any edge that would violate the layering (for
example the domain depending on infrastructure, or an analytics layer depending
on visualisation) is detected by the generator and by a dedicated test.

```mermaid
--8<-- "docs/diagrams/layer_dependencies.mmd"
```

## Module dependencies

```mermaid
--8<-- "docs/diagrams/module_dependencies.mmd"
```

## Regenerating the diagrams

```bash
python tools/gen_diagrams.py          # regenerate docs/diagrams/*
python tools/gen_diagrams.py --check  # verify the diagrams match the source
```

When the Graphviz `dot` binary is available the generator also renders SVG
copies alongside the Mermaid and DOT sources.

## Rationale

The reasoning behind the layering and the key infrastructure patterns is
recorded in the decision records:

- [DDD layering](adr/0001-ddd-layering.md)
- [Repository and Unit of Work](adr/0002-repository-unit-of-work.md)
- [Deterministic seeding](adr/0003-deterministic-seeding.md)
