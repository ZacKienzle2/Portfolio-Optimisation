# 7. Code-derived architecture diagrams

- Status: Accepted
- Date: 2026-06-20

## Context

Hand-drawn architecture diagrams drift from the code they describe and quietly
become misleading.

## Decision

A generator parses the package with the standard-library `ast` (no imports
executed) and emits the module and layer dependency graphs as Mermaid and
Graphviz. It also detects illegal cross-layer edges. A `--check` mode and a CI
job fail when the committed diagrams drift from the source, and a test asserts
the layering has no violations.

## Consequences

- The diagrams are always faithful to the code.
- Layering regressions are caught automatically rather than by review.
- Diagrams render natively in the documentation site and on the repository host.
