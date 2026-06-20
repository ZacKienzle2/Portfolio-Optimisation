# 14. References live in the writeup

- Status: Accepted
- Date: 2026-06-20

## Context

Source docstrings carried bibliographic citations. Maintaining a bibliography
inside the code duplicates effort and clutters the API documentation, while the
accompanying writeup is the natural home for the literature.

## Decision

Source docstrings describe behaviour, arguments, returns and mathematics, but do
not carry citations or publication years. Eponymous method names (for example
Marchenko-Pastur, Black-Litterman, Ledoit-Wolf) are kept as terminology. The
literature is cited in the accompanying writeup, not in the code.

## Consequences

- Docstrings stay focused on usage and behaviour.
- The bibliography has a single home and is not duplicated.
- Method names remain discoverable without the bibliographic detail.
