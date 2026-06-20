# 13. CI gating and commit conventions

- Status: Accepted
- Date: 2026-06-20

## Context

Quality must be enforced by the workflow rather than depending on the author to
remember to run checks, and the history should be legible and machine-readable.

## Decision

The default branch is protected: changes go through pull requests, a summary
status gate and commit-message validation must pass, history is linear, and
merges are squashed. Commit messages follow Conventional Commits. Workflow
actions are pinned by commit SHA and run under a hardened runner.

## Consequences

- Only validated changes reach the default branch.
- History is conventional and linear, which supports automated release notes.
- The supply chain is hardened through pinned actions.
