# 20. Automated dependency updates and coverage floor

- Status: Accepted
- Date: 2026-06-21

## Context

Dependabot watched only the GitHub Actions workflows, so the Python
dependencies received security alerts but no automated update pull requests, and
the test suite measured coverage without enforcing a minimum, allowing it to
erode unnoticed.

## Decision

Dependabot also tracks the Python dependencies through the uv ecosystem, with
minor and patch updates grouped into a single weekly pull request and a capped
number of open requests to keep continuous-integration usage modest. The
coverage report enforces a minimum through `fail_under`, set below the current
level to lock it in and intended to rise as the suite grows. Both run inside the
existing jobs, so neither adds a new billed workflow.

## Consequences

- Python dependency updates arrive as reviewable pull requests on a fixed
  cadence rather than by hand.
- Test coverage cannot silently regress below the floor.
- Grouping and the pull-request cap keep the added continuous-integration load
  small.
