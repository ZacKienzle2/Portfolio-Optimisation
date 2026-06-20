# 18. Documentation publishing

- Status: Accepted
- Date: 2026-06-21

## Context

The documentation site was built in continuous integration but only uploaded as
a short-lived artefact, so there was no published reference for reviewers, and
the architecture diagrams rendered only from files a contributor had to
regenerate by hand. The site is generated from the docstrings and the
architecture diagrams, both of which are derived from the source. A published,
always-current site is more useful than a manual build, and the rendered
diagrams should never lag behind the code they describe.

## Decision

On every push to the default branch the documentation workflow regenerates the
code-derived diagrams, builds the site under `mkdocs build --strict`, and
deploys it to GitHub Pages. Because the diagrams are regenerated as a build step,
the published site always reflects the current source regardless of the
committed diagram files. Pull requests still run the strict build and the
diagram drift check, so a reviewer sees correct diagrams in the diff, and the
build uses only the documentation extra rather than the full dependency set.

## Consequences

- The site, including the API reference and architecture diagrams, refreshes
  automatically after every squash merge.
- The published diagrams cannot drift from the source, since the build
  regenerates them.
- Publishing depends on GitHub Pages being enabled with the workflow build type.
- The documentation build stays lean by installing only the documentation extra.
