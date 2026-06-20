# Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Python 3.12 or
3.13 is required.

## Install from source

```bash
uv sync --frozen --all-extras
```

This installs the core runtime together with the optional extras:

| Extra       | Purpose                                                        |
| ----------- | ------------------------------------------------------------- |
| `perf`      | Numba and numexpr accelerators for hot loops.                 |
| `optim`     | cvxpy and solvers for the CDaR, SSD and goal-programming LPs. |
| `notebooks` | JupyterLab and nbconvert for the worked example.              |
| `docs`      | MkDocs Material and mkdocstrings for this site.               |

To install only the core runtime:

```bash
uv sync --frozen
```

## Verify

```bash
uv run pytest -q
uv run portfolio-opt version
```

## Build the documentation

```bash
uv run --extra docs mkdocs serve   # live preview at http://127.0.0.1:8000
uv run --extra docs mkdocs build   # static site under site/
```
