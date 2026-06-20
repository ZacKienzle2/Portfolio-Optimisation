"""Generate architecture diagrams from the package source via static analysis.

Parses every module under ``portfolio_optimisation`` with the standard-library
``ast`` (no imports executed) to build the intra-package import graph, then
emits:

* a module-level dependency graph,
* a layer-level (DDD) dependency graph with illegal edges flagged,

in both Mermaid (renders natively on GitHub and in MkDocs) and Graphviz DOT
(rendered to SVG when the ``dot`` binary is available). Run with ``--check`` to
fail when the committed diagrams have drifted from the source.

Usage:
    python tools/gen_diagrams.py            # regenerate docs/diagrams/*
    python tools/gen_diagrams.py --check    # verify diagrams are up to date
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE = "portfolio_optimisation"
ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / PACKAGE
OUTPUT_DIR = ROOT / "docs" / "diagrams"

# Allowed dependency targets per DDD layer. An edge into a layer absent from a
# source layer's set is an architectural violation.
ALLOWED: dict[str, set[str]] = {
    # The package facade (__init__) may surface config; importing the facade
    # (e.g. for __version__) is always permitted.
    "root": {"config"},
    "domain": set(),
    "config": set(),
    "infra": {"domain", "config"},
    "optim": {"domain", "config", "infra"},
    "risk": {"domain", "config", "infra"},
    "sde": {"domain", "config", "infra"},
    "econometrics": {"domain", "config", "infra"},
    "viz": {"domain", "config", "infra", "optim", "risk", "sde", "econometrics"},
    "services": {
        "domain", "config", "infra", "optim", "risk", "sde", "econometrics", "viz",
    },
    "cli": {
        "domain", "config", "infra", "optim", "risk", "sde",
        "econometrics", "viz", "services", "root",
    },
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _layer_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "root"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    found.add(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(PACKAGE)
        ):
            found.add(node.module)
    return found


def build_graphs() -> tuple[dict[str, set[str]], dict[str, set[str]], list[tuple[str, str]]]:
    """Return (module_edges, layer_edges, violations) derived from the source."""
    module_edges: dict[str, set[str]] = defaultdict(set)
    layer_edges: dict[str, set[str]] = defaultdict(set)
    violations: list[tuple[str, str]] = []

    modules = sorted(_module_name(p) for p in PACKAGE_DIR.rglob("*.py"))
    module_set = set(modules)

    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        source = _module_name(path)
        source_layer = _layer_of(source)
        for target in sorted(_imports(path)):
            if target == source or target not in module_set:
                continue
            module_edges[source].add(target)
            target_layer = _layer_of(target)
            if source_layer != target_layer:
                layer_edges[source_layer].add(target_layer)
                if target_layer not in ALLOWED.get(source_layer, set()):
                    violations.append((source_layer, target_layer))

    return module_edges, layer_edges, sorted(set(violations))


def _mermaid_layers(layer_edges: dict[str, set[str]], violations: set[tuple[str, str]]) -> str:
    lines = ["flowchart TD"]
    edges = sorted({(s, t) for s, targets in layer_edges.items() for t in targets})
    illegal_indices: list[int] = []
    for index, (source, target) in enumerate(edges):
        arrow = "-. illegal .->" if (source, target) in violations else "-->"
        lines.append(f"    {source} {arrow} {target}")
        if (source, target) in violations:
            illegal_indices.append(index)
    for index in illegal_indices:
        lines.append(f"    linkStyle {index} stroke:#D55E00,stroke-width:2px;")
    return "\n".join(lines) + "\n"


def _mermaid_modules(module_edges: dict[str, set[str]]) -> str:
    by_layer: dict[str, set[str]] = defaultdict(set)
    for source, targets in module_edges.items():
        by_layer[_layer_of(source)].add(source)
        for target in targets:
            by_layer[_layer_of(target)].add(target)

    lines = ["flowchart LR"]
    for layer in sorted(by_layer):
        lines.append(f"    subgraph {layer}")
        for module in sorted(by_layer[layer]):
            lines.append(f'        {_node_id(module)}["{module}"]')
        lines.append("    end")
    for source in sorted(module_edges):
        for target in sorted(module_edges[source]):
            lines.append(f"    {_node_id(source)} --> {_node_id(target)}")
    return "\n".join(lines) + "\n"


def _node_id(module: str) -> str:
    return module.replace(".", "_")


def _dot_layers(layer_edges: dict[str, set[str]], violations: set[tuple[str, str]]) -> str:
    lines = ["digraph layers {", "    rankdir=TB;", '    node [shape=box, style=rounded];']
    for source, targets in sorted(layer_edges.items()):
        for target in sorted(targets):
            colour = ' [color="#D55E00", penwidth=2]' if (source, target) in violations else ""
            lines.append(f"    {source} -> {target}{colour};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_modules(module_edges: dict[str, set[str]]) -> str:
    lines = ["digraph modules {", "    rankdir=LR;", '    node [shape=box, fontsize=10];']
    for source in sorted(module_edges):
        for target in sorted(module_edges[source]):
            lines.append(f'    "{source}" -> "{target}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_svg(dot_path: Path) -> None:
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return
    svg_path = dot_path.with_suffix(".svg")
    subprocess.run(
        [dot_binary, "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )


def _artifacts() -> dict[str, str]:
    module_edges, layer_edges, violations = build_graphs()
    violation_set = set(violations)
    return {
        "module_dependencies.mmd": _mermaid_modules(module_edges),
        "layer_dependencies.mmd": _mermaid_layers(layer_edges, violation_set),
        "module_dependencies.dot": _dot_modules(module_edges),
        "layer_dependencies.dot": _dot_layers(layer_edges, violation_set),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed diagrams differ from the source.",
    )
    args = parser.parse_args(argv)

    artifacts = _artifacts()
    _, _, violations = build_graphs()

    if args.check:
        stale = [
            name
            for name, content in artifacts.items()
            if not (OUTPUT_DIR / name).exists()
            or (OUTPUT_DIR / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            print("Diagrams are stale; run python tools/gen_diagrams.py:", file=sys.stderr)
            for name in stale:
                print(f"  - {name}", file=sys.stderr)
            return 1
        print("Diagrams are up to date.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
        if name.endswith(".dot"):
            _render_svg(OUTPUT_DIR / name)

    if violations:
        print("WARNING: architecture layer violations detected:", file=sys.stderr)
        for source, target in violations:
            print(f"  {source} -> {target}", file=sys.stderr)
    print(f"Wrote {len(artifacts)} diagram(s) to {OUTPUT_DIR.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
