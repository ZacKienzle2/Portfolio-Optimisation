"""Architecture guard: the DDD layering must hold.

Reuses the diagram generator's static import analysis to assert there are no
illegal cross-layer dependencies (for example domain importing infra, or an
analytics layer importing the visualisation layer).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_gen_diagrams() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "tools" / "gen_diagrams.py"
    spec = importlib.util.spec_from_file_location("gen_diagrams", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_layer_violations() -> None:
    gen = _load_gen_diagrams()
    _, _, violations = gen.build_graphs()
    assert violations == [], f"architecture layer violations: {violations}"
