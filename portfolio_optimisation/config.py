"""Central typed configuration for portfolio optimisation runs.

Settings resolve with the precedence explicit-argument > environment variable
(``PORTFOLIO_*``) > ``portfolio.toml`` > built-in default, so the same code path
runs reproducibly from notebooks, the CLI and CI without scattered magic
numbers or hardcoded paths.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

VarMethod = Literal["empirical", "parametric"]
LinkageMethod = Literal["ward", "single", "complete", "average"]

ENV_PREFIX = "PORTFOLIO_"
CONFIG_FILENAME = "portfolio.toml"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable run configuration.

    Attributes:
        risk_free_rate (float): Annual risk-free rate for Sharpe/Sortino.
        n_simulations (int): Monte Carlo simulation paths.
        var_alpha (float): Tail level for VaR/CVaR.
        var_method (VarMethod): Empirical or parametric VaR.
        linkage_method (LinkageMethod): Hierarchical-clustering linkage.
        seed (int | None): Master seed for every stochastic path. None means
            non-deterministic.
        std_floor (float): Lower clamp on standard deviations to avoid division
            by near-zero variance.
        cache_dir (Path): Directory for bootstrap and simulation artefacts.
        data_cache_path (Path): Parquet snapshot for market data.
        log_level (str): Logging level name (e.g. "INFO", "DEBUG").
    """

    risk_free_rate: float = 0.02
    n_simulations: int = 10_000
    var_alpha: float = 0.05
    var_method: VarMethod = "empirical"
    linkage_method: LinkageMethod = "ward"
    seed: int | None = None
    std_floor: float = 1e-12
    cache_dir: Path = Path("cache")
    data_cache_path: Path = Path("data") / "market_data.parquet"
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain types (paths as strings) for logging/provenance."""
        out = asdict(self)
        out["cache_dir"] = str(self.cache_dir)
        out["data_cache_path"] = str(self.data_cache_path)
        return out


def _seed_coercer(raw: str) -> int | None:
    return None if raw.strip().lower() in {"", "none"} else int(raw)


_COERCERS: dict[str, Callable[[str], Any]] = {
    "risk_free_rate": float,
    "n_simulations": int,
    "var_alpha": float,
    "var_method": str,
    "linkage_method": str,
    "seed": _seed_coercer,
    "std_floor": float,
    "cache_dir": Path,
    "data_cache_path": Path,
    "log_level": str,
}


def _read_toml_table(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    tool_table = document.get("tool", {})
    if isinstance(tool_table, dict) and "portfolio" in tool_table:
        return dict(tool_table["portfolio"])
    if "portfolio" in document:
        return dict(document["portfolio"])
    return document


def load_settings(config_path: str | Path | None = None, /, **overrides: Any) -> Settings:
    """Build :class:`Settings` from file, environment and explicit overrides.

    Args:
        config_path (str | Path | None): Explicit TOML path. Defaults to
            ``portfolio.toml`` in the working directory when present.
        **overrides: Explicit field values; any whose value is not None take
            top precedence.

    Returns:
        Settings: The resolved, type-coerced configuration.
    """
    raw: dict[str, str] = {}

    path = Path(config_path) if config_path is not None else Path(CONFIG_FILENAME)
    if path.exists():
        for key, value in _read_toml_table(path).items():
            if key in _COERCERS:
                raw[key] = str(value)

    for key in _COERCERS:
        env_value = os.environ.get(ENV_PREFIX + key.upper())
        if env_value is not None:
            raw[key] = env_value

    resolved: dict[str, Any] = {key: _COERCERS[key](value) for key, value in raw.items()}
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return Settings(**resolved)
