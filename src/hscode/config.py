"""
Runtime configuration for HSCode, loaded from environment variables.

No secrets are stored in code or in this module — everything sensitive
must come from the environment (or a ``.env`` file loaded by the caller).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int) -> int:
    val = _env(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _default_cache_dir() -> Path:
    explicit = _env("HSCODE_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = _env("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "hscode"
    return Path.home() / ".cache" / "hscode"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    provider: Optional[str]
    model: Optional[str]
    cache_dir: Path
    cn_year: Optional[int]
    max_retries: int
    log_level: str
    sparql_endpoint: str
    sparql_timeout: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=_env("HSCODE_PROVIDER"),
            model=_env("HSCODE_MODEL"),
            cache_dir=_default_cache_dir(),
            cn_year=_env_int("HSCODE_CN_YEAR", 0) or None,
            max_retries=_env_int("HSCODE_MAX_RETRIES", 3),
            log_level=_env("HSCODE_LOG_LEVEL", "INFO") or "INFO",
            sparql_endpoint=_env(
                "HSCODE_SPARQL_ENDPOINT",
                "http://publications.europa.eu/webapi/rdf/sparql",
            )
            or "http://publications.europa.eu/webapi/rdf/sparql",
            sparql_timeout=_env_int("HSCODE_SPARQL_TIMEOUT", 300),
        )


# Singleton — read once at import time.
SETTINGS = Settings.from_env()
