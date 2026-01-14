"""Feature flag provider.

Resolution order:
1. In-memory overrides (future)
2. Environment variables (FLAG_<NAME>=1|0)
3. JSON file pointed to by FEATURE_FLAGS_FILE (keys bool)

Usage:
    from config.feature_flags import flag
    if flag('PARALLEL_ENRICHMENT'): ...
"""
from __future__ import annotations
import os, json
from functools import lru_cache
from typing import Any

_ENV_PREFIX = "FLAG_"

@lru_cache(maxsize=1)
def _file_flags() -> dict[str, Any]:
    path = os.getenv("FEATURE_FLAGS_FILE")
    if not path or not os.path.exists(path):  # type: ignore[attr-defined]
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:  # noqa: BLE001
        return {}
    return {}

def flag(name: str, default: bool = False) -> bool:
    env_key = _ENV_PREFIX + name.upper()
    if env_key in os.environ:
        val = os.getenv(env_key, "0")
        return val not in ("0", "false", "False", "")
    file_val = _file_flags().get(name)
    if isinstance(file_val, bool):
        return file_val
    return default

__all__ = ["flag"]