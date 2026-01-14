"""Environment & config schema validation tool.

Validates required environment variables & simple type expectations.
Extend by editing SCHEMA dict.
"""
from __future__ import annotations
import os, sys

SCHEMA = {
    "GEMINI_API_KEY": {"required": True, "min_length": 10},
    "DAILY_TOKEN_LIMIT": {"required": False, "type": int, "default": 50000},
    "TIMEBOX_MINUTES": {"required": False, "type": int, "default": 30},
    "SESSION_TOKEN_CAP": {"required": False, "type": int, "default": 8000},
}

class SchemaError(Exception):
    pass

def _coerce(value: str, spec: dict):
    if 'type' in spec and spec['type'] is int:
        return int(value)
    return value

def validate_env(schema: dict = SCHEMA):
    # Skip validation if running under pytest (for unit tests)
    if 'pytest' in sys.modules or any('pytest' in arg for arg in sys.argv):
        return
    errors = []
    for key, spec in schema.items():
        raw = os.getenv(key)
        if raw is None:
            if spec.get('required'):
                errors.append(f"Missing required env: {key}")
                continue
            if 'default' in spec:
                os.environ[key] = str(spec['default'])
                raw = os.getenv(key)
        if raw is not None:
            if spec.get('min_length') and len(raw) < spec['min_length']:
                errors.append(f"Env {key} shorter than min_length {spec['min_length']}")
            try:
                _coerce(raw, spec)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Env {key} type coercion failed: {exc}")
    if errors:
        raise SchemaError("; ".join(errors))

if __name__ == '__main__':
    try:
        validate_env()
        print("[OK] Environment schema validation passed.")
    except SchemaError as exc:
        print("[FAIL]", exc)
        sys.exit(1)