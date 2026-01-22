#!/usr/bin/env python
"""
JSON Utils - JSON parsing and validation utilities
- Robustly extract JSON from LLM text output
- Provide strict structure validation and error messages
"""

from collections.abc import Iterable
import json
import re
from typing import Any


def _parse_if_json(value: str) -> dict[str, Any] | list[Any] | None:
    """Return parsed JSON only when it is an object or array."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any] | None:
    """
    Extract JSON object or array from text.
    Allows the following formats:
    1) Pure JSON text
    2) Code blocks wrapped in ```json ...``` or ``` ...```
    3) First JSON fragment {...} or [...] contained in text
    """
    if not text:
        return None

    # 1) Code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        snippet = code_block.group(1).strip()
        parsed = _parse_if_json(snippet)
        if parsed is not None:
            return parsed

    # 2) Parse entire text
    parsed_full = _parse_if_json(text)
    if parsed_full is not None:
        return parsed_full

    # 3) Fragment parsing
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        parsed_obj = _parse_if_json(obj_match.group(0))
        if parsed_obj is not None:
            return parsed_obj

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        parsed_arr = _parse_if_json(arr_match.group(0))
        if parsed_arr is not None:
            return parsed_arr

    return None


# --------- Strict Validation Utilities ---------


def ensure_json_dict(
    data: Any, err: str = "Expected JSON object"
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(err)
    return data


def ensure_json_list(data: Any, err: str = "Expected JSON array") -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(err)
    return data


def ensure_keys(data: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys: {', '.join(missing)}")
    return data


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def json_to_text(data: Any, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)


__all__ = [
    "extract_json_from_text",
    "ensure_json_dict",
    "ensure_json_list",
    "ensure_keys",
    "safe_json_loads",
    "json_to_text",
]
