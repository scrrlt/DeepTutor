#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error Utilities - Error formatting and handling utilities
"""

import json


def format_exception_message(exc: Exception) -> str:
    """
    Format exception message for better readability

    Args:
        exc: The exception to format

    Returns:
        Formatted error message
    """
    message = str(exc)

    # Try to parse JSON error messages (common in API errors)
    try:
        # Find JSON-like block by counting braces (handles nested objects)
        start_idx = message.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = -1
            for char_idx in range(start_idx, len(message)):
                if message[char_idx] == "{":
                    brace_count += 1
                elif message[char_idx] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = char_idx
                        break

            if end_idx != -1:  # Found matching braces
                potential_json = message[start_idx : end_idx + 1]
                try:
                    error_data = json.loads(potential_json)

                    # Standard extraction logic
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_info = error_data["error"]
                        if isinstance(error_info, dict):
                            parts = []
                            if "message" in error_info:
                                parts.append(f"Message: {error_info['message']}")
                            if "type" in error_info:
                                parts.append(f"Type: {error_info['type']}")
                            if "code" in error_info:
                                parts.append(f"Code: {error_info['code']}")
                            if parts:
                                return " | ".join(parts)
                except (json.JSONDecodeError, AttributeError):
                    pass
    except Exception:
        pass

    # Return original message if parsing fails
    return message
