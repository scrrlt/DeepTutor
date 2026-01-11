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
        # Find the first '{' and then scan forward to find the matching closing brace,
        # correctly handling nested braces.
        start = message.find("{")
        if start != -1:
            brace_depth = 0
            end = None
            for idx in range(start, len(message)):
                char = message[idx]
                if char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        end = idx
                        break
            if end is not None:
                try:
                    error_data = json.loads(message[start : end + 1])
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_info = error_data["error"]
                        if isinstance(error_info, dict):
                            error_msg = error_info.get("message", "")
                            error_type = error_info.get("type", "")
                            error_code = error_info.get("code", "")
                            formatted_parts = []
                            if error_msg:
                                formatted_parts.append(f"Message: {error_msg}")
                            if error_type:
                                formatted_parts.append(f"Type: {error_type}")
                            if error_code:
                                formatted_parts.append(f"Code: {error_code}")
                            if formatted_parts:
                                return " | ".join(formatted_parts)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass

    # Return original message if parsing fails
    return message
