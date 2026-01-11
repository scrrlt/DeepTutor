"""
Robust JSON Parser - Extract and parse JSON from messy LLM output.
"""

import json
import re


def extract_and_parse_json(text: str):
    """
    Finds the first '{' and last '}' to extract valid JSON from messy LLM output.
    """
    try:
        # 1. Try standard parse first (fastest)
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract substring between first { and last }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Handle Markdown code blocks specifically
    code_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if code_block:
        return json.loads(code_block.group(1))

    raise ValueError("Could not extract valid JSON from response")


def parse_llm_json(text: str):
    """Alias for extract_and_parse_json."""
    return extract_and_parse_json(text)
