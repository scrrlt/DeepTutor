from src.agents.solve.utils.json_utils import extract_json_from_text


def test_extract_json_with_triple_quoted_python_code():
    raw = '''
    {
      "tool_calls": [
        {
          "type": "code_execution",
          "query": """
def foo():
    print("hello")
"""
        }
      ]
    }
    '''

    parsed = extract_json_from_text(raw)

    assert parsed is not None
    assert isinstance(parsed, dict)
    assert "tool_calls" in parsed
    assert parsed["tool_calls"][0]["type"] == "code_execution"
    assert "def foo" in parsed["tool_calls"][0]["query"]



