"""
Fuzz tests for JSON utilities.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.research.utils import json_utils as research_json_utils
from src.agents.solve.utils import json_utils as solve_json_utils


@settings(max_examples=100, deadline=1000)
@given(st.text(min_size=0, max_size=2000))
def test_research_extract_json_is_safe(text: str) -> None:
    """Ensure research JSON extractor never crashes on arbitrary input."""
    result = research_json_utils.extract_json_from_text(text)
    assert result is None or isinstance(result, (dict, list))


@settings(max_examples=100, deadline=1000)
@given(st.text(min_size=0, max_size=2000))
def test_solve_extract_json_is_safe(text: str) -> None:
    """Ensure solve JSON extractor never crashes on arbitrary input."""
    result = solve_json_utils.extract_json_from_text(text)
    assert result is None or isinstance(result, (dict, list))
