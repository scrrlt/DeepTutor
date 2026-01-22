"""
Fuzz tests for TexChunker.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.tools.tex_chunker import TexChunker


@settings(max_examples=50, deadline=1000)
@given(st.text(min_size=0, max_size=2000))
def test_tex_chunker_split_does_not_crash(text: str) -> None:
    """Ensure tex chunker handles arbitrary input safely."""
    chunker = TexChunker(model=None)
    chunks = chunker.split_tex_into_chunks(
        text,
        max_tokens=200,
        overlap=20,
    )
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


@settings(max_examples=50, deadline=1000)
@given(st.text(min_size=0, max_size=2000))
def test_tex_chunker_estimate_tokens_does_not_crash(text: str) -> None:
    """Ensure token estimation does not crash on arbitrary input."""
    chunker = TexChunker(model=None)
    tokens = chunker.estimate_tokens(text)
    assert isinstance(tokens, int)
    assert tokens >= 0
