# -*- coding: utf-8 -*-
"""Tests for StreamParser to ensure it handles partial tag chunks correctly."""

from src.services.llm.stream_parser import StreamParser
from src.services.llm.capabilities import get_thinking_markers


def test_stream_parser_handles_split_think_tags():
    parser = StreamParser(binding="local", model="qwen")

    out = parser.append("Hello <thi")
    assert out == []

    out = parser.append("nk>secret</t")
    # still no close complete
    assert out == []

    out = parser.append("hink> world")
    # 'secret' should be removed by clean_thinking_tags and "Hello " and " world" yielded
    assert out == ["Hello ", " world"]


def test_stream_parser_handles_unicode_markers_split():
    parser = StreamParser(binding="local", model="deeplab")

    # Open marker split across chunks
    out = parser.append("Lead-in ◣par")
    assert out == []

    out = parser.append("tial◢ trailing")
    # thinking block removed, leaves 'Lead-in ' and ' trailing'
    assert out == ["Lead-in ", " trailing"]


def test_stream_parser_finalize_emits_remaining_text():
    parser = StreamParser(binding="local", model="qwen")
    parser.append("partial no marker")
    remaining = parser.finalize()
    assert remaining == ["partial no marker"]


def test_stream_parser_no_duplication_after_think_block():
    parser = StreamParser(binding="local", model="deepseek")
    chunks = parser.append("<think>thought</think>Hello")
    chunks += parser.finalize()
    assert "".join(chunks) == "Hello"


def test_get_thinking_markers_override_for_qwen_and_deepseek():
    open_markers, close_markers = get_thinking_markers("deepseek", None)
    assert "<think>" in open_markers or "◣" in open_markers

    open_markers_q, close_markers_q = get_thinking_markers(None, "qwen-1")
    assert "<think>" in open_markers_q or "◣" in open_markers_q

    # OpenAI should not have thinking markers by default
    open_ai_markers, _ = get_thinking_markers("openai", None)
    assert open_ai_markers == ()
