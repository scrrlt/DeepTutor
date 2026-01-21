# -*- coding: utf-8 -*-
"""Tests for StreamParser to ensure it handles partial tag chunks correctly."""

from src.services.llm.stream_parser import StreamParser


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
