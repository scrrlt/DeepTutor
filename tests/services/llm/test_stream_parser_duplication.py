# -*- coding: utf-8 -*-
"""Regression tests for StreamParser duplication edge cases."""

from src.services.llm.stream_parser import StreamParser


def test_stream_parser_no_duplication_after_think_block():
    parser = StreamParser(binding="local", model="deepseek")

    chunks = []
    chunks += parser.append("<think>thought</think>Hello")
    chunks += parser.finalize()

    assert "".join(chunks) == "Hello"
