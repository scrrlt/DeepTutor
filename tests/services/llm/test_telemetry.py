# -*- coding: utf-8 -*-
"""Tests for LLM telemetry decorators."""

import logging

from _pytest.logging import LogCaptureFixture
import pytest

from src.services.llm.telemetry import track_llm_call


@pytest.mark.asyncio
async def test_track_llm_call_wraps_async_function() -> None:
    """Ensure the decorator preserves return values."""

    @track_llm_call("test-provider")
    async def sample(value: str) -> str:
        return f"ok-{value}"

    assert await sample("payload") == "ok-payload"


@pytest.mark.asyncio
async def test_track_llm_call_records_telemetry(
    caplog: LogCaptureFixture,
) -> None:
    """Ensure telemetry logs are recorded for LLM calls."""
    caplog.set_level(logging.DEBUG, logger="src.services.llm.telemetry")

    @track_llm_call("test-provider")
    async def sample(value: str) -> str:
        return f"ok-{value}"

    result = await sample("payload")

    assert result == "ok-payload"
    assert "LLM call to test-provider: sample" in caplog.text
    assert "LLM call to test-provider completed successfully" in caplog.text
