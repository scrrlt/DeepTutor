# -*- coding: utf-8 -*-
"""Tests for LLM factory routing behavior."""

import pytest

from src.services.llm import cloud_provider, factory, local_provider
from src.services.llm.exceptions import LLMTimeoutError


@pytest.mark.asyncio
async def test_complete_routes_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_local(**_kwargs: object) -> str:
        """
        Simulate a local provider completion by returning the literal "local".
        
        Returns:
            str: The string "local".
        """
        return "local"

    async def fake_cloud(**_kwargs: object) -> str:
        """
        Fail the test if the cloud provider is invoked.
        
        Raises:
            AssertionError: Always raised with message "Cloud provider should not be called".
        """
        raise AssertionError("Cloud provider should not be called")

    monkeypatch.setattr(local_provider, "complete", fake_local)
    monkeypatch.setattr(cloud_provider, "complete", fake_cloud)

    result = await factory.complete(
        prompt="hi",
        model="local-model",
        base_url="http://localhost:1234/v1",
    )

    assert result == "local"


@pytest.mark.asyncio
async def test_complete_routes_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_local(**_kwargs: object) -> str:
        """
        Fake local provider used in tests that raises an AssertionError when invoked to ensure the local provider is not called.
        
        Raises:
            AssertionError: Always raised to indicate the local provider was unexpectedly called.
        """
        raise AssertionError("Local provider should not be called")

    async def fake_cloud(**_kwargs: object) -> str:
        """
        Test helper that simulates a cloud provider completion and returns the string "cloud".
        
        Parameters:
            **_kwargs (object): Arbitrary keyword arguments accepted for compatibility; they are ignored.
        
        Returns:
            The string "cloud".
        """
        return "cloud"

    monkeypatch.setattr(local_provider, "complete", fake_local)
    monkeypatch.setattr(cloud_provider, "complete", fake_cloud)

    result = await factory.complete(
        prompt="hi",
        model="cloud-model",
        base_url="https://api.openai.com/v1",
    )

    assert result == "cloud"


@pytest.mark.asyncio
async def test_stream_routes_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_local_stream(**_kwargs: object):
        """
        Fake local provider stream that yields two deterministic chunks for testing.
        
        Returns:
            A generator that yields the strings "local-1" and "local-2".
        """
        yield "local-1"
        yield "local-2"

    async def fake_cloud_stream(**_kwargs: object):
        """
        Test stub async generator that fails the test if invoked by raising an AssertionError.
        
        Raises:
            AssertionError: Indicates the cloud provider was called unexpectedly.
        """
        if False:
            yield "unused"
        raise AssertionError("Cloud provider should not be called")

    monkeypatch.setattr(local_provider, "stream", fake_local_stream)
    monkeypatch.setattr(cloud_provider, "stream", fake_cloud_stream)

    chunks = [
        chunk
        async for chunk in factory.stream(
            prompt="hi",
            model="local-model",
            base_url="http://localhost:1234/v1",
        )
    ]

    assert chunks == ["local-1", "local-2"]


@pytest.mark.asyncio
async def test_stream_routes_to_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that factory.stream routes streaming requests to the cloud provider.
    
    Replaces local and cloud provider stream implementations and asserts that iterating
    factory.stream with a cloud model yields the chunks produced by the cloud provider.
    """
    async def fake_local_stream(**_kwargs: object):
        if False:
            yield "unused"
        raise AssertionError("Local provider should not be called")

    async def fake_cloud_stream(**_kwargs: object):
        """
        Asynchronous test stream that yields two cloud response chunks.
        
        Parameters:
            **_kwargs: Additional keyword arguments are accepted and ignored.
        
        Yields:
            str: Two sequential chunks: "cloud-1" then "cloud-2".
        """
        yield "cloud-1"
        yield "cloud-2"

    monkeypatch.setattr(local_provider, "stream", fake_local_stream)
    monkeypatch.setattr(cloud_provider, "stream", fake_cloud_stream)

    chunks = [
        chunk
        async for chunk in factory.stream(
            prompt="hi",
            model="cloud-model",
            base_url="https://api.openai.com/v1",
        )
    ]

    assert chunks == ["cloud-1", "cloud-2"]


@pytest.mark.asyncio
async def test_stream_no_retry_after_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_cloud_stream(**_kwargs: object):
        """
        Simulated cloud streaming generator that yields a single partial chunk then simulates a timeout.
        
        Yields:
            str: The partial chunk string "partial".
        
        Raises:
            LLMTimeoutError: Raised after yielding to simulate a dropped/timeout stream.
        """
        nonlocal call_count
        call_count += 1
        yield "partial"
        raise LLMTimeoutError("stream dropped")

    monkeypatch.setattr(cloud_provider, "stream", fake_cloud_stream)

    chunks: list[str] = []
    with pytest.raises(LLMTimeoutError):
        async for chunk in factory.stream(
            prompt="hi",
            model="cloud-model",
            base_url="https://api.openai.com/v1",
            max_retries=1,
            retry_delay=0.0,
        ):
            chunks.append(chunk)

    assert chunks == ["partial"]
    assert call_count == 1


@pytest.mark.asyncio
async def test_complete_passes_binding_and_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_cloud_complete(**kwargs: object) -> str:
        """
        Test helper that validates the cloud `binding` and `api_version` values and returns a confirmation string.
        
        Parameters:
            kwargs (dict): Keyword arguments expected to include `binding` (must be "azure") and `api_version` (must be "2024-02-01").
        
        Returns:
            result (str): The string "ok" when validation succeeds.
        """
        assert kwargs["binding"] == "azure"
        assert kwargs["api_version"] == "2024-02-01"
        return "ok"

    async def fake_local_complete(**_kwargs: object) -> str:
        """
        Test helper that fails if invoked to assert the local provider was not called.
        
        Parameters:
            _kwargs (dict): Any keyword arguments are accepted but ignored; present to match provider signature.
        
        Raises:
            AssertionError: Always raised to indicate unexpected invocation of the local provider.
        """
        raise AssertionError("Local provider should not be called")

    monkeypatch.setattr(cloud_provider, "complete", fake_cloud_complete)
    monkeypatch.setattr(local_provider, "complete", fake_local_complete)

    result = await factory.complete(
        prompt="hi",
        model="cloud-model",
        api_key="test-key",
        base_url="https://example.azure.com/openai/deployments/test",
        api_version="2024-02-01",
        binding="azure",
    )

    assert result == "ok"