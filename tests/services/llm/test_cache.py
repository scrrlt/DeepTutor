import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import os
from src.services.llm.cache import (
    build_completion_cache_key,
    get_cache_client,
    _sanitize_kwargs_for_hashing
)

@pytest.mark.asyncio
async def test_get_cache_client_singleton():
    """Verify singleton behavior and lock usage."""
    with patch("src.services.llm.cache.Redis.from_url") as mock_redis:
        mock_redis.return_value.ping = AsyncMock(return_value=True)
        
        # Reset globals for test
        with patch.dict(
            os.environ,
            {"LLM_CACHE_REDIS_URL": "redis://localhost:6379/0"},
        ), patch(
            "src.services.llm.cache._CACHE_CLIENT",
            None,
        ), patch(
            "src.services.llm.cache._CACHE_LOCK",
            None,
        ):
            
            # First call creates client
            client1 = await get_cache_client()
            assert client1 is not None
            assert mock_redis.call_count == 1
            
            # Second call reuses client
            client2 = await get_cache_client()
            assert client2 is client1
            assert mock_redis.call_count == 1

def test_cache_key_sanitization():
    """Ensure sensitive keys are stripped before hashing."""
    kwargs = {
        "temperature": 0.7,
        "api_key": "sk-secret-do-not-hash",
        "Authorization": "Bearer xyz",
        "custom_param": "value"
    }
    
    sanitized = _sanitize_kwargs_for_hashing(kwargs)
    
    assert "temperature" in sanitized
    assert "custom_param" in sanitized
    assert "api_key" not in sanitized
    assert "Authorization" not in sanitized

def test_cache_key_determinism():
    """Ensure identical inputs produce identical hash keys regardless of dict order."""
    params1 = {"b": 2, "a": 1}
    params2 = {"a": 1, "b": 2}
    
    key1 = build_completion_cache_key(
        model="gpt-4", binding="openai", base_url=None, 
        system_prompt="sys", prompt="hi", messages=[], **params1
    )
    
    key2 = build_completion_cache_key(
        model="gpt-4", binding="openai", base_url=None, 
        system_prompt="sys", prompt="hi", messages=[], **params2
    )
    
    assert key1 == key2