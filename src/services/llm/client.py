import asyncio
import os
import random
import time
from typing import List

from src.utils.network.circuit_breaker import CircuitBreaker


class KeyRotator:
    """Rotate API keys on failures."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("keys must be a non-empty list")
        self.keys = keys
        self.current_index = 0

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def rotate(self):
        self.current_index = (self.current_index + 1) % len(self.keys)


class ResilientLLMClient:
    """Unbreakable LLM client with circuit breaker, retry, and key rotation."""

    def __init__(self):
        self.breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        # Load keys from environment variable
        keys_str = os.environ.get("LLM_KEYS", "")
        if not keys_str:
            raise ValueError("LLM_KEYS environment variable must be set with comma-separated API keys")
        keys = [key.strip() for key in keys_str.split(",") if key.strip()]
        if not keys:
            raise ValueError("LLM_KEYS must contain at least one non-empty API key")
        self.keys = KeyRotator(keys)

    async def complete(self, prompt: str):
        """Complete with resilience."""
        # 1. Circuit Breaker Check
        return await self.breaker.call_async(self._unsafe_complete, prompt)

    async def _unsafe_complete(self, prompt):
        """Internal completion with retry and key rotation."""
        # 2. Retry Logic
        for attempt in range(3):
            try:
                api_key = self.keys.get_current_key()
                # CALL PROVIDER HERE with prompt and api_key
                # Securely call provider without logging sensitive key
                return f"response to: {prompt}"  # Placeholder
            except Exception as e:
                if "429" in str(e):
                    # Exponential Backoff
                    sleep_time = (2 ** attempt) + random.random()
                    await asyncio.sleep(sleep_time)
                    self.keys.rotate()  # Switch key on failure
                else:
                    raise e


LLMClient = ResilientLLMClient

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client() -> None:
    global _llm_client
    _llm_client = None
