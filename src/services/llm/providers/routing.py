"""Routing provider bridging legacy provider functions.

This provider delegates to the existing function-based providers
(`cloud_provider` / `local_provider`) while inheriting the hardened
execution pipeline from `BaseLLMProvider` (traffic control, circuit
breaker, and exception mapping).

It exists to keep the public API stable while incrementally migrating
call sites to provider objects.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.logging import get_logger

from .. import cloud_provider, local_provider
from ..config import LLMConfig
from ..exceptions import LLMConfigError
from ..registry import register_provider
from ..types import AsyncStreamGenerator, TutorResponse, TutorStreamChunk
from ..utils import is_local_llm_server
from .base_provider import BaseLLMProvider

logger = get_logger(__name__)


@register_provider("routing")
class RoutingProvider(BaseLLMProvider):
    """Provider that routes between cloud and local function providers."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # Use per-route provider name for circuit-breaker/metrics when possible.
        if is_local_llm_server(self.base_url or ""):
            self.provider_name = "local"

    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        """Complete via local_provider/cloud_provider with retries."""
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model is required")

        system_prompt = kwargs.pop(
            "system_prompt", "You are a helpful assistant."
        )
        messages = kwargs.pop("messages", None)
        max_retries = int(kwargs.pop("max_retries", 3))
        sleep = kwargs.pop("sleep", None)

        use_cache = bool(kwargs.pop("use_cache", True))
        cache_ttl_seconds = kwargs.pop("cache_ttl_seconds", None)
        cache_key = kwargs.pop("cache_key", None)

        call_kwargs = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "messages": messages,
            **kwargs,
        }

        if is_local_llm_server(self.base_url or ""):
            target = local_provider.complete
        else:
            binding = kwargs.pop("binding", None) or self.config.binding
            api_version = (
                kwargs.pop("api_version", None) or self.config.api_version
            )
            call_kwargs["binding"] = binding
            call_kwargs["api_version"] = api_version
            target = cloud_provider.complete

        async def _call() -> str:
            if use_cache:
                # Import lazily to keep routing provider lightweight.
                from ..cache import (
                    DEFAULT_CACHE_TTL,
                    build_completion_cache_key,
                    get_cached_completion,
                    set_cached_completion,
                )

                computed_cache_key = cache_key or build_completion_cache_key(
                    model=model,
                    binding=str(call_kwargs.get("binding") or "openai"),
                    base_url=call_kwargs.get("base_url"),
                    system_prompt=system_prompt,
                    prompt=prompt,
                    messages=messages,
                    **{
                        k: v
                        for k, v in call_kwargs.items()
                        if k
                        not in {
                            "prompt",
                            "system_prompt",
                            "messages",
                            "model",
                            "api_key",
                            "base_url",
                            "binding",
                        }
                    },
                )
                cached = await get_cached_completion(computed_cache_key)
                if cached is not None:
                    return cached

                result = await target(**call_kwargs)
                await set_cached_completion(
                    computed_cache_key,
                    result,
                    ttl_seconds=int(cache_ttl_seconds or DEFAULT_CACHE_TTL),
                )
                return result

            return await target(**call_kwargs)

        text = await self.execute_with_retry(
            _call,
            max_retries=max_retries,
            sleep=sleep,
        )

        return TutorResponse(
            content=str(text),
            raw_response={},
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            provider=self.provider_name,
            model=model,
            finish_reason=None,
            cost_estimate=0.0,
        )

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        """Stream via local_provider/cloud_provider.

        Retry applies only to failures before the first yielded chunk.
        """
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model is required")

        system_prompt = kwargs.pop(
            "system_prompt", "You are a helpful assistant."
        )
        messages = kwargs.pop("messages", None)
        max_retries = int(kwargs.pop("max_retries", 3))

        call_kwargs = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "messages": messages,
            **kwargs,
        }

        if is_local_llm_server(self.base_url or ""):
            stream_func = local_provider.stream
        else:
            binding = kwargs.pop("binding", None) or self.config.binding
            api_version = (
                kwargs.pop("api_version", None) or self.config.api_version
            )
            call_kwargs["binding"] = binding
            call_kwargs["api_version"] = api_version
            stream_func = cloud_provider.stream

        attempt = 0
        while True:
            attempt += 1
            iterator = None
            emitted_any = False
            accumulated = ""

            try:
                iterator = stream_func(**call_kwargs)
                async for delta in iterator:
                    emitted_any = True
                    accumulated += str(delta)
                    yield TutorStreamChunk(
                        content=accumulated,
                        delta=str(delta),
                        provider=self.provider_name,
                        model=model,
                        is_complete=False,
                    )

                yield TutorStreamChunk(
                    content=accumulated,
                    delta="",
                    provider=self.provider_name,
                    model=model,
                    is_complete=True,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                mapped = self._map_exception(exc)
                if emitted_any:
                    raise mapped from exc

                if attempt > max_retries + 1 or not self._should_retry_error(
                    mapped
                ):
                    raise mapped from exc

                delay_seconds = min(60.0, 1.5**attempt)
                logger.warning(
                    "Stream start failed (attempt %d/%d): %s; retrying in %.2fs"
                    % (attempt, max_retries + 1, mapped, delay_seconds)
                )
                await asyncio.sleep(delay_seconds)
                continue
