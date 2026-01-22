"""Local OpenAI-compatible provider.

Many local servers (Ollama, LM Studio, vLLM, llama.cpp) expose an
OpenAI-compatible chat completions API. This provider uses the OpenAI SDK
with a configurable `base_url`.
"""

from __future__ import annotations

import asyncio
from typing import Any, no_type_check

import openai

from src.logging import get_logger

from ..config import LLMConfig
from ..exceptions import LLMConfigError
from ..http_client import get_shared_http_client
from ..model_rules import get_token_limit_kwargs
from ..registry import register_provider
from ..telemetry import track_llm_call
from ..types import AsyncStreamGenerator, TutorResponse, TutorStreamChunk
from .base_provider import BaseLLMProvider

logger = get_logger(__name__)


@register_provider("local_openai")
class LocalLLMProvider(BaseLLMProvider):
    """OpenAI-SDK based provider for local OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client: openai.AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> openai.AsyncOpenAI:
        if self.client is None:
            async with self._client_lock:
                if self.client is None:
                    http_client = await get_shared_http_client()
                    self.client = openai.AsyncOpenAI(
                        api_key=self.api_key or "local",
                        base_url=self.base_url or None,
                        http_client=http_client,
                    )
        return self.client

    @track_llm_call("local_openai")
    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for local provider")

        # Default token limit from config, using model-specific param name.
        requested_max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_completion_tokens", None)
            or getattr(self.config, "max_tokens", 4096)
        )
        kwargs.update(get_token_limit_kwargs(model, int(requested_max_tokens)))

        async def _call_api() -> TutorResponse:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else {}

            raw_response: dict[str, Any] = {}
            if hasattr(response, "model_dump"):
                try:
                    dumped = response.model_dump()
                    if isinstance(dumped, dict):
                        raw_response = dumped
                except Exception:
                    raw_response = {}

            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None and not isinstance(
                finish_reason, str
            ):
                finish_reason = str(finish_reason)

            return TutorResponse(
                content=choice.message.content or "",
                raw_response=raw_response,
                usage=usage,
                provider=self.config.binding,
                model=model,
                finish_reason=finish_reason,
                cost_estimate=self.calculate_cost(usage),
            )

        return await self.execute(_call_api)

    @no_type_check
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for local provider")

        requested_max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_completion_tokens", None)
            or getattr(self.config, "max_tokens", 4096)
        )
        kwargs.update(get_token_limit_kwargs(model, int(requested_max_tokens)))

        async def _create_stream():
            client = await self._get_client()
            return await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )

        stream = await self.execute_with_retry(_create_stream)
        accumulated = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated += delta
                yield TutorStreamChunk(
                    content=delta,
                    delta=delta,
                    provider=self.config.binding,
                    model=model,
                    is_complete=False,
                )

        yield TutorStreamChunk(
            content="",
            delta="",
            provider=self.config.binding,
            model=model,
            is_complete=True,
        )
