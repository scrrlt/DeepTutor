"""OpenAI provider implementation using shared HTTP client."""

import asyncio
import inspect
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

from src.logging import get_logger



logger = get_logger(__name__)

@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Production-ready OpenAI Provider with shared HTTP client."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize OpenAI provider with shared client."""
        super().__init__(config)
        self.client: openai.AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> openai.AsyncOpenAI:
        if self.client is None:
            async with self._client_lock:
                if self.client is None:
                    http_client = await get_shared_http_client()
                    self.client = openai.AsyncOpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url or None,
                        http_client=http_client,
                    )
        return self.client

    @track_llm_call("openai")
    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        """Complete a prompt using OpenAI chat completions."""
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for OpenAI provider")
        kwargs.pop("stream", None)

        requested_max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_completion_tokens", None)
            or getattr(self.config, "max_tokens", 4096)
        )
        kwargs.update(get_token_limit_kwargs(model, int(requested_max_tokens)))

        async def _call_api():
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else {}

            return TutorResponse(
                content=content,
                raw_response=raw_response,
                usage=usage,
                provider="azure" if isinstance(self.client, openai.AsyncAzureOpenAI) else "openai",
                model=model,
                finish_reason=finish_reason,
                cost_estimate=self.calculate_cost(usage),
            )

        return await self.execute(_call_api)

    @no_type_check
    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncStreamGenerator:  # type: ignore[override]
        """Stream chat completion deltas from OpenAI."""
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for OpenAI provider")

        async def _create_stream():
            client = await self._get_client()
            return await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )

        stream = await self.execute(_create_stream)
        accumulated_content = ""
        provider_label = "azure" if isinstance(self.client, openai.AsyncAzureOpenAI) else "openai"
        final_usage = None

        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    accumulated_content += delta

                yield TutorStreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    provider="openai",
                    model=model,
                    is_complete=False,
                )

        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider="openai",
            model=model,
            is_complete=True,
        )
