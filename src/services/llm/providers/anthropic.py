"""Anthropic provider implementation using shared HTTP client."""

from typing import Any

import anthropic

from ..config import LLMConfig
from ..exceptions import LLMConfigError
from ..http_client import get_shared_http_client
from ..registry import register_provider
from ..telemetry import track_llm_call
from ..types import AsyncStreamGenerator, TutorResponse, TutorStreamChunk
from .base_provider import BaseLLMProvider

from src.logging import get_logger

logger = get_logger(__name__)


@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider with shared HTTP client."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Anthropic provider with shared client."""
        super().__init__(config)
        self.client: anthropic.AsyncAnthropic | None = None

    async def _get_client(self) -> anthropic.AsyncAnthropic:
        if self.client is None:
            http_client = await get_shared_http_client()
            self.client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                http_client=http_client,
            )
        return self.client

    @track_llm_call("anthropic")
    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        """Complete a prompt using Anthropic Messages API."""
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for Anthropic provider")
        kwargs.pop("stream", None)

        async def _call_api():
            client = await self._get_client()
            response = await client.messages.create(
                model=model,
                max_tokens=kwargs.pop("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            content = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            return TutorResponse(
                content=content,
                raw_response=response.model_dump(),
                usage=usage,
                provider="anthropic",
                model=model,
                finish_reason=response.stop_reason,
                cost_estimate=self.calculate_cost(usage),
            )

        return await self.execute(_call_api)

    @track_llm_call("anthropic")
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        """Stream content from Anthropic Messages API."""
        model = kwargs.pop("model", None) or self.config.model
        if not model:
            raise LLMConfigError("Model not configured for Anthropic provider")
        max_tokens = kwargs.pop("max_tokens", 1024)

        async def _create_stream():
            client = await self._get_client()
            return await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )

        stream = await self.execute_with_retry(_create_stream)
        accumulated_content = ""
        usage = None

        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.text:
                delta = chunk.delta.text
                accumulated_content += delta

                yield TutorStreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    provider="anthropic",
                    model=model,
                    is_complete=False,
                )
            elif chunk.type == "message_delta" and hasattr(chunk, "usage"):
                # Extract usage from the final message delta
                usage = {
                    "input_tokens": chunk.usage.input_tokens,
                    "output_tokens": chunk.usage.output_tokens,
                }

        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider="anthropic",
            model=model,
            is_complete=True,
            usage=usage,
        )
