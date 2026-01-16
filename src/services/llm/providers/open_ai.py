"""OpenAI provider implementation using shared HTTP client."""

from typing import Any, no_type_check

import openai

from ..config import LLMConfig
from ..exceptions import LLMConfigError
from ..http_client import get_shared_http_client
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

    async def _get_client(self) -> openai.AsyncOpenAI:
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
                content=choice.message.content or "",
                raw_response=response.model_dump(),
                usage=usage,
                provider="openai",
                model=model,
                finish_reason=choice.finish_reason,
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
        except openai.APIConnectionError as e:
            logger.error(f"Stream connection failed: {e}")
            raise
        except TypeError as exc:
            # Compatibility shim for OpenAI APIConnectionError signature changes
            try:
                error = openai.APIConnectionError(
                    request=None, message=str(exc)
                )
                logger.error(f"Stream connection failed: {error}")
                raise error from exc
            except TypeError:
                raise

        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider="openai",
            model=model,
            is_complete=True,
        )
