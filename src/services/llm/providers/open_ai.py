# -*- coding: utf-8 -*-
import os

import httpx
import openai

from ..registry import register_provider
from ..telemetry import track_llm_call
from ..types import AsyncStreamGenerator, TutorResponse, TutorStreamChunk
from .base_provider import BaseLLMProvider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Production-ready OpenAI/Azure Provider."""

    def __init__(self, config):
        super().__init__(config)

        # SSL handling for dev/troubleshooting
        http_client = None
        if os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes"):
            http_client = httpx.AsyncClient(verify=False)

        binding = getattr(self.config, "binding", "")
        binding_lower = binding.lower() if isinstance(binding, str) else ""
        is_azure = (
            binding_lower in ("azure", "azure_openai")
            or "openai.azure.com" in (self.base_url or "")
        )

        if is_azure:
            api_version = getattr(self.config, "api_version", None)
            if not api_version:
                raise ValueError("api_version is required for Azure OpenAI")

            self.client = openai.AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=api_version,
                http_client=http_client,
            )
        else:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                http_client=http_client,
                is_azure = binding_lower in ("azure", "azure_openai")
        # Check for deprecated parameters
                if is_azure:
                    api_version = getattr(self.config, "api_version", None)
                    if not api_version:
                        raise LLMConfigError(
                            "Azure OpenAI requires api_version in configuration."
                        )
                    self.client = openai.AsyncAzureOpenAI(
                        api_key=self.api_key,
                        azure_endpoint=self.base_url or None,
                        api_version=api_version,
                        http_client=http_client,
                    )
        self._check_deprecated_kwargs(kwargs)

        model = kwargs.pop("model", None) or getattr(self.config, "model_name", None)
        if not model:
            raise ValueError("Model must be specified in config or call args.")

        kwargs.pop("stream", None)

        async def _call_api():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else {}
                kwargs.pop("max_retries", None)

            return TutorResponse(
                content=choice.message.content or "",
                raw_response=response.model_dump(),
                usage=usage,
                provider="azure" if isinstance(self.client, openai.AsyncAzureOpenAI) else "openai",
                model=model,
                finish_reason=choice.finish_reason,
                cost_estimate=self.calculate_cost(usage),
            )

        return await self.execute_guarded(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncStreamGenerator:  # type: ignore[override]
        # Check for deprecated parameters
        self._check_deprecated_kwargs(kwargs)

        model = kwargs.pop("model", None) or getattr(self.config, "model_name", None)
        if not model:
            raise ValueError("Model must be specified in config or call args.")

                return await self.execute_guarded(_call_api)
        stream_options = {"include_usage": True}

        async def _create_stream():
            return await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options=stream_options,
                **kwargs,
            )

                kwargs.pop("max_retries", None)
        stream = await self.execute_guarded(_create_stream)
        accumulated_content = ""
        provider_label = "azure" if isinstance(self.client, openai.AsyncAzureOpenAI) else "openai"
        final_usage = None

        async for chunk in stream:
            # Capture usage from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                final_usage = chunk.usage.model_dump()

                stream = await self.execute_guarded(_create_stream)
                delta = chunk.choices[0].delta.content
                accumulated_content += delta

                yield TutorStreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    provider=provider_label,
                    model=model,
                    is_complete=False,
                )

        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider=provider_label,
            model=model,
            is_complete=True,
            usage=final_usage or {},
            cost_estimate=self.calculate_cost(final_usage) if final_usage else 0.0,
        )
