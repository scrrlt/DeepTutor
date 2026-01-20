# -*- coding: utf-8 -*-
"""
OpenAI/Azure LLM provider implementation.
"""

import logging
import os
from typing import Any

import httpx
import openai

from ..config import LLMConfig
from ..exceptions import LLMConfigError
from ..registry import register_provider
from ..telemetry import track_llm_call
from ..types import AsyncStreamGenerator, TutorResponse, TutorStreamChunk
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Production-ready OpenAI/Azure Provider."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the OpenAI/Azure provider.

        Args:
            config: Provider configuration object.

        Returns:
            None.

        Raises:
            LLMConfigError: If Azure binding is used without api_version.
            Exception: Propagates client initialization failures.
        """
        super().__init__(config)

        http_client = None
        # Allow disabling SSL verification for development/testing environments
        # This is controlled by environment variable for explicit opt-in
        disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "").lower()
        if disable_ssl in ("true", "1", "yes"):
            logger.warning(
                "SSL verification is DISABLED for OpenAIProvider HTTP client. "
                "This is insecure and must never be used in production environments. "
                "Unset the DISABLE_SSL_VERIFY environment variable to re-enable SSL verification."
            )
            http_client = httpx.AsyncClient(verify=False)  # nosec B501
        self._http_client = http_client

        binding = getattr(self.config, "binding", "")
        binding_lower = binding.lower() if isinstance(binding, str) else ""
        is_azure = (
            binding_lower in ("azure", "azure_openai")
            or "openai.azure.com" in (self.base_url or "")
        )
        api_version = getattr(self.config, "api_version", None)

        if is_azure:
            if not api_version:
                raise LLMConfigError("Azure OpenAI requires api_version in configuration.")
            if not self.base_url:
                raise LLMConfigError("Azure OpenAI requires base_url in configuration.")
            self.client: openai.AsyncAzureOpenAI | openai.AsyncOpenAI = openai.AsyncAzureOpenAI(
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
            )

    @property
    def provider_label(self) -> str:
        """
        Resolve the provider label based on the active client type.

        Returns:
            Provider label string.
        """
        return "azure" if isinstance(self.client, openai.AsyncAzureOpenAI) else "openai"

    async def close(self) -> None:
        """
        Close any custom HTTP client resources.

        Returns:
            None.
        """
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "OpenAIProvider":
        """
        Enter async context manager.

        Returns:
            OpenAIProvider instance.

        Raises:
            None.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """
        Exit async context manager and release resources.

        Args:
            exc_type: Exception type if raised.
            exc: Exception instance if raised.
            traceback: Traceback object if raised.

        Returns:
            None.

        Raises:
            None.
        """
        await self.close()

    @track_llm_call("openai")
    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        """
        Generate a completion using OpenAI or Azure OpenAI.

        Args:
            prompt: User prompt content.
            **kwargs: Provider-specific options.

        Returns:
            TutorResponse containing the completion result.

        Raises:
            Exception: Propagates SDK or execution errors.
        """
        self._check_deprecated_kwargs(kwargs)

        model = (
            kwargs.pop("model", None)
            or getattr(self.config, "model", None)
            or getattr(self.config, "model_name", None)
        )
        if not model:
            raise ValueError("Model must be specified in config or call args.")

        kwargs.pop("stream", None)
        kwargs.pop("max_retries", None)

        async def _call_api() -> TutorResponse:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            if not response.choices:
                raise ValueError("OpenAI returned no choices in response.")

            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else {}

            return TutorResponse(
                content=choice.message.content or "",
                raw_response=response.model_dump(),
                usage=usage,
                provider=self.provider_label,
                model=model,
                finish_reason=choice.finish_reason,
                cost_estimate=self.calculate_cost(usage),
            )

        return await self.execute_guarded(_call_api)

    @track_llm_call("openai")
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        """
        Stream a completion from OpenAI or Azure OpenAI.

        Args:
            prompt: User prompt content.
            **kwargs: Provider-specific options.

        Returns:
            AsyncStreamGenerator yielding TutorStreamChunk items.

        Raises:
            Exception: Propagates SDK or execution errors.
        """
        self._check_deprecated_kwargs(kwargs)

        model = (
            kwargs.pop("model", None)
            or getattr(self.config, "model", None)
            or getattr(self.config, "model_name", None)
        )
        if not model:
            raise ValueError("Model must be specified in config or call args.")

        kwargs.pop("max_retries", None)
        stream_options = {"include_usage": True}

        async def _create_stream():
            return await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options=stream_options,
                **kwargs,
            )

        stream = await self.execute_guarded(_create_stream)
        accumulated_content = ""
        final_usage: dict[str, int] | None = None

        try:
            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    final_usage = chunk.usage.model_dump()

                if (
                    chunk.choices
                    and len(chunk.choices) > 0
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    delta = chunk.choices[0].delta.content
                    accumulated_content += delta

                    yield TutorStreamChunk(
                        content=accumulated_content,
                        delta=delta,
                        provider=self.provider_label,
                        model=model,
                        is_complete=False,
                    )
        except Exception as e:
            logger.error("OpenAI streaming iteration failed: %s", e)
            raise

        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider=self.provider_label,
            model=model,
            is_complete=True,
            usage=final_usage or {},
            cost_estimate=self.calculate_cost(final_usage or {}),
        )
