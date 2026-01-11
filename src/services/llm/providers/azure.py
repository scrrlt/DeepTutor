from typing import AsyncGenerator, Dict
import os

import openai

from .base_provider import BaseLLMProvider
from ..registry import register_provider


@register_provider("azure")
class AzureProvider(BaseLLMProvider):
    """Azure OpenAI Provider."""

    def __init__(self, config):
        super().__init__(config)

        self.deployment_map = {
            "gpt-4": os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4"),
            "gpt-3.5-turbo": os.getenv("AZURE_GPT35_DEPLOYMENT", "gpt-35-turbo"),
        }

        self.client = openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=self.base_url or os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    @property
    def extra_headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key}

    def resolve_model(self, requested_model: str) -> str:
        return self.deployment_map.get(requested_model, requested_model)

    async def complete(self, prompt: str, **kwargs) -> str:
        model = self.resolve_model(kwargs.get("model") or self.config.model_name)

        async def _call_api():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        model = self.resolve_model(kwargs.pop("model", None) or self.config.model_name)

        async def _create_stream():
            return await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs
            )

        stream = await self.execute_with_retry(_create_stream)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
