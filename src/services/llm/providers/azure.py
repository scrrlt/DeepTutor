from ..provider import BaseLLMProvider
from ..registry import register_provider
from typing import AsyncGenerator
import openai
import os

@register_provider("azure")
class AzureProvider(BaseLLMProvider):
    """
    Azure OpenAI Provider.
    Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION
    """

    def __init__(self, config):
        super().__init__(config)

        self.deployment_map = {
            "gpt-4": os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4"),
            "gpt-3.5-turbo": os.getenv("AZURE_GPT35_DEPLOYMENT", "gpt-35-turbo")
        }

        self.client = openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=config.api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=config.base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def resolve_model(self, requested_model: str) -> str:
        return self.deployment_map.get(requested_model, requested_model)

    async def complete(self, prompt: str, **kwargs) -> str:
        model = self.resolve_model(kwargs.get("model") or self.config.model_name)

        async def _call_api():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        model = self.resolve_model(kwargs.get("model") or self.config.model_name)

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
