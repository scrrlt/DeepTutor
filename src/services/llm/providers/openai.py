from ..provider import BaseLLMProvider
from ..registry import register_provider
from typing import AsyncGenerator
import openai
import os

@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Standard OpenAI Provider"""

    price_per_input_token = 5.00 / 1_000_000
    price_per_output_token = 15.00 / 1_000_000

    def __init__(self, config):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url  # Optional override
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model") or self.config.model_name or "gpt-4o"

        async def _call_api():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        model = kwargs.get("model") or self.config.model_name or "gpt-4o"

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
