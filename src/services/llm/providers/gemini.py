from ..provider import BaseLLMProvider
from ..registry import register_provider
from typing import AsyncGenerator
import google.generativeai as genai
import asyncio
import random

@register_provider("gemini")
class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini Provider.
    """

    def __init__(self, config):
        super().__init__(config)
        if not self.api_key:
            raise ValueError("Gemini requires API Key")

        genai.configure(api_key=self.api_key)
        model_name = config.model_name or "gemini-1.5-flash"
        self.model = genai.GenerativeModel(model_name)

    async def complete(self, prompt: str, **kwargs) -> str:
        async def _call_api():
            response = await self.model.generate_content_async(prompt)
            return response.text

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        async def _create_stream():
            return await self.model.generate_content_async(prompt, stream=True)

        stream = await self.execute_with_retry(_create_stream)

        async for chunk in stream:
            if chunk.text:
                yield chunk.text
