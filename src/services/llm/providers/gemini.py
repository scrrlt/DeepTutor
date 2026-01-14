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
        try:
            async for chunk in await self.model.generate_content_async(prompt, stream=True):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            # Log and re-raise; retry logic for generators is complex
            raise
