from ..provider import BaseLLMProvider
from ..registry import register_provider
from typing import AsyncGenerator
import anthropic

@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) Provider"""

    price_per_input_token = 3.00 / 1_000_000
    price_per_output_token = 15.00 / 1_000_000

    def __init__(self, config):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model") or self.config.model_name or "claude-3-5-sonnet-20240620"

        async def _call_api():
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4096),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        model = kwargs.get("model") or self.config.model_name or "claude-3-5-sonnet-20240620"

        stream = await self.client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        async for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
