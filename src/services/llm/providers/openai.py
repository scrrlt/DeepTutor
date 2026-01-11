from ..provider import BaseLLMProvider
from ..registry import register_provider
from ..types import TutorResponse, TutorStreamChunk, AsyncStreamGenerator
from ..telemetry import track_llm_call
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

    @track_llm_call("openai")
    async def complete(self, prompt: str, **kwargs) -> TutorResponse:
        model = kwargs.pop("model", None) or self.config.model_name or "gpt-4o"
        kwargs.pop("stream", None)

        async def _call_api():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            
            # Convert to standardized response
            choice = response.choices[0]
            usage = response.usage.model_dump() if response.usage else {}
            cost = self.calculate_cost(usage)
            
            return TutorResponse(
                content=choice.message.content or "",
                raw_response=response.model_dump(),
                usage=usage,
                provider="openai",
                model=model,
                finish_reason=choice.finish_reason,
                cost_estimate=cost
            )

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncStreamGenerator:
        model = kwargs.pop("model", None) or self.config.model_name or "gpt-4o"
        kwargs.pop("stream", None)

        async def _create_stream():
            return await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs
            )

        stream = await self.execute_with_retry(_create_stream)
        accumulated_content = ""
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated_content += delta
                
                yield TutorStreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    provider="openai",
                    model=model,
                    is_complete=False
                )
        
        # Final chunk with usage info if available
        yield TutorStreamChunk(
            content=accumulated_content,
            delta="",
            provider="openai",
            model=model,
            is_complete=True,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
