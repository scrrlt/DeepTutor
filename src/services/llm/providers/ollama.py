from ..provider import BaseLLMProvider
from ..registry import register_provider
from typing import AsyncGenerator
import httpx
import json

@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """
    Local Ollama Provider.
    """

    def __init__(self, config):
        super().__init__(config)
        self.base_url = self.base_url or "http://localhost:11434"
        self.model_name = config.model_name or "llama3"

    async def complete(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model") or self.model_name

        async def _call_api():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=kwargs.get("timeout", 120.0)
                )
                response.raise_for_status()
                return response.json().get("response", "")

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        model = kwargs.get("model") or self.model_name

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=kwargs.get("timeout", 120.0)
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except:
                            pass
