"""
Ollama Provider - Local LLM server.
"""

import aiohttp

from ..exceptions import LLMAPIError
from ..registry import register_provider
from .base_provider import BaseLLMProvider


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """Ollama local provider."""

    def _default_base_url(self) -> str:
        return "http://localhost:11434"

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "llama3", **kwargs) -> str:
        """Complete using Ollama."""
        return await self._ollama_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def _ollama_complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Ollama completion."""
        url = f"{self.base_url}/api/generate"

        data = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }
        data.update(kwargs)

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("response", "")
                else:
                    error_text = await resp.text()
                    raise LLMAPIError(f"Ollama API error: {error_text}", status_code=resp.status, provider="ollama")
