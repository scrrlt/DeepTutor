"""
Anthropic Provider - Claude API.
"""

from typing import AsyncGenerator, Dict, List, Optional

import aiohttp

from ..exceptions import LLMAPIError
from ..registry import register_provider
from .base_provider import BaseLLMProvider


@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def _default_base_url(self) -> str:
        return "https://api.anthropic.com/v1"

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Anthropic headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "claude-3-5-sonnet-20241022", **kwargs) -> str:
        """Complete using Anthropic."""
        return await self._anthropic_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def stream(self, prompt: str, system_prompt: str = "", model: str = "claude-3-5-sonnet-20241022", messages: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream using Anthropic."""
        async for chunk in self._anthropic_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        ):
            yield chunk

    async def _anthropic_complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Anthropic completion."""
        url = self.base_url.rstrip("/")
        if not url.endswith("/messages"):
            url += "/messages"

        # Build messages - Anthropic doesn't use system in messages
        data = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self.extra_headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(f"Anthropic API error: {error_text}", status_code=response.status, provider="anthropic")

                result = await response.json()
                return result["content"][0]["text"]

    async def _anthropic_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Anthropic streaming."""
        import json

        url = self.base_url.rstrip("/")
        if not url.endswith("/messages"):
            url += "/messages"

        # Build messages
        if messages:
            # Filter out system messages for Anthropic
            msg_list = [m for m in messages if m.get("role") != "system"]
            system_content = next(
                (m["content"] for m in messages if m.get("role") == "system"),
                system_prompt,
            )
        else:
            msg_list = [{"role": "user", "content": prompt}]
            system_content = system_prompt

        data = {
            "model": model,
            "system": system_content,
            "messages": msg_list,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self.extra_headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(f"Anthropic stream error: {error_text}", status_code=response.status, provider="anthropic")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()
                    if not data_str:
                        continue

                    try:
                        chunk_data = json.loads(data_str)
                        event_type = chunk_data.get("type")
                        if event_type == "content_block_delta":
                            delta = chunk_data.get("delta", {})
                            text = delta.get("text")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue
