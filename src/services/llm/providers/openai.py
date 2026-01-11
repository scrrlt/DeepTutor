"""
OpenAI Provider - OpenAI and compatible APIs.
"""

from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from lightrag.llm.openai import openai_complete_if_cache

from ..exceptions import LLMAPIError
from ..registry import register_provider
from ..utils import extract_response_content, sanitize_url
from .base_provider import BaseLLMProvider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """OpenAI and compatible provider."""

    def _default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "gpt-4", **kwargs) -> str:
        """Complete using OpenAI-compatible API."""
        return await self._openai_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def stream(self, prompt: str, system_prompt: str = "", model: str = "gpt-4", messages: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream using OpenAI-compatible API."""
        async for chunk in self._openai_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        ):
            yield chunk

    async def _openai_complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """OpenAI-compatible completion."""
        # Sanitize URL
        base_url = sanitize_url(self.base_url, model)

        try:
            # Try using lightrag's openai_complete_if_cache first (has caching)
            response = await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=self.api_key,
                base_url=base_url,
                **kwargs,
            )
            if response:
                return response
        except Exception:
            pass  # Fall through to direct call

        # Fallback: Direct aiohttp call
        if base_url:
            url = base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url += "/chat/completions"

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if "choices" in result and result["choices"]:
                            msg = result["choices"][0].get("message", {})
                            content = extract_response_content(msg)
                            return content
                    else:
                        error_text = await resp.text()
                        raise LLMAPIError(f"OpenAI API error: {error_text}", status_code=resp.status, provider="openai")

        raise LLMAPIError("OpenAI completion failed: no valid configuration")

    async def _openai_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """OpenAI-compatible streaming."""
        import json

        base_url = sanitize_url(self.base_url, model)

        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Build messages
        if messages:
            msg_list = messages
        else:
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        data = {
            "model": model,
            "messages": msg_list,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }
        if kwargs.get("max_tokens"):
            data["max_tokens"] = kwargs["max_tokens"]

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise LLMAPIError(f"OpenAI stream error: {error_text}", status_code=resp.status, provider="openai")

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        if "choices" in chunk_data and chunk_data["choices"]:
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
