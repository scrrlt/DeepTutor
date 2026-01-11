"""
Google Gemini Provider.
"""

from typing import AsyncGenerator, Dict, List, Optional

import aiohttp

from ..exceptions import LLMAPIError
from ..registry import register_provider
from .base_provider import BaseLLMProvider


@register_provider("gemini")
class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def _default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "gemini-2.0-flash-exp", messages: Optional[List[Dict]] = None, **kwargs) -> str:
        """Complete using Gemini."""
        return await self._gemini_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        )

    async def stream(self, prompt: str, system_prompt: str = "", model: str = "gemini-2.0-flash-exp", messages: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream using Gemini."""
        async for chunk in self._gemini_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        ):
            yield chunk

    async def _gemini_complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Gemini completion."""
        url = f"{self.base_url.rstrip('/')}/models/{model}:generateContent?key={self.api_key}"

        headers = {"Content-Type": "application/json"}

        # Build parts
        parts = []
        if messages:
            last_msg = messages[-1] if messages else None
            if last_msg and isinstance(last_msg.get("content"), list):
                for item in last_msg["content"]:
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        img_url = item["image_url"]["url"]
                        if "," in img_url:
                            img_data = img_url.split(",", 1)[1]
                            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_data}})
        if not parts:
            parts = [{"text": prompt}]

        data = {
            "contents": [{"parts": parts}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }

        if system_prompt and system_prompt != "You are a helpful assistant.":
            data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(f"Gemini API error: {error_text}", status_code=response.status, provider="gemini")

                result = await response.json()
                if "candidates" in result and result["candidates"]:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]

                raise LLMAPIError("Gemini API returned unexpected response format", provider="gemini")

    async def _gemini_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Gemini streaming."""
        import json

        url = f"{self.base_url.rstrip('/')}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        headers = {"Content-Type": "application/json"}

        # Build parts
        parts = []
        if messages:
            last_msg = messages[-1] if messages else None
            if last_msg and isinstance(last_msg.get("content"), list):
                for item in last_msg["content"]:
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        img_url = item["image_url"]["url"]
                        if "," in img_url:
                            img_data = img_url.split(",", 1)[1]
                            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_data}})

        if not parts:
            parts = [{"text": prompt}]

        data = {
            "contents": [{"parts": parts}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }

        if system_prompt and system_prompt != "You are a helpful assistant.":
            data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(f"Gemini stream error: {error_text}", status_code=response.status, provider="gemini")

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()
                    if not data_str:
                        continue

                    try:
                        chunk_data = json.loads(data_str)
                        if "candidates" in chunk_data and chunk_data["candidates"]:
                            candidate = chunk_data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]
                    except json.JSONDecodeError:
                        continue
