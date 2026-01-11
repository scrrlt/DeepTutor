"""
Azure OpenAI Provider.
"""


import aiohttp
from typing import AsyncGenerator, Dict, List, Optional

from ..exceptions import LLMAPIError
from ..registry import register_provider
from ..utils import extract_response_content
from .base_provider import BaseLLMProvider


@register_provider("azure")
class AzureProvider(BaseLLMProvider):
    """Azure OpenAI provider."""

    def _default_base_url(self) -> str:
        return ""  # Must be provided

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Azure uses api-key header."""
        return {"api-key": self.api_key}

    def resolve_model(self, requested_model: str) -> str:
        """Resolve to deployment name."""
        model_map = {
            "gpt-4": self.deployment or "gpt-4",
            "gpt-3.5-turbo": self.deployment or "gpt-35-turbo",
        }
        return model_map.get(requested_model, requested_model)

    async def complete(self, prompt: str, system_prompt: str = "", model: str = "gpt-4", **kwargs) -> str:
        """Complete using Azure OpenAI."""
        return await self._azure_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def stream(self, prompt: str, system_prompt: str = "", model: str = "gpt-4", messages: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream using Azure OpenAI."""
        async for chunk in self._azure_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        ):
            yield chunk

    async def _azure_complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Azure OpenAI completion."""
        if not self.base_url:
            raise LLMAPIError("Azure OpenAI requires a base_url")

        # Azure URL format
        api_version = self.api_version or "2023-05-15"
        url = f"{self.base_url.rstrip('/')}/openai/deployments/{model}/chat/completions?api-version={api_version}"

        headers = self.extra_headers
        headers["Content-Type"] = "application/json"

        # Handle messages
        messages = kwargs.get("messages")
        if messages:
            data = {"messages": messages}
        else:
            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }

        # Add params
        data.update({
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        })

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
                    raise LLMAPIError(f"Azure OpenAI API error: {error_text}", status_code=resp.status, provider="azure")

        raise LLMAPIError("Azure OpenAI completion failed")

    async def _azure_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Azure OpenAI streaming."""
        import json

        if not self.base_url:
            raise LLMAPIError("Azure OpenAI requires a base_url")

        api_version = self.api_version or "2023-05-15"
        url = f"{self.base_url.rstrip('/')}/openai/deployments/{model}/chat/completions?api-version={api_version}"

        headers = self.extra_headers
        headers["Content-Type"] = "application/json"

        # Build messages
        if messages:
            msg_list = messages
        else:
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        data = {
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
                    raise LLMAPIError(f"Azure OpenAI stream error: {error_text}", status_code=resp.status, provider="azure")

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

    def validate_config(self):
        """Validate Azure config."""
        if not self.deployment:
            raise ValueError("AZURE_DEPLOYMENT_NAME is required for Azure provider")

    def __init__(self, provider_name: str):
        super().__init__(provider_name)
        # Azure-specific: extract deployment from URL if full path provided
        if self.base_url and '/openai/deployments/' in self.base_url:
            parts = self.base_url.split('/openai/deployments/')
            self.base_url = parts[0]
            if not self.deployment:
                self.deployment = parts[1].split('/')[0] if '/' in parts[1] else parts[1]
