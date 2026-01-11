"""
OpenAI Compatible Provider - For providers like DeepSeek, Qwen, etc.
"""

from ..registry import register_provider

from .base_provider import BaseLLMProvider
KNOWN_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://api.qwen.ai",  # Example
    # Add more as needed
}


@register_provider("openai_compatible")
class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible provider for various endpoints."""

    def __init__(self, provider_name: str, flavor: str = None):
        # For flavors, override base_url
        if flavor and flavor in KNOWN_ENDPOINTS:
            self.base_url = KNOWN_ENDPOINTS[flavor]
        super().__init__(provider_name)

    def _default_base_url(self) -> str:
        return "https://api.openai.com/v1"  # Fallback

    # Reuse OpenAI methods, but with custom base_url

    async def complete(self, prompt: str, **kwargs) -> str:
        # Delegate to OpenAI provider logic
        from .openai import OpenAIProvider

        provider = OpenAIProvider(self.provider_name)
        provider.base_url = self.base_url
        return await provider.complete(prompt, **kwargs)

    def calculate_cost(self, usage) -> float:
        # Use OpenAI pricing as default
        return 0.0

    def validate_config(self):
        pass
