from ..provider import BaseLLMProvider
from ..registry import register_provider
from .openai import OpenAIProvider
import os

KNOWN_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "groq": "https://api.groq.com/openai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "together": "https://api.together.xyz/v1",
    "perplexity": "https://api.perplexity.ai",
    "mistral": "https://api.mistral.ai/v1",
    "moonshot": "https://api.moonshot.cn/v1"
}

@register_provider("openai_compatible")
class OpenAICompatibleProvider(OpenAIProvider):
    """
    Universal adapter for any provider mimicking OpenAI.
    Inherits retry and stream logic from OpenAIProvider.
    """

    def __init__(self, config):
        self.flavor = os.getenv(f"{config.env_prefix}_FLAVOR", "").lower()

        # Auto-configure URL
        if not config.base_url and self.flavor in KNOWN_ENDPOINTS:
            config.base_url = KNOWN_ENDPOINTS[self.flavor]

        # Auto-configure API Key override
        flavor_key = os.getenv(f"{self.flavor.upper()}_API_KEY")
        if flavor_key:
            config.api_key = flavor_key

        super().__init__(config)
