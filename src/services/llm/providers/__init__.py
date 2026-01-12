"""
LLM Providers - Auto-registered provider classes.
"""

# Import all providers to trigger @register_provider decorators
from .azure import AzureProvider
try:
    from .anthropic import AnthropicProvider
except Exception:
    AnthropicProvider = None
from .base_provider import BaseLLMProvider
try:
    from .gemini import GeminiProvider
except Exception:
    GeminiProvider = None
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "OpenAIProvider",
    "AzureProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
]

if AnthropicProvider is not None:
    __all__.append("AnthropicProvider")

if GeminiProvider is not None:
    __all__.append("GeminiProvider")
