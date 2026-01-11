"""
LLM Providers - Auto-registered provider classes.
"""

# Import all providers to trigger @register_provider decorators
from .anthropic import AnthropicProvider
from .azure import AzureProvider
from .base_provider import BaseLLMProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AzureProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
]
