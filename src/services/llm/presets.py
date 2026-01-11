"""
Provider Presets
================

Configuration for LLM provider presets used in UI dropdowns.
This provides the data for provider selection interfaces.

Usage:
    from src.services.llm.presets import get_provider_presets, API_PROVIDER_PRESETS, LOCAL_PROVIDER_PRESETS
"""

# Provider presets for API and Local providers
API_PROVIDER_PRESETS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o",
        "description": "Official OpenAI API with GPT models"
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1",
        "requires_key": True,
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "default_model": "claude-3-5-sonnet-20241022",
        "description": "Anthropic's Claude models via official API"
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "requires_key": True,
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        "default_model": "gemini-1.5-flash",
        "description": "Google's Gemini models via AI Studio API"
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "requires_key": True,
        "models": ["deepseek-chat", "deepseek-coder"],
        "default_model": "deepseek-chat",
        "description": "DeepSeek's efficient and affordable models"
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "requires_key": True,
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        "default_model": "llama3-8b-8192",
        "description": "Fast inference with Groq's LPU technology"
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "meta-llama/llama-3.1-405b-instruct"],
        "default_model": "anthropic/claude-3.5-sonnet",
        "description": "Unified API for multiple AI providers"
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "requires_key": True,
        "models": ["meta-llama/Llama-3-70b-chat-hf", "meta-llama/Llama-3-8b-chat-hf"],
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "description": "Open-source models hosted by Together AI"
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "requires_key": True,
        "models": ["mistral-large-latest", "mistral-medium", "mistral-small"],
        "default_model": "mistral-large-latest",
        "description": "Mistral's advanced language models"
    },
    "azure_openai": {
        "name": "Azure OpenAI",
        "base_url": "",  # User must provide their Azure endpoint
        "requires_key": True,
        "models": ["gpt-4", "gpt-35-turbo"],  # Deployment names, user must configure
        "default_model": "gpt-4",
        "description": "OpenAI models hosted on Azure",
        "requires_endpoint": True
    }
}

LOCAL_PROVIDER_PRESETS = {
    "ollama": {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "requires_key": False,
        "models": ["llama3.2", "llama3.3", "qwen2.5", "qwen3", "mistral-nemo", "deepseek-r1", "gemma2", "phi3"],
        "default_model": "llama3.2",
        "description": "Run models locally with Ollama",
        "setup_instructions": "Install Ollama and run 'ollama serve'"
    },
    "lm_studio": {
        "name": "LM Studio",
        "base_url": "http://127.0.0.1:1234/v1",
        "requires_key": False,
        "models": [],  # Dynamic - loaded from server
        "default_model": "local-model",
        "description": "Local OpenAI-compatible API server",
        "setup_instructions": "Download LM Studio and start local server"
    },
    "llama_cpp": {
        "name": "llama.cpp Server",
        "base_url": "http://localhost:8080/v1",
        "requires_key": False,
        "models": [],  # Dynamic - loaded from server
        "default_model": "local-model",
        "description": "C++ implementation of Llama models",
        "setup_instructions": "Run llama.cpp server with --server flag"
    },
    "vllm": {
        "name": "vLLM",
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "models": [],  # Dynamic - loaded from server
        "default_model": "local-model",
        "description": "High-throughput inference server",
        "setup_instructions": "Install vLLM and run server"
    },
    "custom": {
        "name": "Custom Local Server",
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "models": [],  # User specified
        "default_model": "",
        "description": "Any OpenAI-compatible local server",
        "setup_instructions": "Configure your server URL and model manually"
    }
}


def get_provider_presets():
    """
    Get provider presets for API and Local providers.

    Returns:
        Dict containing:
        - api: API provider presets
        - local: Local provider presets
    """
    return {
        "api": API_PROVIDER_PRESETS,
        "local": LOCAL_PROVIDER_PRESETS
    }
