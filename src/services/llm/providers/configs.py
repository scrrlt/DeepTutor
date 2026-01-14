"""
Provider-Specific Configurations.
"""

import os


class AzureConfig:
    """Azure-specific configuration."""

    def __init__(self):
        self.deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

    def validate(self):
        """Validate Azure config."""
        if not self.deployment:
            raise ValueError("AZURE_DEPLOYMENT_NAME is required for Azure provider")
        # Validate deployment name format (alphanumeric, hyphens, underscores, dots)
        if not all(c.isalnum() or c in '-_.' for c in self.deployment):
            raise ValueError("AZURE_DEPLOYMENT_NAME contains invalid characters")


class OllamaConfig:
    """Ollama-specific configuration."""

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "llama3")

    def validate(self):
        """Validate Ollama config."""
        pass  # No special validation needed


class GeminiConfig:
    """Gemini-specific configuration."""

    def __init__(self):
        self.safety_settings = {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

    def validate(self):
        """Validate Gemini config."""
        pass
