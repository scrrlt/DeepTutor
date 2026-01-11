import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class BaseProviderConfig(BaseModel):
    """Base configuration for all LLM providers."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    
    # PM NOTE: These manual os.getenv calls inside __init__ are exactly 
    # what we need to move to Pydantic-Settings classes.
    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            self.api_key = os.getenv(f"{self.__class__.__name__.upper().replace('CONFIG', '')}_API_KEY")

class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI and compatible providers."""
    organization: Optional[str] = os.getenv("OPENAI_ORG_ID")
    
    @validator("base_url", pre=True, always=True)
    def set_default_url(cls, v):
        return v or "https://api.openai.com/v1"

class AzureConfig(BaseProviderConfig):
    """Configuration for Azure OpenAI."""
    api_version: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
    deployment_name: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @validator("api_key")
    def validate_azure_key(cls, v):
        if not v:
            raise ValueError("Azure requires an API Key")
        return v

class GeminiConfig(BaseProviderConfig):
    """Configuration for Google Gemini."""
    safety_threshold: str = os.getenv("GEMINI_SAFETY_LEVEL", "BLOCK_MEDIUM_AND_ABOVE")

class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic Claude."""
    @validator("base_url", pre=True, always=True)
    def set_anthropic_url(cls, v):
        return v or "https://api.anthropic.com/v1"

class DeepSeekConfig(BaseProviderConfig):
    """Configuration for DeepSeek."""
    @validator("base_url", pre=True, always=True)
    def set_deepseek_url(cls, v):
        return v or "https://api.deepseek.com"
