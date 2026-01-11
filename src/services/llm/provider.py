from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Handles common logic like cost calculation and model resolution.
    """
    
    # Default pricing (can be overridden by subclasses or config)
    price_per_input_token = 0.0
    price_per_output_token = 0.0
    provider_name = "base"

    def __init__(self, config: Any):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        
        # Optional: Allow overriding pricing from config
        if hasattr(config, "pricing"):
            self.price_per_input_token = config.pricing.get("input", self.price_per_input_token)
            self.price_per_output_token = config.pricing.get("output", self.price_per_output_token)

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Generates a text completion for the given prompt.
        """
        pass

    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """
        Calculates the cost of a request based on token usage.
        
        Args:
            usage: Dict with 'prompt_tokens' and 'completion_tokens'
        """
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        cost = (input_tokens * self.price_per_input_token) + \
               (output_tokens * self.price_per_output_token)
        return round(cost, 6)

    def resolve_model(self, requested_model: str) -> str:
        """
        Resolves the actual model name to send to the API.
        Useful for Azure deployment mapping.
        """
        # Default behavior: pass through
        return requested_model
