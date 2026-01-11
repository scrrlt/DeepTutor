from dataclasses import dataclass, field
import os
from typing import Optional, Dict


@dataclass
class LLMConfig:
    """
    Unified Configuration Object for all LLM Providers.
    Automatically fetches ENV vars based on provider name.
    """
    provider_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_version: Optional[str] = None
    env_prefix: str = field(init=False)
    pricing: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Determine ENV prefix (e.g. "OPENAI", "AZURE", "ANTHROPIC")
        self.env_prefix = self.provider_name.upper()

        # 1. Load API Key (if not passed explicitly)
        if not self.api_key:
            self.api_key = os.getenv(f"{self.env_prefix}_API_KEY")

        # 2. Load Base URL
        if not self.base_url:
            self.base_url = os.getenv(f"{self.env_prefix}_BASE_URL") or os.getenv(
                f"{self.env_prefix}_ENDPOINT"
                )

        # 3. Load Model Name
        if not self.model_name:
            self.model_name = os.getenv(f"{self.env_prefix}_MODEL")

        # 4. Load API Version (Common for Azure)
        if not self.api_version:
            self.api_version = os.getenv(f"{self.env_prefix}_API_VERSION")

        # 5. Load Pricing (optional, for cost calculation)
        if not self.pricing:
            input_price = os.getenv(f"{self.env_prefix}_PRICE_PER_INPUT_TOKEN")
            output_price = os.getenv(f"{self.env_prefix}_PRICE_PER_OUTPUT_TOKEN")
            if input_price and output_price:
                self.pricing = {
                    "input": float(input_price),
                    "output": float(output_price)
                }

    @property
    def is_valid(self) -> bool:
        """Basic validation check"""
        # Local providers like Ollama might not need API keys
        if self.provider_name == "ollama":
            return True
        return bool(self.api_key)
