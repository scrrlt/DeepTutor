"""
Smart Router - Route users to LLM providers based on rules.
"""

import hashlib

from ..utils.feature_flags import flag


class LLMRouter:
    """Router for LLM providers with A/B testing and feature flags."""

    def __init__(self):
        self.flags = flag  # Use the flag function
        self.routes = {"openai": 0.8, "anthropic": 0.2}

    def get_provider(self, user_id: str) -> str:
        """Get provider for user, considering flags and weights."""
        # Check flags first (from feature_flags.py logic)
        if self.flags('force_openai'):
            return "openai"

        # Hash logic from ab_router.py
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized = (hash_val % 100) / 100.0

        cumulative = 0.0
        for provider, weight in self.routes.items():
            cumulative += weight
            if normalized < cumulative:
                return provider
        return "openai"
