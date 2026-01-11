"""
Smart Router - Route users to LLM providers based on rules.
"""

from typing import Protocol, Dict
import hashlib

from ..utils.feature_flags import flag


class RoutingStrategy(Protocol):
    """Protocol for routing strategies."""
    def select_provider(self, user_id: str, candidates: Dict[str, float]) -> str:
        ...


class HashRoutingStrategy:
    """Deterministic routing based on User ID hash."""
    def select_provider(self, user_id: str, candidates: Dict[str, float]) -> str:
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized = (hash_val % 100) / 100.0

        cumulative = 0.0
        for provider, weight in candidates.items():
            cumulative += weight
            if normalized < cumulative:
                return provider
        return list(candidates.keys())[0]


class LLMRouter:
    """Router for LLM providers with configurable strategy."""

    def __init__(self, strategy: RoutingStrategy = None):
        self.strategy = strategy or HashRoutingStrategy()
        # Configuration injected, not hardcoded
        self.routes = {"openai": 0.8, "anthropic": 0.2}

    def get_provider(self, user_id: str) -> str:
        """Get provider for user, considering flags and weights."""
        # 1. Check overrides (Feature Flags)
        if flag('force_openai'):
            return "openai"

        # 2. Delegate calculation to strategy
        return self.strategy.select_provider(user_id, self.routes)
