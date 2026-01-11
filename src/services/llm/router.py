"""
LLM Router - Failover and cost-routing capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Protocol
import hashlib

from ..types import TutorResponse, AsyncStreamGenerator
from ..exceptions import LLMRateLimitError, LLMAPIError, LLMTimeoutError
from ..utils.feature_flags import flag
from .factory import LLMFactory
from .registry import get_provider_class

logger = logging.getLogger(__name__)


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
    """
    Intelligent router with failover and cost-routing capabilities.
    
    Features:
    - Primary/fallback provider failover
    - Cost-based routing (cheap models for summarization, expensive for reasoning)
    - Provider health tracking
    - Automatic retry with fallback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, strategy: RoutingStrategy = None):
        """
        Initialize router with configuration.
        
        Args:
            config: Router configuration containing:
                - primary_provider: Primary provider name
                - fallback_provider: Fallback provider name (optional)
                - cost_thresholds: Dict mapping task types to cost limits
                - health_check_enabled: Whether to enable health checks
            strategy: Routing strategy for load balancing
        """
        self.config = config or {}
        self.strategy = strategy or HashRoutingStrategy()
        
        # Backward compatibility
        if "routes" not in self.config:
            self.config["routes"] = {"openai": 0.8, "anthropic": 0.2}
        
        self.primary_provider = self.config.get("primary_provider")
        self.fallback_provider = self.config.get("fallback_provider")
        self.cost_thresholds = self.config.get("cost_thresholds", {})
        self.health_check_enabled = self.config.get("health_check_enabled", True)
        self.routes = self.config.get("routes", {})
        
        # Health tracking
        self._provider_health: Dict[str, Dict[str, Any]] = {}
        self._last_health_check: Dict[str, float] = {}
        
        # Initialize factory
        self.factory = LLMFactory()
    
    def get_provider(self, user_id: str) -> str:
        """Get provider for user, considering flags and weights."""
        # 1. Check overrides (Feature Flags)
        if flag('force_openai'):
            return "openai"

        # 2. Delegate calculation to strategy
        return self.strategy.select_provider(user_id, self.routes)
    
    async def complete(
        self, 
        prompt: str, 
        task_type: str = "general",
        user_id: Optional[str] = None,
        **kwargs
    ) -> TutorResponse:
        """
        Complete a prompt with intelligent routing.
        
        Args:
            prompt: The prompt to complete
            task_type: Type of task (e.g., "summarization", "reasoning", "general")
            user_id: User ID for consistent routing
            **kwargs: Additional arguments for the provider
            
        Returns:
            TutorResponse: Standardized response
        """
        # Select provider based on routing strategy or cost routing
        if user_id and self.routes:
            provider_name = self.get_provider(user_id)
            selected_providers = [provider_name]
        else:
            selected_providers = self._select_providers_for_task(task_type)
        
        last_error = None
        
        for provider_name in selected_providers:
            try:
                # Check provider health if enabled
                if self.health_check_enabled and not await self._is_provider_healthy(provider_name):
                    logger.warning(f"Provider {provider_name} is unhealthy, trying next")
                    continue
                
                # Get provider and complete request
                provider = self.factory.get_provider(provider_name)
                response = await provider.complete(prompt, **kwargs)
                
                # Update health tracking
                self._record_success(provider_name)
                
                # Add routing metadata
                response.raw_response["routing_metadata"] = {
                    "provider": provider_name,
                    "task_type": task_type,
                    "was_fallback": provider_name != self.primary_provider
                }
                
                logger.info(f"Successfully completed request using {provider_name}")
                return response
                
            except (LLMRateLimitError, LLMAPIError, LLMTimeoutError) as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}, trying fallback")
                self._record_failure(provider_name, str(e))
                continue
            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider_name} failed with unexpected error: {e}")
                self._record_failure(provider_name, str(e))
                continue
        
        # All providers failed
        if last_error:
            raise last_error
        raise RuntimeError("No providers available")
    
    async def stream(
        self, 
        prompt: str, 
        task_type: str = "general",
        user_id: Optional[str] = None,
        **kwargs
    ) -> AsyncStreamGenerator:
        """
        Stream a prompt with intelligent routing.
        
        Args:
            prompt: The prompt to stream
            task_type: Type of task
            user_id: User ID for consistent routing
            **kwargs: Additional arguments
            
        Yields:
            TutorStreamChunk: Standardized stream chunks
        """
        # Select provider based on routing strategy or cost routing
        if user_id and self.routes:
            provider_name = self.get_provider(user_id)
            selected_providers = [provider_name]
        else:
            selected_providers = self._select_providers_for_task(task_type)
        
        for provider_name in selected_providers:
            try:
                if self.health_check_enabled and not await self._is_provider_healthy(provider_name):
                    continue
                
                provider = self.factory.get_provider(provider_name)
                async for chunk in provider.stream(prompt, **kwargs):
                    # Add routing metadata to first chunk
                    if not hasattr(chunk, 'routing_added'):
                        chunk.provider = provider_name
                        chunk.routing_added = True
                    
                    yield chunk
                
                self._record_success(provider_name)
                return
                
            except (LLMRateLimitError, LLMAPIError, LLMTimeoutError) as e:
                logger.warning(f"Provider {provider_name} streaming failed: {e}")
                self._record_failure(provider_name, str(e))
                continue
            except Exception as e:
                logger.error(f"Provider {provider_name} streaming failed: {e}")
                self._record_failure(provider_name, str(e))
                continue
        
        raise RuntimeError("No providers available for streaming")
    
    def _select_providers_for_task(self, task_type: str) -> List[str]:
        """
        Select providers based on task type and cost routing.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of provider names in order of preference
        """
        providers = []
        
        # Add primary provider first
        if self.primary_provider:
            providers.append(self.primary_provider)
        
        # Add fallback provider if different
        if self.fallback_provider and self.fallback_provider != self.primary_provider:
            providers.append(self.fallback_provider)
        
        # For cost-sensitive tasks, prefer cheaper providers
        if task_type in self.cost_thresholds:
            max_cost = self.cost_thresholds[task_type]
            # This could be extended to sort by known provider costs
            # For now, just use the configured order
        
        return providers
    
    async def _is_provider_healthy(self, provider_name: str) -> bool:
        """
        Check if a provider is healthy.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if healthy, False otherwise
        """
        import time
        
        # Check if we have recent health data
        last_check = self._last_health_check.get(provider_name, 0)
        if time.time() - last_check < 60:  # Cache for 1 minute
            health = self._provider_health.get(provider_name, {})
            return health.get("healthy", True)
        
        # Perform health check
        try:
            provider = self.factory.get_provider(provider_name)
            # Simple health check - try a minimal completion
            await provider.complete("test", max_tokens=1)
            self._record_success(provider_name)
            return True
        except Exception as e:
            self._record_failure(provider_name, str(e))
            return False
    
    def _record_success(self, provider_name: str):
        """Record a successful request for a provider."""
        import time
        
        if provider_name not in self._provider_health:
            self._provider_health[provider_name] = {
                "successes": 0,
                "failures": 0,
                "healthy": True
            }
        
        self._provider_health[provider_name]["successes"] += 1
        self._provider_health[provider_name]["healthy"] = True
        self._last_health_check[provider_name] = time.time()
    
    def _record_failure(self, provider_name: str, error: str):
        """Record a failed request for a provider."""
        import time
        
        if provider_name not in self._provider_health:
            self._provider_health[provider_name] = {
                "successes": 0,
                "failures": 0,
                "healthy": True
            }
        
        self._provider_health[provider_name]["failures"] += 1
        self._provider_health[provider_name]["last_error"] = error
        
        # Mark as unhealthy if too many failures
        if self._provider_health[provider_name]["failures"] > 3:
            self._provider_health[provider_name]["healthy"] = False
        
        self._last_health_check[provider_name] = time.time()
    
    def get_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all providers."""
        return self._provider_health.copy()
