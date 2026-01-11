"""
LLM Telemetry - Track latency, success rate, and costs by provider.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from ..types import TutorResponse, TutorStreamChunk

# Legacy exports for backward compatibility
from .latency_tracker import log_metric, measure_latency
from .tracing import get_trace_id, start_trace

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    last_request_time: Optional[float] = None
    last_error: Optional[str] = None
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 requests
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request in USD."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests
    
    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.recent_latencies:
            return 0.0
        sorted_latencies = sorted(self.recent_latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


class LLMTelemetry:
    """
    Telemetry tracking for LLM providers.
    
    Tracks:
    - Request latency and success rates
    - Token usage and costs
    - Error patterns
    - Provider health over time
    """
    
    def __init__(self):
        """Initialize telemetry tracker."""
        self._metrics: Dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self._lock = threading.Lock()
        
        # Global metrics
        self.global_start_time = time.time()
        self.total_requests = 0
        self.total_cost_usd = 0.0
    
    def track_request_start(self, provider: str) -> float:
        """
        Track the start of a request.
        
        Args:
            provider: Provider name
            
        Returns:
            Request start timestamp
        """
        return time.time()
    
    def track_request_success(
        self, 
        provider: str, 
        start_time: float, 
        response: TutorResponse
    ) -> None:
        """
        Track a successful request.
        
        Args:
            provider: Provider name
            start_time: Request start timestamp
            response: Response from provider
        """
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        with self._lock:
            metrics = self._metrics[provider]
            
            # Update basic counters
            metrics.total_requests += 1
            metrics.successful_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.last_request_time = end_time
            
            # Update latency tracking
            metrics.recent_latencies.append(latency_ms)
            
            # Update token usage
            usage = response.usage or {}
            for token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                if token_type in usage:
                    key = token_type.replace("_tokens", "")
                    metrics.total_tokens[key] += usage[token_type]
            
            # Update cost
            if response.cost_estimate:
                metrics.total_cost_usd += response.cost_estimate
                self.total_cost_usd += response.cost_estimate
            
            # Update global counters
            self.total_requests += 1
    
    def track_request_failure(
        self, 
        provider: str, 
        start_time: float, 
        error: Exception
    ) -> None:
        """
        Track a failed request.
        
        Args:
            provider: Provider name
            start_time: Request start timestamp
            error: The error that occurred
        """
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        error_type = type(error).__name__
        error_message = str(error)
        
        with self._lock:
            metrics = self._metrics[provider]
            
            # Update basic counters
            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.last_request_time = end_time
            metrics.last_error = error_message
            
            # Update error tracking
            metrics.error_counts[error_type] += 1
            
            # Update global counters
            self.total_requests += 1
    
    def track_stream_chunk(
        self, 
        provider: str, 
        chunk: TutorStreamChunk
    ) -> None:
        """
        Track a streaming chunk.
        
        Args:
            provider: Provider name
            chunk: Stream chunk
        """
        # For streaming, we primarily track completion and token usage
        # The actual timing is tracked at the stream level
        if chunk.is_complete and chunk.usage:
            with self._lock:
                metrics = self._metrics[provider]
                usage = chunk.usage or {}
                for token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    if token_type in usage:
                        key = token_type.replace("_tokens", "")
                        metrics.total_tokens[key] += usage[token_type]
    
    def get_provider_metrics(self, provider: str) -> Optional[ProviderMetrics]:
        """
        Get metrics for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider metrics or None if not found
        """
        with self._lock:
            return self._metrics.get(provider)
    
    def get_all_metrics(self) -> Dict[str, ProviderMetrics]:
        """
        Get metrics for all providers.
        
        Returns:
            Dictionary mapping provider names to metrics
        """
        with self._lock:
            return dict(self._metrics)
    
    def get_provider_ranking(self, metric: str = "success_rate") -> List[tuple]:
        """
        Get providers ranked by a specific metric.
        
        Args:
            metric: Metric to rank by ('success_rate', 'latency', 'cost', 'throughput')
            
        Returns:
            List of (provider, value) tuples sorted by the metric
        """
        with self._lock:
            rankings = []
            
            for provider, metrics in self._metrics.items():
                if metric == "success_rate":
                    value = metrics.success_rate
                elif metric == "latency":
                    value = metrics.average_latency_ms
                elif metric == "cost":
                    value = metrics.average_cost_per_request
                elif metric == "throughput":
                    # Requests per second
                    if metrics.last_request_time and self.global_start_time:
                        elapsed = metrics.last_request_time - self.global_start_time
                        value = metrics.total_requests / elapsed if elapsed > 0 else 0
                    else:
                        value = 0
                else:
                    continue
                
                rankings.append((provider, value))
            
            # Sort by value (lower is better for latency and cost, higher for success rate and throughput)
            reverse_order = metric in ["success_rate", "throughput"]
            rankings.sort(key=lambda x: x[1], reverse=reverse_order)
            
            return rankings
    
    def get_health_status(self) -> Dict[str, str]:
        """
        Get health status for all providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        with self._lock:
            status = {}
            
            for provider, metrics in self._metrics.items():
                if metrics.success_rate >= 95 and metrics.average_latency_ms < 5000:
                    status[provider] = "healthy"
                elif metrics.success_rate >= 80 and metrics.average_latency_ms < 10000:
                    status[provider] = "degraded"
                else:
                    status[provider] = "unhealthy"
            
            return status
    
    def reset_metrics(self, provider: Optional[str] = None) -> None:
        """
        Reset metrics for a provider or all providers.
        
        Args:
            provider: Provider to reset, or None to reset all
        """
        with self._lock:
            if provider:
                if provider in self._metrics:
                    self._metrics[provider] = ProviderMetrics()
            else:
                self._metrics.clear()
                self.global_start_time = time.time()
                self.total_requests = 0
                self.total_cost_usd = 0.0
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics in a serializable format.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            export_data = {
                "global": {
                    "total_requests": self.total_requests,
                    "total_cost_usd": self.total_cost_usd,
                    "uptime_seconds": time.time() - self.global_start_time
                },
                "providers": {}
            }
            
            for provider, metrics in self._metrics.items():
                export_data["providers"][provider] = {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.success_rate,
                    "average_latency_ms": metrics.average_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "total_cost_usd": metrics.total_cost_usd,
                    "average_cost_per_request": metrics.average_cost_per_request,
                    "total_tokens": metrics.total_tokens,
                    "last_request_time": metrics.last_request_time,
                    "last_error": metrics.last_error,
                    "error_counts": dict(metrics.error_counts)
                }
            
            return export_data


# Global telemetry instance
_telemetry = None


def get_telemetry() -> LLMTelemetry:
    """Get or create the global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = LLMTelemetry()
    return _telemetry


# Decorator for automatic telemetry tracking
def track_llm_call(provider_name: str):
    """
    Decorator to automatically track LLM calls.
    
    Args:
        provider_name: Name of the provider
        
    Usage:
        @track_llm_call("openai")
        async def complete(prompt: str, **kwargs) -> TutorResponse:
            # Implementation
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            start_time = telemetry.track_request_start(provider_name)
            
            try:
                result = await func(*args, **kwargs)
                if isinstance(result, TutorResponse):
                    telemetry.track_request_success(provider_name, start_time, result)
                return result
            except Exception as e:
                telemetry.track_request_failure(provider_name, start_time, e)
                raise
        
        return wrapper
    return decorator


__all__ = [
    "measure_latency", 
    "log_metric", 
    "start_trace", 
    "get_trace_id",
    "LLMTelemetry",
    "ProviderMetrics",
    "get_telemetry",
    "track_llm_call"
]
