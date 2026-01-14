"""
Exponential Backoff Retry Logic

Implements retry strategy with exponential backoff, jitter, and error categorization.
Handles transient and permanent failures appropriately.
"""

import time
import logging
import random
from typing import Callable, Any, Optional, Dict, Tuple
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categorization for retry decisions."""
    TRANSIENT = "transient"  # Retry
    RATE_LIMIT = "rate_limit"  # Backoff then retry
    PERMANENT = "permanent"  # Don't retry
    VALIDATION = "validation"  # Don't retry
    UNKNOWN = "unknown"  # Treat as transient


class RetryConfig:
    """Retry configuration."""
    
    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1  # ±10%
    ):
        """
        Initialize retry config.
        
        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay in seconds (1s)
            max_delay: Max delay in seconds (60s)
            exponential_base: Exponential backoff base (2x)
            jitter: Add random jitter to delays
            jitter_range: Jitter range (±10%)
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt number.
        
        Formula: min(max_delay, initial_delay * base^attempt)
        With jitter: ±jitter_range%
        
        Args:
            attempt: Attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, ...
        delay = min(
            self.max_delay,
            self.initial_delay * (self.exponential_base ** attempt)
        )
        
        # Add jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay


def categorize_error(error: Exception, status_code: Optional[int] = None) -> ErrorCategory:
    """
    Categorize error for retry decision.
    
    Args:
        error: The exception
        status_code: HTTP status code if applicable
    
    Returns:
        ErrorCategory
    """
    # HTTP status codes
    if status_code:
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        elif status_code in [500, 502, 503, 504]:
            return ErrorCategory.TRANSIENT
        elif status_code in [400, 401, 403, 404]:
            return ErrorCategory.PERMANENT
    
    # Exception types
    error_str = str(error).lower()
    
    if "timeout" in error_str or "connection" in error_str:
        return ErrorCategory.TRANSIENT
    
    if "rate limit" in error_str or "quota" in error_str:
        return ErrorCategory.RATE_LIMIT
    
    if "validation" in error_str or "invalid" in error_str:
        return ErrorCategory.VALIDATION
    
    if "not found" in error_str or "404" in error_str:
        return ErrorCategory.PERMANENT
    
    # Default: treat as transient (safer)
    return ErrorCategory.TRANSIENT


class RetryableOperation:
    """Wrapper for retryable operations."""
    
    def __init__(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        error_categorizer: Optional[Callable] = None
    ):
        """
        Initialize retryable operation.
        
        Args:
            func: Function to retry
            config: Retry configuration
            error_categorizer: Custom error categorizer
        """
        self.func = func
        self.config = config or RetryConfig()
        self.error_categorizer = error_categorizer or categorize_error
        self.attempt_log: list = []
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Returns:
            Function result
        
        Raises:
            Exception if all attempts fail
        """
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.config.max_attempts}: {self.func.__name__}")
                
                result = self.func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Succeeded on attempt {attempt + 1}: {self.func.__name__}")
                
                self.attempt_log.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "timestamp": time.time()
                })
                
                return result
            
            except Exception as e:
                last_error = e
                
                # Categorize error
                category = self.error_categorizer(e)
                
                self.attempt_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "category": category.value,
                    "error": str(e),
                    "timestamp": time.time()
                })
                
                # Don't retry permanent errors
                if category == ErrorCategory.PERMANENT:
                    logger.error(f"Permanent error on attempt {attempt + 1}: {e}")
                    raise
                
                if category == ErrorCategory.VALIDATION:
                    logger.error(f"Validation error on attempt {attempt + 1}: {e}")
                    raise
                
                # Last attempt
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"Max attempts ({self.config.max_attempts}) exceeded: {e}")
                    raise
                
                # Calculate delay
                delay = self.config.get_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed ({category.value}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                time.sleep(delay)
        
        # Should not reach here
        if last_error:
            raise last_error


def retry(
    config: Optional[RetryConfig] = None,
    error_categorizer: Optional[Callable] = None
):
    """
    Decorator for retryable functions.
    
    Usage:
        @retry()
        def api_call():
            return requests.get(url)
    
    Args:
        config: Retry configuration
        error_categorizer: Custom error categorizer
    
    Returns:
        Decorated function
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation = RetryableOperation(func, config, error_categorizer)
            return operation.execute(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Simple retry decorator
    @retry(config=RetryConfig(max_attempts=3, initial_delay=0.5))
    def flaky_function(x):
        """Simulates a function that fails sometimes."""
        import random
        if random.random() < 0.7:
            raise ConnectionError("Temporary network error")
        return f"Success: {x}"
    
    # Example 2: Manual retry with custom config
    config = RetryConfig(
        max_attempts=5,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True
    )
    
    def api_call():
        """Simulates an API call."""
        raise ConnectionError("Network timeout")
    
    operation = RetryableOperation(api_call, config)
    
    print("Testing retry logic...")
    print("\nAttempt log:")
    for log in operation.attempt_log:
        print(f"  {log}")
    
    print("\nDelay progression:")
    for attempt in range(5):
        delay = config.get_delay(attempt)
        print(f"  Attempt {attempt + 1}: {delay:.2f}s delay")