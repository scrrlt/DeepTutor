import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class TrafficController:
    """
    Manages concurrency limits for LLM providers.
    
    This implementation enforces a mandatory context manager pattern to ensure 
    slots are always released, even if an exception occurs during the LLM call.
    Manual acquire/release logic is kept private and synchronized with a 
    threading lock to prevent race conditions on the concurrency counter.
    """
    def __init__(self, max_concurrency: int = 10, provider_name: str = "global"):
        self.max_concurrency = max_concurrency
        self.provider_name = provider_name
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_requests = 0
        self._lock = threading.Lock()

    async def _acquire(self, priority: str = "normal"):
        """
        Private acquire logic with load shedding support.
        
        Args:
            priority: "normal" or "background". Background tasks are shed 
                     at 80% capacity to reserve slots for interactive users.
        """
        # Load shedding check before waiting on semaphore
        with self._lock:
            if priority == "background" and self._active_requests >= (self.max_concurrency * 0.8):
                logger.warning(f"[{self.provider_name}] Background request shed. Current: {self._active_requests}")
                raise asyncio.QueueFull(f"Capacity reached for background tasks on {self.provider_name}")

        await self._semaphore.acquire()
        with self._lock:
            self._active_requests += 1
        logger.debug(f"[{self.provider_name}] Slot acquired. Active: {self._active_requests}")

    def _release(self):
        """Private release logic with safety checks."""
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1
        
        self._semaphore.release()
        logger.debug(f"[{self.provider_name}] Slot released. Active: {self._active_requests}")

    async def __aenter__(self):
        """
        Entrance for the async context manager.
        Usage: 
            async with traffic_controller:
                response = await provider.complete(...)
        """
        await self._acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit for the async context manager.
        Guarantees release of the concurrency slot regardless of success or failure.
        """
        self._release()
        if exc_type:
            logger.warning(
                f"[{self.provider_name}] Request context exited with error: {exc_type.__name__}. "
                "Slot successfully recovered."
            )
        return False # Do not suppress the exception

    @property
    def current_concurrency(self) -> int:
        """Atomic read of the number of active requests."""
        with self._lock:
            return self._active_requests

    def should_allow_request(self, priority: str = "normal") -> bool:
        """Non-blocking check if a request would be accepted based on current load."""
        with self._lock:
            if priority == "background":
                return self._active_requests < (self.max_concurrency * 0.8)
            return self._active_requests < self.max_concurrency
