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
    Manual acquire/release methods are kept private to prevent state leaks.
    """

    def __init__(self, max_concurrency: int = 10, provider_name: str = "global"):
        self.max_concurrency = max_concurrency
        self.provider_name = provider_name
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_requests = 0
        self._lock = threading.Lock()

    async def _acquire(self):
        """Private acquire logic."""
        await self._semaphore.acquire()
        with self._lock:
            self._active_requests += 1
        logger.debug(f"[{self.provider_name}] Slot acquired. Active: {self._active_requests}")

    def _release(self):
        """Private release logic."""
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
        return False  # Do not suppress the exception

    @property
    def current_concurrency(self) -> int:
        """Returns the number of active requests currently held by the controller."""
        with self._lock:
            return self._active_requests
