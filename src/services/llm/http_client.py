"""Shared async HTTP client for LLM providers."""

import asyncio
from typing import Optional

import httpx

from src.logging import get_logger

logger = get_logger(__name__)


class HTTPClientManager:
    """
    Singleton manager for the shared HTTPX client.
    Ensures connection pooling and correct event loop binding.
    """

    _instance: Optional["HTTPClientManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @classmethod
    async def get_instance(cls) -> "HTTPClientManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_client(self) -> httpx.AsyncClient:
        """Returns the active client, initializing if necessary."""
        if self._client is None or self._client.is_closed:
            logger.info("Initializing new shared HTTPX client")
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
                http2=True,
                # INCREASED TIMEOUT: 30s is fatal for reasoning models.
                # connect=10s (fail fast on net down)
                # read=120s (wait for long generation)
                timeout=httpx.Timeout(120.0, connect=10.0),
                follow_redirects=True,
                event_hooks={"response": [self._log_error_responses]},
            )
        return self._client

    async def close(self) -> None:
        """Graceful shutdown."""
        if self._client and not self._client.is_closed:
            logger.info("Closing shared HTTPX client")
            await self._client.aclose()
            self._client = None

    async def _log_error_responses(self, response: httpx.Response) -> None:
        """Network-layer observability for failed requests."""
        if response.status_code >= 400:
            await response.aread()  # Ensure we have body for logging
            logger.warning(
                f"HTTP {response.status_code} from {response.url}: "
                f"{response.text[:200]}"  # Truncate for sanity
            )


# Public Accessors (Facade)
_manager = HTTPClientManager()


async def get_shared_http_client() -> httpx.AsyncClient:
    # Ensure manager is initialized safely
    manager = await HTTPClientManager.get_instance()
    return await manager.get_client()


async def close_shared_http_client() -> None:
    if HTTPClientManager._instance:
        await HTTPClientManager._instance.close()
