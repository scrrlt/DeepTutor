"""
Shared HTTP Client for LLM Providers
====================================

Provides a singleton httpx.AsyncClient instance for all LLM providers to
share, enabling connection pooling, Keep-Alive, and reducing SSL handshake
overhead.

This addresses the memory and dependency bloat from having multiple HTTP
clients (requests, aiohttp, httpx) by standardizing on httpx which supports:
- Async/await natively
- Full type hints
- HTTP/2 support
- Connection pooling
- Keep-Alive connections
"""

import asyncio

import httpx

_shared_client: httpx.AsyncClient | None = None
_shared_client_lock = asyncio.Lock()


async def get_shared_http_client() -> httpx.AsyncClient:
    """
    Get or create the shared httpx.AsyncClient instance.

    This client is configured with:
    - Connection pooling (max 100 connections, 20 keepalive connections)
    - HTTP/2 support
    - 30 second timeout
    - Keep-Alive connections

    Returns:
        httpx.AsyncClient: Shared HTTP client instance
    """
    global _shared_client

    async with _shared_client_lock:
        if _shared_client is None:
            _shared_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
                http2=True,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )

    return _shared_client


async def close_shared_http_client() -> None:
    """
    Close the shared HTTP client.

    Should be called during application shutdown to ensure all connections
    are properly closed.
    """
    global _shared_client

    client_to_close: httpx.AsyncClient | None = None
    async with _shared_client_lock:
        if _shared_client is not None:
            client_to_close = _shared_client
            _shared_client = None

    if client_to_close is not None:
        await client_to_close.aclose()
