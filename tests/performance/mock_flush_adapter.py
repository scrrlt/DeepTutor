#!/usr/bin/env python3
# tests/performance/oom_repro/mock_flush_adapter.py
"""
Deterministic mock flush adapter for CI and local smoke tests.

Purpose:
- Provide a predictable, side‑effect‑free stand‑in for a real vector store adapter.
- Allow controlled latency, memory pressure, and failure injection.
- Be safe to use inside multiprocessing workers and asyncio event loops.

Usage:
    from tests.performance.oom_repro.mock_flush_adapter import MockFlushAdapter
    adapter = MockFlushAdapter(latency_ms=100, mem_delta_mb=0, fail_rate=0.0)
    await adapter.flush(chunks)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MockFlushAdapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MockFlushAdapter:
    """
    Deterministic async flush adapter.

    Design principles:
    - No external I/O.
    - No retained references to batch data.
    - Optional, bounded synthetic memory pressure.
    - Optional, reproducible failure injection.
    """

    def __init__(
        self,
        latency_ms: int = 100,
        mem_delta_mb: int = 0,
        fail_rate: float = 0.0,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            latency_ms: Simulated processing latency per flush (milliseconds).
            mem_delta_mb: Temporary synthetic allocation to emulate native pressure (MB).
                          Allocation is immediately released.
            fail_rate: Probability [0.0, 1.0] to simulate a transient failure.
            seed: Optional RNG seed for deterministic failure behavior in CI.
        """
        self.latency_ms = max(0, int(latency_ms))
        self.mem_delta_mb = max(0, int(mem_delta_mb))
        self.fail_rate = min(max(float(fail_rate), 0.0), 1.0)

        # Deterministic RNG for CI reproducibility
        self._rng = random.Random(seed)

    async def flush(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate flushing a batch to a vector store.

        Returns a dict compatible with KnowledgeBaseManager expectations:
            {
              "success": bool,
              "persisted": int,
              "latency_ms": int,
              "error": Optional[str]
            }
        """
        start = time.time()

        # Simulated async latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Optional synthetic memory pressure (bounded, temporary)
        if self.mem_delta_mb > 0:
            try:
                _tmp = bytearray(self.mem_delta_mb * 1024 * 1024)
            finally:
                # Explicitly drop reference immediately
                del _tmp

        # Optional deterministic failure injection
        if self.fail_rate > 0.0 and self._rng.random() < self.fail_rate:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.warning("MockFlushAdapter injected failure")
            return {
                "success": False,
                "persisted": 0,
                "latency_ms": elapsed_ms,
                "error": "injected-failure",
            }

        persisted = len(batch)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(f"MockFlushAdapter flushed {persisted} items in {elapsed_ms}ms")

        return {
            "success": True,
            "persisted": persisted,
            "latency_ms": elapsed_ms,
        }
