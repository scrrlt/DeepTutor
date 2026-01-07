# tests/performance/oom_repro/mock_flush_adapter.py
"""
Deterministic mock flush adapter for CI and local smoke tests.

Usage:
    from tests.performance.oom_repro.mock_flush_adapter import MockFlushAdapter
    adapter = MockFlushAdapter(latency_ms=100, mem_delta_mb=0, fail_rate=0.0)
    await adapter.flush(chunks)
"""

import time
import random
import asyncio
import logging

logger = logging.getLogger("MockFlushAdapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MockFlushAdapter:
    def __init__(self, latency_ms: int = 100, mem_delta_mb: int = 0, fail_rate: float = 0.0):
        """
        latency_ms: simulated processing latency per flush (ms)
        mem_delta_mb: temporary synthetic allocation to emulate native pressure (MB)
        fail_rate: probability [0.0, 1.0] to simulate a transient failure
        """
        self.latency_ms = int(latency_ms)
        self.mem_delta_mb = int(mem_delta_mb)
        self.fail_rate = float(fail_rate)

    async def flush(self, batch):
        start = time.time()
        # Simulate CPU-bound work (non-blocking sleep for async)
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Optional synthetic memory pressure (allocated and released immediately)
        if self.mem_delta_mb > 0:
            _tmp = bytearray(self.mem_delta_mb * 1024 * 1024)
            del _tmp

        # Optional failure injection
        if self.fail_rate > 0.0 and random.random() < self.fail_rate:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.warning("MockFlushAdapter injected failure")
            return {"success": False, "persisted": 0, "latency_ms": elapsed_ms, "error": "injected-failure"}

        persisted = len(batch)
        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(f"MockFlushAdapter flushed {persisted} items in {elapsed_ms}ms")
        return {"success": True, "persisted": persisted, "latency_ms": elapsed_ms}