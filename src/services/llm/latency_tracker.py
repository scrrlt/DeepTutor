"""
Latency Tracker - Measure execution time and log metrics.
"""

from contextlib import contextmanager
import logging
import time

logger = logging.getLogger(__name__)


@contextmanager
def measure_latency(operation_name: str, tags: dict = None):
    """Context manager to measure execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000  # ms
        tag_str = " ".join([f"{k}={v}" for k, v in (tags or {}).items()])
        logger.info(f"[METRIC] {operation_name} duration={duration:.2f}ms {tag_str}")


def log_metric(name: str, value: float, tags: dict = None):
    """Log a metric with optional tags."""
    tag_str = " ".join([f"{k}={v}" for k, v in (tags or {}).items()])
    logger.info(f"METRIC {name}={value} {tag_str}")
