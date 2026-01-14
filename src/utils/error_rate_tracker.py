"""
Error Rate Tracker - Monitor service health and error rates.
"""

from collections import deque
from typing import Deque


class ErrorRateTracker:
    """Track the error rate over a sliding window of the last N outcomes."""

    def __init__(self, window: int = 100) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        self.win: Deque[bool] = deque(maxlen=window)

    def record(self, success: bool) -> None:
        """Record a success or failure outcome."""
        self.win.append(bool(success))

    @property
    def rate(self) -> float:
        """Get current error rate (0.0 to 1.0)."""
        n = len(self.win)
        if n == 0:
            return 0.0
        errs = sum(1 for s in self.win if not s)
        return errs / n

    @property
    def success_rate(self) -> float:
        """Get current success rate (0.0 to 1.0)."""
        return 1.0 - self.rate

    def is_healthy(self, threshold: float = 0.1) -> bool:
        """Check if error rate is below threshold."""
        return self.rate <= threshold
