"""
Traffic Controller - Load shedding and backpressure.
"""


class TrafficController:
    """Load shedder for high-priority requests."""

    def __init__(self, max_concurrent: int = 50):
        self.current = 0
        self.limit = max_concurrent

    def acquire(self, priority: str = "normal") -> bool:
        """Acquire a slot for processing. Returns True if allowed."""
        if priority == "background" and self.current >= (self.limit * 0.8):
            return False  # Shed load for background tasks
        if self.current >= self.limit:
            return False  # At limit
        self.current += 1
        return True

    def release(self):
        """Release a processing slot."""
        if self.current > 0:
            self.current -= 1

    def should_allow_request(self, priority: str = "normal") -> bool:
        """Check if request would be allowed (doesn't acquire)."""
        if priority == "background" and self.current >= (self.limit * 0.8):
            return False  # Shed load
        return self.current < self.limit
