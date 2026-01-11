"""
Traffic Controller - Load shedding and backpressure.
"""



class TrafficController:
    """Load shedder for high-priority requests."""

    def __init__(self, max_concurrent: int = 50):
        self.current = 0
        self.limit = max_concurrent

    def should_allow_request(self, priority: str = "normal") -> bool:
        """Allow request based on priority and load."""
        if priority == "background" and self.current > (self.limit * 0.8):
            return False  # Shed load
        return self.current < self.limit
