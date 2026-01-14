"""
Idempotency Manager - Prevent duplicate requests.
"""

import hashlib
from typing import Set


class IdempotencyManager:
    """Manager for request idempotency to prevent duplicates."""

    def __init__(self):
        self._locks: Set[str] = set()

    def get_key(self, user_id: str, prompt: str) -> str:
        """Generate idempotency key from user and prompt."""
        return hashlib.sha256(f"{user_id}:{prompt}".encode()).hexdigest()

    def lock(self, key: str) -> bool:
        """Attempt to acquire lock for key. Returns True if successful."""
        if key in self._locks:
            return False
        self._locks.add(key)
        return True

    def unlock(self, key: str):
        """Release lock for key."""
        self._locks.discard(key)
