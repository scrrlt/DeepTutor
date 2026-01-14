"""Daily token & cost budget enforcement utility (simple file-based prototype).

This is a local prototype; in cloud, replace storage with GCS or Firestore.

Usage:
    from config.budget_enforcer import BudgetEnforcer
    be = BudgetEnforcer(daily_token_limit=50000)
    be.consume(estimated_tokens)
    if be.blocked: ...
"""
from __future__ import annotations
import json, time, os, threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

STATE_FILE = Path(os.getenv("BUDGET_STATE_FILE", "budget_state.json"))
RESET_HOUR_UTC = 0  # midnight UTC
_lock = threading.Lock()

@dataclass
class BudgetState:
    day: str
    tokens_used: int

class BudgetEnforcer:
    def __init__(self, daily_token_limit: int, warn_ratio: float = 0.8, block_ratio: float = 0.95):
        self.daily_token_limit = daily_token_limit
        self.warn_threshold = int(daily_token_limit * warn_ratio)
        self.block_threshold = int(daily_token_limit * block_ratio)
        self.state = self._load()
        self.blocked = False

    def _load(self) -> BudgetState:
        now_day = time.strftime('%Y-%m-%d')
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
                if data.get('day') == now_day:
                    return BudgetState(day=now_day, tokens_used=int(data.get('tokens_used', 0)))
            except Exception:  # noqa: BLE001
                pass
        return BudgetState(day=now_day, tokens_used=0)

    def _persist(self) -> None:
        tmp = STATE_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state.__dict__), encoding='utf-8')
        tmp.replace(STATE_FILE)

    def _rollover_if_needed(self) -> None:
        now_day = time.strftime('%Y-%m-%d')
        if self.state.day != now_day:
            self.state = BudgetState(day=now_day, tokens_used=0)
            self.blocked = False

    def consume(self, tokens: int) -> None:
        with _lock:
            self._rollover_if_needed()
            if self.state.tokens_used >= self.block_threshold:
                self.blocked = True
                return
            self.state.tokens_used += tokens
            self._persist()
            if self.state.tokens_used >= self.block_threshold:
                self.blocked = True

    def status(self) -> dict:
        return {
            "day": self.state.day,
            "tokens_used": self.state.tokens_used,
            "limit": self.daily_token_limit,
            "warn_threshold": self.warn_threshold,
            "block_threshold": self.block_threshold,
            "blocked": self.blocked,
        }

__all__ = ["BudgetEnforcer"]