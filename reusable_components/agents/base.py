"""Base class shared by Chimera style research agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentContext:
    company: str
    topic: str
    snapshot: Dict[str, Any]


class Agent(ABC):
    code: str
    name: str
    description: str

    @abstractmethod
    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Produce agent output for the given context."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
        }