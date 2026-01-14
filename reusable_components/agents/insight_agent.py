"""Sample agent focused on strategic insights."""
from __future__ import annotations

from typing import Any, Dict

from .base import Agent, AgentContext


class InsightAgent(Agent):
    code = "INS"
    name = "Strategic Insight"
    description = "Summarises recent signals into strategic recommendations."

    def run(self, context: AgentContext) -> Dict[str, Any]:
        sources = context.snapshot.get("records", [])
        highlights = [s.get("payload", {}).get("headline", "") for s in sources]
        return {
            "summary": f"{context.company} should accelerate {context.topic} initiatives.",
            "highlights": [h for h in highlights if h],
        }