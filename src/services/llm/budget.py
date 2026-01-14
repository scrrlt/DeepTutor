"""
Cost & Budgeting - Calculate costs and enforce budgets.
"""

from typing import Dict


class CostCalculator:
    """Calculate token costs accurately."""

    PRICING = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    }

    def calculate_cost(self, usage: Dict, model: str) -> float:
        """Calculate cost for usage."""
        if model not in self.PRICING:
            return 0.0

        pricing = self.PRICING[model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens * pricing["input"] / 1_000_000) + (
            output_tokens * pricing["output"] / 1_000_000
        )

    def check_budget(self, user_id: str, estimated_tokens: int) -> bool:
        """Check if user can afford the request."""
        # Placeholder - integrate with BudgetEnforcer logic
        return True
