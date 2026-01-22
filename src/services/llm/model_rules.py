"""Model-specific calling conventions.

This module intentionally keeps vendor-specific parameter decisions out of
configuration objects.
"""

from __future__ import annotations


def uses_max_completion_tokens(model: str) -> bool:
    """Return True if the model expects ``max_completion_tokens``.

    Args:
        model: Model identifier.

    Returns:
        True if the model uses ``max_completion_tokens``.
    """
    model_lower = model.lower()
    return model.startswith(("o1", "o3")) or model_lower.startswith("gpt-4o")


def get_token_limit_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    """Return the correct token-limit kwargs for the model.

    Args:
        model: Model identifier.
        max_tokens: The desired token limit.

    Returns:
        Dict with either ``max_tokens`` or ``max_completion_tokens``.
    """
    if uses_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}
