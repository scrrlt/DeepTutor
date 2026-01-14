"""
Error messages for embedding adapters.
"""

from typing import Dict

EMBEDDING_ERRORS: Dict[str, str] = {
    "invalid_parameters": (
        "Embedding request failed: Invalid parameters. "
        "Please check your model name '{model}' and ensure it exists. "
        "Response: {error_text}"
    ),
    "auth_error": (
        "Embedding request failed: Authentication error. "
        "Please check your API key for {base_url}."
    ),
    "model_not_found": (
        "Embedding request failed: Model '{model}' not found. "
        "Please verify the model name and ensure it's available."
    ),
    "rate_limit": (
        "Embedding request failed: Rate limit exceeded. "
        "Please try again later."
    ),
    "empty_embedding": (
        "Embedding request failed: Empty embedding returned. "
        "Model: '{model}'. Base URL: {base_url}. "
        "If you are using a local OpenAI-compatible server (e.g. LM Studio), "
        "ensure the embeddings endpoint is reachable (often /v1/embeddings) and "
        "try disabling custom dimensions if the server doesn't support them."
    ),
}


def format_error_message(error_type: str, **kwargs) -> str:
    """
    Format an error message with provided parameters.

    Args:
        error_type: Type of error (key in EMBEDDING_ERRORS)
        **kwargs: Parameters to format into the message

    Returns:
        Formatted error message
    """

    template = EMBEDDING_ERRORS.get(error_type, "Unknown error occurred")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # If formatting fails due to missing keys, return a fallback message
        return f"{template} (Missing parameter: {e})"
