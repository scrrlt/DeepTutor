# -*- coding: utf-8 -*-
"""Chat template resolution and rendering for local LLMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

ENV_CHAT_TEMPLATE = "LOCAL_LLM_CHAT_TEMPLATE"
ENV_CHAT_TEMPLATE_PATH = "LOCAL_LLM_CHAT_TEMPLATE_PATH"
ENV_TOKENIZER_DIR = "LOCAL_LLM_TOKENIZER_DIR"
ENV_MODEL_DIR = "LOCAL_LLM_MODEL_DIR"
ENV_TEMPLATE_DIR = "LOCAL_LLM_TEMPLATE_DIR"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "config" / "chat_templates"


@dataclass(frozen=True)
class ChatTemplateInfo:
    """Resolved chat template details."""

    template: str
    metadata: dict[str, Any]
    source: str


def resolve_chat_template(
    model: str | None,
    template: str | None = None,
    template_path: str | None = None,
    tokenizer_dir: str | None = None,
    model_dir: str | None = None,
    template_dir: str | None = None,
) -> ChatTemplateInfo | None:
    """
    Resolve a chat template from overrides, tokenizer metadata, or registry.

    Args:
        model: Model name used for registry lookup.
        template: Explicit template override.
        template_path: Path to a template file.
        tokenizer_dir: Directory containing tokenizer metadata.
        model_dir: Directory containing model metadata.
        template_dir: Directory containing registry templates.

    Returns:
        ChatTemplateInfo if found, otherwise None.

    Raises:
        None.
    """
    if template:
        return ChatTemplateInfo(template=template, metadata={}, source="override")

    template_path_value = template_path or os.getenv(ENV_CHAT_TEMPLATE_PATH)
    if template_path_value:
        path = Path(template_path_value)
        if path.exists():
            return ChatTemplateInfo(
                template=path.read_text(encoding="utf-8"),
                metadata={},
                source=str(path),
            )

    direct_template = os.getenv(ENV_CHAT_TEMPLATE)
    if direct_template:
        return ChatTemplateInfo(
            template=direct_template,
            metadata={},
            source=ENV_CHAT_TEMPLATE,
        )

    tokenizer_path = tokenizer_dir or model_dir or os.getenv(ENV_TOKENIZER_DIR)
    tokenizer_path = tokenizer_path or os.getenv(ENV_MODEL_DIR)
    if tokenizer_path:
        metadata_info = _load_template_from_tokenizer_dir(Path(tokenizer_path))
        if metadata_info:
            return metadata_info

    registry_dir = template_dir or os.getenv(ENV_TEMPLATE_DIR)
    registry_root = Path(registry_dir) if registry_dir else DEFAULT_TEMPLATE_DIR
    if model and registry_root.exists():
        registry_info = _load_template_from_registry(registry_root, model)
        if registry_info:
            return registry_info

    return None


def render_chat_template(
    messages: list[dict[str, str]],
    template_info: ChatTemplateInfo,
    add_generation_prompt: bool = True,
) -> str:
    """
    Render a chat template using the provided messages.

    Args:
        messages: List of role/content message dictionaries.
        template_info: Template metadata to render.
        add_generation_prompt: Whether to append the generation prompt.

    Returns:
        Rendered prompt string.

    Raises:
        None.
    """
    metadata = _normalize_metadata(template_info.metadata)
    context: dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        **metadata,
    }

    rendered = _render_template(template_info.template, context)
    return rendered


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    """
    Render a Jinja-style template with minja or fallback renderer.

    Args:
        template: Template string.
        context: Context variables for rendering.

    Returns:
        Rendered string.

    Raises:
        Exception: If template rendering fails.
    """
    try:
        import minja  # type: ignore

        template_class = getattr(minja, "Template", None)
        if template_class is not None:
            return template_class(template).render(context)

        render_fn = getattr(minja, "render", None)
        if callable(render_fn):
            try:
                return str(render_fn(template, context))
            except TypeError:
                return str(render_fn(template, **context))
    except Exception as exc:  # pragma: no cover - fallback path
        logger.debug("minja render failed, falling back: %s", exc)

    from jinja2 import BaseLoader, Environment

    # Disable autoescape for chat templates - these contain system prompts and user messages
    # that need to preserve formatting like newlines and special characters.
    # Input validation is handled at the API layer before reaching this function.
    env = Environment(loader=BaseLoader(), autoescape=True)  # Enable autoescaping for security
    return env.from_string(template).render(**context)


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize tokenizer metadata for template rendering.

    Args:
        metadata: Raw metadata mapping.

    Returns:
        Normalized metadata with safe token defaults.

    Raises:
        None.
    """
    normalized = dict(metadata)
    for token_key in ("bos_token", "eos_token", "eot_token", "unk_token", "pad_token"):
        value = normalized.get(token_key)
        if value is None:
            normalized[token_key] = ""
    return normalized


def _load_template_from_tokenizer_dir(directory: Path) -> ChatTemplateInfo | None:
    """
    Load a chat template from tokenizer metadata files.

    Args:
        directory: Directory containing tokenizer metadata.

    Returns:
        ChatTemplateInfo if a template is found.

    Raises:
        None.
    """
    if not directory.exists():
        return None

    config_path = directory / "tokenizer_config.json"
    tokenizer_path = directory / "tokenizer.json"

    config_data = _load_json_file(config_path)
    tokenizer_data = _load_json_file(tokenizer_path)

    template = None
    metadata: dict[str, Any] = {}

    for source, data in ((config_path, config_data), (tokenizer_path, tokenizer_data)):
        if not data:
            continue
        if template is None:
            template = data.get("chat_template")
        metadata.update(_extract_tokens(data))
        if template:
            return ChatTemplateInfo(
                template=template,
                metadata=metadata,
                source=str(source),
            )

    return None


def _load_template_from_registry(
    template_dir: Path,
    model: str,
) -> ChatTemplateInfo | None:
    """
    Load a template from the registry directory.

    Args:
        template_dir: Directory containing template files.
        model: Model name for lookup.

    Returns:
        ChatTemplateInfo if a template is found.

    Raises:
        None.
    """
    model_key = model.split("/")[-1].lower()
    candidates = {
        model_key,
        model_key.split("-")[0],
        model_key.split("_")[0],
    }

    for candidate in candidates:
        for extension in (".jinja", ".j2", ".tmpl"):
            path = template_dir / f"{candidate}{extension}"
            if path.exists():
                return ChatTemplateInfo(
                    template=path.read_text(encoding="utf-8"),
                    metadata={},
                    source=str(path),
                )

    return None


def _extract_tokens(data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Extract special tokens from tokenizer metadata.

    Args:
        data: Tokenizer metadata mapping.

    Returns:
        Dictionary of token values.

    Raises:
        None.
    """
    tokens: dict[str, Any] = {}
    for key in ("bos_token", "eos_token", "eot_token", "unk_token", "pad_token"):
        if key in data:
            tokens[key] = data[key]

    special_tokens = data.get("special_tokens_map")
    if isinstance(special_tokens, Mapping):
        for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
            if key not in tokens and key in special_tokens:
                tokens[key] = special_tokens[key]

    return tokens


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """
    Load a JSON file if it exists.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON mapping or None.

    Raises:
        None.
    """
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON at %s: %s", path, exc)
        return None


__all__ = [
    "ChatTemplateInfo",
    "render_chat_template",
    "resolve_chat_template",
]
