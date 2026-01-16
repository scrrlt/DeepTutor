"""LLM client interface.

Legacy client wrapper around the LLM factory functions.
"""

from typing import Any, Awaitable, Callable

from src.logging import get_logger

from .capabilities import system_in_messages
from .config import LLMConfig, get_llm_config


class LLMClient:
    """
    Unified LLM client for all services.

    Wraps the LLM Factory with a class-based interface.
    Prefer using factory functions (complete, stream) directly for new code.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize LLM client."""
        self.config = config or get_llm_config()
        self.logger = get_logger("LLMClient")

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Call LLM completion via Factory."""
        from . import factory

        return await factory.complete(
            prompt=prompt,
            system_prompt=system_prompt or "You are a helpful assistant.",
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=getattr(self.config, "api_version", None),
            binding=getattr(self.config, "binding", "openai"),
            history_messages=history,
            **kwargs,
        )

    def complete_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Run completion synchronously for non-async contexts."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.complete(prompt, system_prompt, history, **kwargs)
            )

        raise RuntimeError(
            "LLMClient.complete_sync() cannot be called from a running event loop. "
            "Use `await llm.complete(...)` instead."
        )

    def get_model_func(self) -> Callable[..., Awaitable[str]]:
        """
        Get a function compatible with LightRAG's llm_model_func parameter.
        """

        binding = getattr(self.config, "binding", "openai")
        uses_openai_style = system_in_messages(binding, self.config.model)

        from . import factory

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            messages: list[dict[str, Any]] | None = None
            if uses_openai_style:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

            return await factory.complete(
                prompt="" if messages else prompt,
                system_prompt=system_prompt or "You are a helpful assistant.",
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                api_version=getattr(self.config, "api_version", None),
                binding=binding,
                messages=messages,
                **kwargs,
            )

        return llm_model_func

    def get_vision_model_func(self) -> Callable[..., Any]:
        """
        Get a function compatible with RAG-Anything's vision_model_func parameter.
        """

        binding = getattr(self.config, "binding", "openai")
        uses_openai_style = system_in_messages(binding, self.config.model)

        if not uses_openai_style:
            from . import factory

            def vision_model_func_via_factory(
                prompt: str,
                system_prompt: str | None = None,
                history_messages: list[dict[str, Any]] | None = None,
                image_data: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> Awaitable[str]:
                return factory.complete(
                    prompt=prompt,
                    system_prompt=system_prompt or "You are a helpful assistant.",
                    model=self.config.model,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    binding=binding,
                    messages=messages,
                    history_messages=history_messages,
                    image_data=image_data,
                    **kwargs,
                )

            return vision_model_func_via_factory

        from lightrag.llm.openai import openai_complete_if_cache

        api_version = getattr(self.config, "api_version", None)

        def vision_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            image_data: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            if messages:
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "messages",
                        "prompt",
                        "system_prompt",
                        "history_messages",
                    ]
                }
                lightrag_kwargs = {
                    "messages": messages,
                    "api_key": self.config.api_key,
                    "base_url": self.config.base_url,
                    **clean_kwargs,
                }
                if api_version:
                    lightrag_kwargs["api_version"] = api_version
                return openai_complete_if_cache(
                    self.config.model,
                    prompt="",
                    **lightrag_kwargs,
                )

            if image_data:
                image_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }
                lightrag_kwargs = {
                    "messages": [image_message],
                    "api_key": self.config.api_key,
                    "base_url": self.config.base_url,
                    **kwargs,
                }
                if api_version:
                    lightrag_kwargs["api_version"] = api_version
                return openai_complete_if_cache(
                    self.config.model,
                    prompt="",
                    **lightrag_kwargs,
                )

            lightrag_kwargs = {
                "system_prompt": system_prompt,
                "history_messages": history_messages or [],
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                **kwargs,
            }
            if api_version:
                lightrag_kwargs["api_version"] = api_version
            return openai_complete_if_cache(
                self.config.model,
                prompt,
                **lightrag_kwargs,
            )

        return vision_model_func


_client: LLMClient | None = None


def get_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Get or create the singleton LLM client."""

    global _client
    if _client is None:
        _client = LLMClient(config)
    return _client


def reset_llm_client() -> None:
    """Reset the singleton LLM client."""

    global _client
    _client = None
