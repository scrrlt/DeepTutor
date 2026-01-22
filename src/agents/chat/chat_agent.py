"""ChatAgent - Orchestrator for conversational flows.

This agent is intentionally lightweight:

- Delegates LLM calls to BaseAgent (provider-backed)
- Accepts optional injected tool functions for RAG/Web
- Uses a model-agnostic history truncation heuristic
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

from src.agents.base_agent import BaseAgent
from src.config.config import LLMConfig


class ChatAgent(BaseAgent):
    """Conversational agent with optional RAG/Web context."""

    DEFAULT_MAX_HISTORY_TOKENS = 4000

    def __init__(
        self,
        language: str = "zh",
        config: LLMConfig | None = None,
        max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
        rag_search_func: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
        web_search_func: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__(
            module_name="chat",
            agent_name="chat_agent",
            language=language,
            config=config,
        )
        self.max_history_tokens = max_history_tokens
        self._rag_search = rag_search_func
        self._web_search = web_search_func

    def _truncate_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Truncate history using a conservative token heuristic."""
        if not history:
            return []

        estimated_tokens = 0
        truncated: list[dict[str, str]] = []

        for msg in reversed(history):
            content_len = len(msg.get("content", ""))
            tokens = int(content_len / 3.5)
            if estimated_tokens + tokens > self.max_history_tokens:
                break
            truncated.insert(0, msg)
            estimated_tokens += tokens

        return truncated

    async def _retrieve_context(
        self,
        message: str,
        kb_name: str | None,
        use_rag: bool,
        use_web: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Execute tool lookups safely."""
        context_parts: list[str] = []
        sources: dict[str, Any] = {"rag": [], "web": []}

        # RAG retrieval
        if enable_rag and kb_name:
            try:
                self.logger.info(f"RAG search: {message[:50]}...")
                rag_result = await rag_search(
                    query=message,
                    kb_name=kb_name,
                    mode="hybrid",
                )
                rag_answer = rag_result.get("answer", "")
                if rag_answer:
                    context_parts.append(f"[Knowledge Base: {kb_name}]\n{rag_answer}")
                    sources["rag"].append(
                        {
                            "kb_name": kb_name,
                            "content": (
                                rag_answer[:500] + "..." if len(rag_answer) > 500 else rag_answer
                            ),
                        }
                    )
                    self.logger.info(f"RAG retrieved {len(rag_answer)} chars")
            except Exception as e:
                self.logger.warning(f"RAG search failed: {e}")
        if use_rag and kb_name and self._rag_search is not None:
            try:
                result = await self._rag_search(message, kb_name=kb_name)
                if result.get("answer"):
                    context_parts.append(f"[Knowledge Base]\n{result['answer']}")
                    sources["rag"].append(result)
            except Exception as exc:
                self.logger.error(f"RAG failed: {exc}")

        if use_web and self._web_search is not None:
            try:
                result = await self._web_search(message)
                if result.get("answer"):
                    context_parts.append(f"[Web Search]\n{result['answer']}")
                    sources["web"] = result.get("citations", [])
            except Exception as exc:
                self.logger.error(f"Web search failed: {exc}")

        return "\n\n".join(context_parts), sources

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        kb_name: str | None = None,
        enable_rag: bool = False,
        enable_web_search: bool = False,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        history = history or []
        truncated_history = self._truncate_history(history)

        context_str, sources = await self._retrieve_context(
            message,
            kb_name,
            use_rag=enable_rag,
            use_web=enable_web_search,
        )

        sys_template = self.get_prompt("system", "You are a helpful AI.")

        messages: list[dict[str, str]] = [{"role": "system", "content": sys_template}]

        if context_str:
            messages.append(
                {
                    "role": "system",
                    "content": f"Context:\n{context_str}",
                }
            )

        messages.extend(truncated_history)
        messages.append({"role": "user", "content": message})

        if stream:

            async def generator() -> AsyncGenerator[dict[str, Any], None]:
                async for chunk in self.stream_llm(
                    user_prompt=message,
                    system_prompt=sys_template,
                    messages=messages,
                    stage="chat_stream",
                ):
                    yield {"type": "chunk", "content": chunk}

                yield {
                    "type": "meta",
                    "sources": sources,
                    "truncated_history": truncated_history,
                }

            return generator()

        response_text = await self.call_llm(
            user_prompt=message,
            system_prompt=sys_template,
            messages=messages,
            verbose=False,
            stage="chat_complete",
        )

        return {
            "response": response_text,
            "sources": sources,
            "truncated_history": truncated_history,
        }


__all__ = ["ChatAgent"]
