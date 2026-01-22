"""ChatAgent - Orchestrator for conversational flows.

This agent is intentionally lightweight:

- Delegates LLM calls to BaseAgent (provider-backed)
- Accepts optional injected tool functions for RAG/Web
- Uses a model-agnostic history truncation heuristic
"""

import asyncio
from functools import partial
from pathlib import Path
import sys
from typing import Any, AsyncGenerator

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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

        # Web search
        if enable_web_search:
            try:
                self.logger.info(f"Web search: {message[:50]}...")
                loop = asyncio.get_running_loop()
                web_result = await loop.run_in_executor(
                    None,
                    partial(web_search, query=message, verbose=False),
                )
                web_answer = web_result.get("answer", "")
                web_citations = web_result.get("citations", [])

                if web_answer:
                    context_parts.append(f"[Web Search Results]\n{web_answer}")
                    sources["web"] = web_citations[:5]
                    self.logger.info(
                        f"Web search returned {len(web_answer)} chars, "
                        f"{len(web_citations)} citations"
                    )
            except Exception as e:
                self.logger.warning(f"Web search failed: {e}")

        context = "\n\n".join(context_parts)
        return context, sources

    def build_messages(
        self,
        message: str,
        history: list[dict[str, str]],
        context: str = "",
    ) -> list[dict[str, str]]:
        """
        Build the messages array for the LLM API call.

        Args:
            message: Current user message
            history: Truncated conversation history
            context: Retrieved context (RAG/Web)

        Returns:
            List of message dicts for OpenAI API
        """
        messages = []

        # System prompt
        system_prompt = self.get_prompt("system", "You are a helpful AI assistant.")
        messages.append({"role": "system", "content": system_prompt})

        # Add context if available
        if context:
            context_template = self.get_prompt("context_template", "Reference context:\n{context}")
            context_msg = context_template.format(context=context)
            messages.append({"role": "system", "content": context_msg})

        # Add conversation history
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.

        Uses BaseAgent.stream_llm() which routes to the appropriate provider
        (cloud or local) based on configuration.

        Args:
            messages: Messages array for OpenAI API

        Yields:
            Response chunks as strings
        """
        # Extract system prompt from messages
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        # Get the last user message as user_prompt (for logging/tracking)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

        # Use BaseAgent's stream_llm which routes through the factory
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            stage="chat_stream",
        ):
            yield chunk

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Generate complete response from LLM (non-streaming).

        Uses BaseAgent.call_llm() which routes to the appropriate provider
        (cloud or local) based on configuration.

        Args:
            messages: Messages array for OpenAI API

        Returns:
            Complete response string
        """
        # Extract system prompt from messages
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

        # Use BaseAgent's call_llm which routes through the factory
        # Note: call_llm expects simple prompt/system_prompt, but for multi-turn
        # we need to use the factory directly with messages
        from src.services.llm import complete as llm_complete

        response = await llm_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=self.get_model(),
            api_key=self.api_key,
            base_url=self.base_url,
            messages=messages,
            temperature=self.get_temperature(),
        )

        # Track token usage
        self._track_tokens(
            model=self.get_model(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            stage="chat",
        )

        return response

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
