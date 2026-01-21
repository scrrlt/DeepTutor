# -*- coding: utf-8 -*-
"""
ChatAgent (Hardened)
====================

Conversational agent with multi-turn support, optimized for throughput
and resilience. Enforces strict token management and explicit error handling.
"""

import asyncio
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

# Add project root to path (Defensive path setup)
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import tiktoken
except ImportError:
    tiktoken = None

from src.agents.base_agent import BaseAgent
from src.services.llm.exceptions import LLMConfigError
from src.services.llm.utils import is_local_llm_server
from src.tools import rag_search, web_search

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """
    Hardened conversational agent.

    Features:
    - Cached tokenizer (zero-latency lookups).
    - Strict configuration validation (no implicit side-effects).
    - Type-safe streaming contracts (content-only).
    - Defensive history truncation.
    """

    DEFAULT_MAX_HISTORY_TOKENS = 4000
    _tokenizer = None  # Class-level cache

    def __init__(
        self,
        language: str = "zh",
        config: Optional[Dict[str, Any]] = None,
        max_history_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            module_name="chat",
            agent_name="chat_agent",
            language=language,
            config=config,
            **kwargs,
        )

        self.max_history_tokens = max_history_tokens or self.agent_config.get(
            "max_history_tokens", self.DEFAULT_MAX_HISTORY_TOKENS
        )

        base_url = (self.base_url or "").strip()
        api_key = (self.api_key or "").strip()
        if not base_url:
            raise LLMConfigError(
                "ChatAgent requires a non-empty 'base_url'. "
                "Ensure LLM factory is fully initialized before instantiating agents."
            )

        if not api_key and not is_local_llm_server(base_url):
            raise LLMConfigError(
                "ChatAgent requires 'api_key' for remote providers. "
                "Local providers may omit the key."
            )

        self.logger.info(
            "ChatAgent initialized: model=%s, max_tokens=%d",
            self.model,
            self.max_history_tokens,
        )

    @classmethod
    def _get_tokenizer(cls):
        """Singleton accessor for tokenizer to avoid re-initialization cost."""
        if cls._tokenizer is None and tiktoken is not None:
            try:
                cls._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning("Failed to initialize tiktoken: %s. Using fallback.", e)
        return cls._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count tokens with cached encoding. Optimized hot path.

        Note: Fallback character-based estimation is intentionally pessimistic
        to ensure context limits are never exceeded.
        """
        if not text:
            return 0

        encoder = self._get_tokenizer()
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass

        # Fallback: Pessimistic character heuristic
        return len(text) // 3

    def truncate_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Defensive history truncation prioritizing recent context."""
        limit = max_tokens or self.max_history_tokens
        if not history or limit <= 0:
            return []

        truncated = []
        current_tokens = 0

        for msg in reversed(history):
            content = msg.get("content", "") or ""
            msg_tokens = self.count_tokens(content)

            if current_tokens + msg_tokens > limit:
                if not truncated:
                    self.logger.warning(
                        "Latest message (%d tokens) exceeds limit (%d). Dropping.",
                        msg_tokens,
                        limit,
                    )
                break

            truncated.append(msg)
            current_tokens += msg_tokens

        truncated.reverse()
        return truncated

    async def retrieve_context(
        self,
        message: str,
        kb_name: Optional[str] = None,
        enable_rag: bool = False,
        enable_web_search: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Retrieve context with explicit failure visibility in logs."""
        context_parts = []
        sources = {"rag": [], "web": []}

        if enable_rag and kb_name:
            try:
                rag_result = await rag_search(query=message, kb_name=kb_name, mode="hybrid")
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
            except Exception as e:
                self.logger.error("RAG search failed: %s", e)

        if enable_web_search:
            try:
                from functools import partial

                loop = asyncio.get_running_loop()
                web_result = await loop.run_in_executor(
                    None, partial(web_search, query=message, verbose=False)
                )
                web_answer = web_result.get("answer", "")
                if web_answer:
                    context_parts.append(f"[Web Search Results]\n{web_answer}")
                    sources["web"] = web_result.get("citations", [])[:5]
            except Exception as e:
                self.logger.error("Web search failed: %s", e)

        return "\n\n".join(context_parts), sources

    def build_messages(
        self, message: str, history: List[Dict[str, str]], context: str = ""
    ) -> List[Dict[str, str]]:
        """Build message array for LLM provider."""
        messages = []
        system_prompt = self.get_prompt("system", "You are a helpful AI assistant.")
        messages.append({"role": "system", "content": system_prompt})

        if context:
            template = self.get_prompt("context_template", "Reference context:\n{context}")
            messages.append({"role": "system", "content": template.format(context=context)})

        for msg in history:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": message})
        return messages

    async def generate_stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate streaming response content chunks."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            stage="chat_stream",
        ):
            yield chunk

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate complete non-streaming response."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

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
        history: Optional[List[Dict[str, str]]] = None,
        kb_name: Optional[str] = None,
        enable_rag: bool = False,
        enable_web_search: bool = False,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """
        Process chat.

        If stream=True, returns AsyncGenerator yielding ONLY content chunks (strings).
        If stream=False, returns Dict with response and metadata.
        """
        history = history or []
        truncated_history = self.truncate_history(history)
        context, sources = await self.retrieve_context(
            message=message,
            kb_name=kb_name,
            enable_rag=enable_rag,
            enable_web_search=enable_web_search,
        )

        messages = self.build_messages(message, truncated_history, context)

        if stream:
            return self._stream_generator(messages)
        else:
            response = await self.generate(messages)
            return {
                "response": response,
                "sources": sources,
                "truncated_history": truncated_history,
                "usage": {"token_count": self.count_tokens(response)},
            }

    async def _stream_generator(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Encapsulated generator yielding pure content chunks."""
        try:
            async for chunk in self.generate_stream(messages):
                yield chunk
        except Exception as e:
            self.logger.error("Streaming failed: %s", e)
            raise
