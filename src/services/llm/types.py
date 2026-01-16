"""Shared LLM response data models."""

from typing import Any, AsyncGenerator, Dict, Optional

from pydantic import BaseModel, Field


class TutorResponse(BaseModel):
    """LLM completion response container."""

    content: str
    raw_response: Dict[str, Any]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    provider: str
    model: str
    finish_reason: Optional[str] = None
    cost_estimate: float = 0.0


class TutorStreamChunk(BaseModel):
    """Chunk emitted during streamed LLM responses."""

    content: str
    delta: str
    provider: str
    model: str
    is_complete: bool = False
    usage: Optional[Dict[str, int]] = None


AsyncStreamGenerator = AsyncGenerator[TutorStreamChunk, None]
