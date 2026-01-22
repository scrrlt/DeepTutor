"""Integration tests for LLM and Embedding providers."""

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import pytest

# Load environment variables from .env.local if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_local = PROJECT_ROOT / ".env.local"
if env_local.exists():
    load_dotenv(env_local, override=True)

from src.services.llm import complete as llm_complete

RUN_NETWORK_TESTS = os.getenv("RUN_LLM_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set RUN_LLM_NETWORK_TESTS=1 to enable integration provider calls.",
)


@pytest.mark.asyncio
async def test_llm_provider_integration():
    """Verify the active LLM provider works as intended."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key and "localhost" not in os.getenv("LLM_HOST", ""):
        pytest.skip(
            "LLM_API_KEY not set and not using local host, skipping integration test"
        )

    # Bare call letting pytest report failures naturally
    response = await asyncio.wait_for(
        llm_complete("Say 'DeepTutor' and nothing else."),
        timeout=30.0,
    )
    assert "DeepTutor" in response
    logger.info(f"✓ LLM provider verified: {os.getenv('LLM_BINDING', 'openai')}")
