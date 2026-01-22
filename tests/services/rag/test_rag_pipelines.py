#!/usr/bin/env python
"""
RAG Pipeline Integration Tests
==============================

Lightweight validation of pipeline registry and provider selection.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

from dotenv import load_dotenv

from src.logging import get_logger
from src.services.rag.factory import has_pipeline
from src.tools.rag_tool import get_available_providers, get_current_provider

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / "DeepTutor.env", override=False)
load_dotenv(project_root / ".env", override=False)

logger = get_logger("TestRAGPipeline")

SUPPORTED_PIPELINES = ["raganything", "lightrag", "llamaindex"]


class TestRAGPipelineRegistry(unittest.IsolatedAsyncioTestCase):
    """Validate that known pipelines are registered and selectable."""

    async def test_list_available_providers(self) -> None:
        providers = get_available_providers()
        logger.info("Found %d providers", len(providers))

        self.assertIsInstance(providers, list)
        self.assertGreater(len(providers), 0)

        provider_ids = [provider["id"] for provider in providers]
        for expected in SUPPORTED_PIPELINES:
            self.assertIn(expected, provider_ids, f"Missing provider: {expected}")

    async def test_get_current_provider(self) -> None:
        provider = get_current_provider()
        logger.info("Current provider: %s", provider)

        self.assertIsInstance(provider, str)
        self.assertIn(provider, SUPPORTED_PIPELINES)

    async def test_has_pipeline(self) -> None:
        for name in SUPPORTED_PIPELINES:
            self.assertTrue(has_pipeline(name), f"Pipeline {name} should exist")

        self.assertFalse(has_pipeline("unknown_pipeline"))


if __name__ == "__main__":
    unittest.main()
