"""Deprecated: legacy cloud provider tests.

This module formerly exercised the old OpenAIProvider seam in
``src.services.llm.cloud_provider``. The provider has since been
replaced by the new routing/provider architecture, and coverage is now
provided by new tests that target the unified LLM client and routing
layer.

The file is intentionally left without tests to avoid importing the
old API surface while keeping the path available for any external
references.
"""
