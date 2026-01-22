"""Deprecated: legacy local provider tests.

These tests targeted the previous ``LocalLLMProvider`` seam exposed via
``src.services.llm.local_provider``. The local-provider integration is
now exercised through the unified provider routing and client tests.

The module is intentionally left without active tests to avoid
depending on obsolete behaviour while keeping the filename stable.
"""
