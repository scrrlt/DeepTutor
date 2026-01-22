"""Deprecated: legacy cloud provider resilience tests.

Previously validated circuit breaker and telemetry behaviour on the
old ``OpenAIProvider`` surface. These behaviours are now covered by
tests targeting the shared BaseLLMProvider and routing layer.

The module is intentionally test-free to avoid importing obsolete
APIs while preserving the file path for any external references.
"""
