# -*- coding: utf-8 -*-
"""Tests for TLS diagnostics helper."""

from src.services.llm.cloud_provider import run_tls_diagnostics


def test_run_tls_diagnostics_invalid_host():
    info = run_tls_diagnostics("https://nonexistent.invalid.local:443")
    assert "host" in info and "port" in info
    # Since host is invalid, diagnostic should indicate failure
    assert info.get("success") is False
    assert "connect_error" in info or "tls_error" in info
