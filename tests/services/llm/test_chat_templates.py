# -*- coding: utf-8 -*-
"""Tests for chat template resolution utilities."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# Prevent optional provider import-time failures during collection
mod = types.ModuleType("src.services.search.providers")
mod.get_available_providers = lambda: []
mod.get_default_provider = lambda: "perplexity"
mod.get_provider = lambda name: types.SimpleNamespace(
    name=name,
    supports_answer=True,
    search=lambda query, **kwargs: types.SimpleNamespace(
        to_dict=lambda: {"answer": "", "citations": [], "search_results": []}
    ),
)
mod.get_providers_info = lambda: []
mod.list_providers = lambda: []
sys.modules.setdefault("src.services.search.providers", mod)

from src.services.llm.chat_templates import render_chat_template, resolve_chat_template


def test_resolve_chat_template_from_tokenizer_config(tmp_path: Path) -> None:
    template = "{{ messages[0].content }}"
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(
        json.dumps({"chat_template": template, "bos_token": "<s>"}),
        encoding="utf-8",
    )

    info = resolve_chat_template(model="test", tokenizer_dir=str(tmp_path))
    assert info is not None
    assert info.template == template

    rendered = render_chat_template(
        [{"role": "user", "content": "hello"}],
        info,
        add_generation_prompt=False,
    )
    assert "hello" in rendered


def test_tokenizer_dir_collects_tokens_from_both_files(tmp_path: Path) -> None:
    # tokenizer_config.json contains the template, tokenizer.json contains extra tokens
    tokenizer_dir = tmp_path
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "hello"}), encoding="utf-8"
    )
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps({"bos_token": "<BOS>"}), encoding="utf-8"
    )

    info = resolve_chat_template(model=None, tokenizer_dir=str(tokenizer_dir))
    assert info is not None
    assert info.template == "hello"
    assert info.metadata.get("bos_token") == "<BOS>"


def test_registry_prefers_full_model_name(tmp_path: Path) -> None:
    # Create templates for both full name and prefix; ensure full name is selected
    template_dir = tmp_path
    (template_dir / "llama-3.1-8b.jinja").write_text("full", encoding="utf-8")
    (template_dir / "llama.jinja").write_text("prefix", encoding="utf-8")

    info = resolve_chat_template(model="llama-3.1-8b", template_dir=str(template_dir))
    assert info is not None
    assert info.template == "full"
    assert "llama-3.1-8b.jinja" in info.source
