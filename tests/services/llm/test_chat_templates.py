"""Tests for chat template resolution utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.services.llm.chat_templates import (
    render_chat_template,
    resolve_chat_template,
)


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
