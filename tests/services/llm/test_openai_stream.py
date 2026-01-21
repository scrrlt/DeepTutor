# -*- coding: utf-8 -*-
import pytest

from types import SimpleNamespace

import importlib.util
import pathlib

# Load cloud_provider by file path to avoid package import-time side-effects
_cloud_path = (
    pathlib.Path(__file__).resolve().parents[3] / "src" / "services" / "llm" / "cloud_provider.py"
)
import sys
import types

# Prepare minimal package modules to avoid importing src.services.__init__
if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
if "src.services" not in sys.modules:
    sys.modules["src.services"] = types.ModuleType("src.services")
if "src.services.llm" not in sys.modules:
    llm_pkg = types.ModuleType("src.services.llm")
    # Mark as a package so relative imports inside submodules work during test
    llm_pkg.__path__ = [str(_cloud_path.parent)]
    sys.modules["src.services.llm"] = llm_pkg

# Preload capabilities and utils under their full dotted names so cloud_provider's
# relative imports resolve without executing package __init__ side effects.
_cap_path = _cloud_path.parent / "capabilities.py"
_utils_path = _cloud_path.parent / "utils.py"

spec_caps = importlib.util.spec_from_file_location("src.services.llm.capabilities", str(_cap_path))
caps_mod = importlib.util.module_from_spec(spec_caps)
spec_caps.loader.exec_module(caps_mod)  # type: ignore
sys.modules["src.services.llm.capabilities"] = caps_mod

spec_utils = importlib.util.spec_from_file_location("src.services.llm.utils", str(_utils_path))
utils_mod = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(utils_mod)  # type: ignore
sys.modules["src.services.llm.utils"] = utils_mod

# Preload config and exceptions as cloud_provider imports them at module import time
_conf_path = _cloud_path.parent / "config.py"
_exc_path = _cloud_path.parent / "exceptions.py"

spec_conf = importlib.util.spec_from_file_location("src.services.llm.config", str(_conf_path))
conf_mod = importlib.util.module_from_spec(spec_conf)
spec_conf.loader.exec_module(conf_mod)  # type: ignore
sys.modules["src.services.llm.config"] = conf_mod

spec_exc = importlib.util.spec_from_file_location("src.services.llm.exceptions", str(_exc_path))
exc_mod = importlib.util.module_from_spec(spec_exc)
spec_exc.loader.exec_module(exc_mod)  # type: ignore
sys.modules["src.services.llm.exceptions"] = exc_mod

spec = importlib.util.spec_from_file_location("src.services.llm.cloud_provider", str(_cloud_path))
cloud_provider = importlib.util.module_from_spec(spec)
# Execute cloud_provider with the above preloaded modules present
spec.loader.exec_module(cloud_provider)  # type: ignore


class AsyncBytesIterator:
    def __init__(self, lines):
        # copy to allow reuse
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class FakeResp:
    def __init__(self, lines, status=200):
        self.status = status
        self.content = AsyncBytesIterator(lines)

    async def text(self):
        # return joined payload for diagnostics
        return b"".join(getattr(self.content, "_lines", [])).decode("utf-8")


class FakePostCtx:
    def __init__(self, resp: FakeResp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, resp: FakeResp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return FakePostCtx(self._resp)


@pytest.mark.asyncio
async def test_openai_stream_skips_malformed_and_empty_and_role_only(monkeypatch):
    # SSE frames: malformed JSON, empty choices, role-only delta, then content
    lines = [
        b"data: {malformed}\n",
        b'data: {"choices": []}\n',
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b"data: [DONE]\n",
    ]

    resp = FakeResp(lines)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: FakeSession(resp))

    # Ensure module-level SSL warning flag exists for connector helper
    setattr(cloud_provider, "_ssl_warning_logged", False)

    results = []
    async for chunk in cloud_provider._openai_stream(
        model="test",
        prompt="p",
        system_prompt="s",
        api_key=None,
        base_url="http://localhost:11434",
        binding="openai",
    ):
        results.append(chunk)

    assert results == ["Hello"]


@pytest.mark.asyncio
async def test_openai_stream_thinking_block_flush(monkeypatch):
    # Use binding that supports thinking tags so StreamParser will remove them
    lines = [
        b'data: {"choices":[{"delta":{"content":"Lead <think>secret"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"more</think> trailing"}}]}\n',
        b"data: [DONE]\n",
    ]

    resp = FakeResp(lines)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: FakeSession(resp))

    # Ensure module-level SSL warning flag exists for connector helper
    setattr(cloud_provider, "_ssl_warning_logged", False)

    results = []
    async for chunk in cloud_provider._openai_stream(
        model="deepseek-1",
        prompt="p",
        system_prompt="s",
        api_key=None,
        base_url="http://localhost:1234",
        binding="deepseek",
    ):
        results.append(chunk)

    # Expect the thinking block to be removed and surrounding text emitted
    assert results == ["Lead ", " trailing"]


@pytest.mark.asyncio
async def test_openai_stream_handles_non_string_or_no_content(monkeypatch):
    # Frame with choices but delta missing content, should be ignored
    lines = [
        b'data: {"choices":[{"delta":{}}]}\n',
        b'data: {"choices":[{"delta":{"content":"World"}}]}\n',
        b"data: [DONE]\n",
    ]

    resp = FakeResp(lines)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: FakeSession(resp))

    # Ensure module-level SSL warning flag exists for connector helper
    setattr(cloud_provider, "_ssl_warning_logged", False)

    results = []
    async for chunk in cloud_provider._openai_stream(
        model="test",
        prompt="p",
        system_prompt="s",
        api_key=None,
        base_url="https://api.openai.com/v1",
        binding="openai",
    ):
        results.append(chunk)

    assert results == ["World"]
