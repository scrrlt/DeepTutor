"""
Tests for LLM utility functions.
"""

from src.services.llm import utils


def test_is_local_llm_server() -> None:
    assert utils.is_local_llm_server("http://localhost:11434")
    assert utils.is_local_llm_server("http://127.0.0.1:8000")
    assert utils.is_local_llm_server("http://0.0.0.0:5000")
    assert utils.is_local_llm_server(
        "http://host.docker.internal:8000"
    )  # Common dev setup might be considered local effectively or caught by port logic if port matches

    # We might need to check implementation for host.docker.internal, usually it's not strictly local IP but development.
    # The current implementation checks: localhost, 127/8, strict IP/hostname match, OR known ports.
    # 11434 is a known port.

    assert not utils.is_local_llm_server("https://api.openai.com")
    assert not utils.is_local_llm_server("https://api.anthropic.com")
    assert not utils.is_local_llm_server("https://google.com")
    assert not utils.is_local_llm_server(None)
    assert not utils.is_local_llm_server("")
    assert not utils.is_local_llm_server(
        "http://192.168.1.1:1234"
    )  # default: private LAN is not treated as local

    # Explicitly allow private prefixts via parameter
    assert utils.is_local_llm_server("http://192.168.1.1:1234", allow_private=True)
    # Or via environment variable
    import os

    os.environ["LLM_TREAT_PRIVATE_AS_LOCAL"] = "true"
    assert utils.is_local_llm_server("http://192.168.1.1:1234")
    del os.environ["LLM_TREAT_PRIVATE_AS_LOCAL"]


def test_sanitize_url() -> None:
    # sanitize_url only strips specific suffixes, not arbitrary paths
    assert utils.sanitize_url("localhost:11434/api/chat") == "http://localhost:11434/api"
    # But it adds http
    assert utils.sanitize_url("localhost:11434/v1") == "http://localhost:11434"
    assert utils.sanitize_url("example.com") == "http://example.com"
    assert utils.sanitize_url(None) == ""
    assert utils.sanitize_url("http://example.com/v1/chat/completions") == "http://example.com"


def test_clean_thinking_tags() -> None:
    content = "Here is <think>some thought</think> the answer."
    assert utils.clean_thinking_tags(content) == "Here is  the answer."

    content_no_tags = "Just answer."
    assert utils.clean_thinking_tags(content_no_tags) == "Just answer."

    assert utils.clean_thinking_tags(None) == ""

    # Test binding check logic (mocking capabilities would be needed if we passed binding)
    # But clean_thinking_tags imports inside function.
    # For now test basic tag removal.
    content_with_binding = "<think>thought</think> answer"
    assert utils.clean_thinking_tags(content_with_binding, binding="deepseek") == "answer"
    assert utils.clean_thinking_tags(content_with_binding, binding="openai") == content_with_binding


def test_build_chat_url() -> None:
    base = "http://localhost:11434"
    assert utils.build_chat_url(base) == "http://localhost:11434/chat/completions"

    assert utils.build_chat_url(base, binding="anthropic") == "http://localhost:11434/messages"

    # Already has suffix
    assert (
        utils.build_chat_url("http://host/v1/chat/completions") == "http://host/v1/chat/completions"
    )

    # Azure
    assert "api-version=2023-05-15" in utils.build_chat_url(
        "https://azure", api_version="2023-05-15"
    )
    assert utils.build_chat_url(None) is None


def test_extract_response_content() -> None:
    assert utils.extract_response_content({"content": "hello"}) == "hello"
    assert utils.extract_response_content({"reasoning_content": "thought"}) == "thought"
    assert utils.extract_response_content({"tool_calls": ["call"]}) == "<tool_call>"
    assert utils.extract_response_content({"reasoning": "reasoning"}) == "reasoning"
    assert utils.extract_response_content({"thought": "thought"}) == "thought"
    assert utils.extract_response_content({}) == ""
    assert utils.extract_response_content(None) == ""


def test_build_auth_headers() -> None:
    assert utils.build_auth_headers("sk-key")["Authorization"] == "Bearer sk-key"

    headers = utils.build_auth_headers("sk-key", binding="anthropic")
    assert headers["x-api-key"] == "sk-key"
    assert "anthropic-version" in headers

    headers = utils.build_auth_headers("sk-key", binding="azure_openai")
    assert headers["api-key"] == "sk-key"

    assert utils.build_auth_headers(None) == {"Content-Type": "application/json"}
