import sys
import types
import pytest

# Insert light-weight stubs for heavy optional service modules that are not
# necessary for this unit test. This prevents import-time errors during test
# collection without pulling in optional providers.
sys.modules.setdefault(
    "src.services.search.providers", types.ModuleType("src.services.search.providers")
)

from src.agents.chat import ChatAgent
from src.services.llm.exceptions import LLMConfigError
from src.services.llm.config import LLMConfig


def test_chat_agent_raises_when_credentials_missing(monkeypatch):
    """If LLM credentials are missing, ChatAgent should attempt refresh and then raise LLMConfigError."""

    # Simulate get_llm_config returning incomplete credentials
    monkeypatch.setattr(
        "src.services.llm.config.get_llm_config",
        lambda: LLMConfig(model="gpt-test", api_key="", base_url=None),
    )

    with pytest.raises(LLMConfigError):
        ChatAgent(language="en", config={})


def test_chat_agent_initializes_with_credentials(monkeypatch):
    """ChatAgent initializes normally when credentials are available."""

    # Simulate valid config
    monkeypatch.setattr(
        "src.services.llm.config.get_llm_config",
        lambda: LLMConfig(model="gpt-test", api_key="fake-key", base_url="https://api.example.com"),
    )

    agent = ChatAgent(language="en", config={})
    assert agent.api_key == "fake-key"
    assert agent.base_url == "https://api.example.com"
    assert agent.model == "gpt-test"
