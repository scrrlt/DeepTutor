# tests/agents/chat/test_chat_agent.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.agents.chat.chat_agent import ChatAgent
from src.config.config import LLMConfig

@pytest.fixture
def mock_config():
    return LLMConfig(
        model="gpt-4o",
        api_key="sk-test",
        binding="openai"
    )

@pytest.fixture
def mock_provider():
    """Mock the LLMProvider to avoid hitting OpenAI."""
    provider = AsyncMock()
    # Mock the response object structure
    response = MagicMock()
    response.content = "I am a test bot."
    provider.complete.return_value = response
    
    # Mock streaming response
    async def mock_stream(*args, **kwargs):
        yield MagicMock(content="I am ")
        yield MagicMock(content="streaming.")
    provider.stream = mock_stream
    
    return provider

@pytest.fixture
def chat_agent(mock_config, mock_provider):
    with patch("src.agents.base_agent.LLMFactory.get_provider", return_value=mock_provider):
        agent = ChatAgent(config=mock_config)
        return agent

@pytest.mark.asyncio
async def test_agent_initialization(chat_agent):
    assert chat_agent.agent_name == "chat_agent"
    assert chat_agent.model == "gpt-4o"

@pytest.mark.asyncio
async def test_process_simple_chat(chat_agent):
    response = await chat_agent.process(
        message="Hello",
        history=[],
        stream=False
    )
    
    # Verify LLM was called
    chat_agent.provider.complete.assert_called_once()
    assert response["response"] == "I am a test bot."
    assert len(response["truncated_history"]) == 0

@pytest.mark.asyncio
async def test_context_truncation(chat_agent):
    # Create a huge history
    long_history = [{"role": "user", "content": "test " * 100} for _ in range(50)]
    
    response = await chat_agent.process("Hi", history=long_history)
    
    # Ensure history passed to LLM is shorter than the input history
    call_args = chat_agent.provider.complete.call_args
    passed_messages = call_args.kwargs["messages"]
    
    # Count user messages in the final payload
    history_count = sum(1 for m in passed_messages if m["role"] == "user")
    # Should be less than 50 + 1 (current message)
    assert history_count < 51

@pytest.mark.asyncio
async def test_rag_integration(chat_agent):
    # Mock the injected RAG tool
    mock_rag = AsyncMock(return_value={"answer": "DeepTutor is an AI tool."})
    chat_agent._rag_search = mock_rag
    
    await chat_agent.process(
        message="What is DeepTutor?",
        kb_name="docs",
        enable_rag=True
    )
    
    # Verify RAG tool was called
    mock_rag.assert_called_with("What is DeepTutor?", kb_name="docs")
    
    # Verify Context was injected into System Prompt
    call_args = chat_agent.provider.complete.call_args
    messages = call_args.kwargs["messages"]
    system_msg = next((m for m in messages if m["role"] == "system" and "Context:" in m["content"]), None)
    assert system_msg is not None, "Expected system message with Context not found"
    assert "DeepTutor is an AI tool" in system_msg["content"]

@pytest.mark.asyncio
async def test_streaming_flow(chat_agent):
    chunks = []
    generator = await chat_agent.process("Stream me", stream=True)
    
    async for chunk in generator:
        if chunk["type"] == "chunk":
            chunks.append(chunk["content"])
            
    assert "".join(chunks) == "I am streaming."