from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.co_writer.narrator_agent import NarratorAgent


@pytest.fixture
def narrator_agent() -> NarratorAgent:
    """Provides a NarratorAgent instance with mocked dependencies.

    Yields:
        NarratorAgent: An instance of the NarratorAgent.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
        patch(
            "src.agents.co_writer.narrator_agent.get_tts_config",
            return_value={
                "model": "tts-1",
                "api_key": "test_key",
                "base_url": "http://localhost:1234",
            },
        ),
    ):
        agent = NarratorAgent()
        yield agent


@pytest.mark.asyncio
async def test_narrator_agent_generate_script(narrator_agent: NarratorAgent):
    """Test that the generate_script method correctly calls the LLM and returns the script.

    Args:
        narrator_agent (NarratorAgent): The NarratorAgent instance.
    """
    narrator_agent.call_llm = AsyncMock(return_value="test script")
    narrator_agent.get_prompt = MagicMock(return_value="prompt")
    narrator_agent._extract_key_points = AsyncMock(return_value=[])

    result = await narrator_agent.generate_script("test content")

    narrator_agent.call_llm.assert_called_once()
    assert result["script"] == "test script"


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_narrator_agent_generate_audio(mock_openai, narrator_agent: NarratorAgent):
    """Test that the generate_audio method correctly calls the TTS API and saves the audio file.

    Args:
        mock_openai: Mock object for the OpenAI client.
        narrator_agent (NarratorAgent): The NarratorAgent instance.
    """
    mock_audio_client = MagicMock()
    mock_audio_client.speech.create.return_value = AsyncMock()
    mock_openai.return_value = mock_audio_client

    result = await narrator_agent.generate_audio("test script")

    mock_audio_client.speech.create.assert_called_once()
    assert "audio_path" in result
    assert "audio_url" in result


@pytest.mark.asyncio
async def test_narrator_agent_narrate_workflow(narrator_agent: NarratorAgent):
    """Test the overall workflow of the narrate method.

    Args:
        narrator_agent (NarratorAgent): The NarratorAgent instance.
    """
    narrator_agent.generate_script = AsyncMock(
        return_value={
            "script": "s",
            "key_points": [],
            "original_length": 1,
            "script_length": 1,
        }
    )
    narrator_agent.generate_audio = AsyncMock(
        return_value={
            "audio_path": "path",
            "audio_url": "url",
            "audio_id": "id",
        }
    )

    result = await narrator_agent.narrate("test content")

    narrator_agent.generate_script.assert_called_once()
    narrator_agent.generate_audio.assert_called_once()
    assert result["has_audio"] is True


@pytest.mark.asyncio
async def test_extract_key_points(narrator_agent: NarratorAgent):
    """Test the _extract_key_points method.

    Args:
        narrator_agent (NarratorAgent): The NarratorAgent instance.
    """
    narrator_agent.call_llm = AsyncMock(return_value='["point1", "point2"]')
    narrator_agent.get_prompt = MagicMock(return_value="prompt")

    points = await narrator_agent._extract_key_points("content")

    assert points == ["point1", "point2"]


def test_validate_tts_config(narrator_agent: NarratorAgent):
    """Test the _validate_tts_config method.

    Args:
        narrator_agent (NarratorAgent): The NarratorAgent instance.
    """
    # Should not raise any exception
    narrator_agent._validate_tts_config()

    with pytest.raises(ValueError):
        narrator_agent.tts_config = {}
        narrator_agent._validate_tts_config()
