import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agents.guide.guide_manager import GuideManager
from src.utils.config import load_config_with_main


@pytest.fixture
def guide_manager(tmp_path) -> GuideManager:
    """Provides a GuideManager instance with mocked dependencies.

    Args:
        tmp_path: Pytest fixture for a temporary directory.

    Yields:
        GuideManager: An instance of the GuideManager.
    """
    with patch(
        "src.agents.guide.guide_manager.load_config_with_main",
        return_value={
            "system": {"language": "en"},
            "paths": {"user_log_dir": "/tmp/logs"},
        },
    ):
        manager = GuideManager(
            api_key="test_key",
            base_url="http://localhost:1234",
            output_dir=str(tmp_path),
        )
        yield manager


@pytest.mark.asyncio
async def test_create_session(guide_manager: GuideManager):
    """Test that the create_session method correctly creates a new session.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    guide_manager.locate_agent.process = AsyncMock(
        return_value={"success": True, "knowledge_points": [{"title": "kp1"}]}
    )

    result = await guide_manager.create_session("nid", "nname", [])

    assert result["success"] is True
    assert "session_id" in result
    assert len(result["knowledge_points"]) == 1


@pytest.mark.asyncio
async def test_start_learning(guide_manager: GuideManager):
    """Test that the start_learning method correctly starts the learning process.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    session = MagicMock()
    session.knowledge_points = [{"title": "kp1"}]
    session.current_index = 0
    guide_manager._load_session = MagicMock(return_value=session)
    guide_manager.interactive_agent.process = AsyncMock(return_value={"html": "<html>"})

    result = await guide_manager.start_learning("sid")

    assert result["success"] is True
    assert result["current_index"] == 0
    assert result["html"] == "<html>"
    

@pytest.mark.asyncio
async def test_next_knowledge(guide_manager: GuideManager):
    """Test that the next_knowledge method correctly moves to the next knowledge point.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    session = MagicMock()
    session.knowledge_points = [{"title": "kp1"}, {"title": "kp2"}]
    session.current_index = 0
    guide_manager._load_session = MagicMock(return_value=session)
    guide_manager.interactive_agent.process = AsyncMock(return_value={"html": "<html>"})

    result = await guide_manager.next_knowledge("sid")

    assert result["success"] is True
    assert result["current_index"] == 1


@pytest.mark.asyncio
async def test_chat(guide_manager: GuideManager):
    """Test that the chat method correctly handles chat interactions.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    session = MagicMock()
    session.status = "learning"
    session.knowledge_points = [{"title": "kp1"}]
    session.current_index = 0
    guide_manager._load_session = MagicMock(return_value=session)
    guide_manager.chat_agent.process = AsyncMock(return_value={"answer": "assistant_response"})

    result = await guide_manager.chat("sid", "user_message")

    assert result["success"] is True
    assert result["answer"] == "assistant_response"


@pytest.mark.asyncio
async def test_fix_html(guide_manager: GuideManager):
    """Test that the fix_html method correctly fixes HTML page bugs.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    session = MagicMock()
    session.knowledge_points = [{"title": "kp1"}]
    session.current_index = 0
    guide_manager._load_session = MagicMock(return_value=session)
    guide_manager.interactive_agent.process = AsyncMock(
        return_value={"success": True, "html": "<html>"}
    )

    result = await guide_manager.fix_html("sid", "bug")

    assert result["success"] is True
    assert result["html"] == "<html>"

@pytest.mark.asyncio
async def test_save_and_load_session(guide_manager: GuideManager):
    """Test that sessions are saved and loaded correctly.

    Args:
        guide_manager (GuideManager): The GuideManager instance.
    """
    guide_manager.locate_agent.process = AsyncMock(
        return_value={"success": True, "knowledge_points": [{"title": "kp1"}]}
    )

    result = await guide_manager.create_session("nid", "nname", [])
    session_id = result["session_id"]

    # Clear cache to force loading from file
    guide_manager._sessions = {}

    loaded_session = guide_manager._load_session(session_id)
    assert loaded_session is not None
    assert loaded_session.session_id == session_id
