# tests/agents/chat/test_session_manager.py
import json
import time
import pytest
from unittest.mock import MagicMock, patch
from src.agents.chat.session_manager import SessionManager, RedisSessionManager

# --- File-Based Session Manager Tests ---

@pytest.fixture
def session_manager(tmp_path):
    """Create a SessionManager using a temporary directory."""
    return SessionManager(base_dir=str(tmp_path))

def test_create_session(session_manager):
    session = session_manager.create_session(title="Test Chat", settings={"rag": True})
    
    assert session["title"] == "Test Chat"
    assert session["settings"]["rag"] is True
    assert "session_id" in session
    assert len(session["messages"]) == 0

def test_add_message_and_persistence(session_manager):
    # 1. Create Session
    session = session_manager.create_session(title="Persist Test")
    sid = session["session_id"]
    
    # 2. Add Messages
    session_manager.add_message(sid, "user", "Hello World")
    updated_session = session_manager.add_message(sid, "assistant", "Hi there")
    
    # 3. Verify Memory State
    assert len(updated_session["messages"]) == 2
    assert updated_session["messages"][0]["content"] == "Hello World"
    
    # 4. Verify Disk Persistence
    # Force a new manager instance to reload from disk
    new_manager = SessionManager(base_dir=str(session_manager.base_dir))
    loaded_session = new_manager.get_session(sid)
    
    assert loaded_session is not None
    assert len(loaded_session["messages"]) == 2

def test_concurrency_locking(session_manager, tmp_path):
    """Ensure FileLock prevents race conditions."""
    # We mock the FileLock to ensure it's actually being called
    with patch("src.agents.chat.session_manager.FileLock") as mock_lock:
        session_manager.create_session("Concurrent Test")
        # Ensure lock was acquired at least once
        assert mock_lock.called
        assert mock_lock.return_value.__enter__.called

# --- Redis Session Manager Tests ---

@pytest.fixture
def mock_redis():
    with patch("src.agents.chat.session_manager.Redis") as mock:
        yield mock.from_url.return_value

def test_redis_create_session(mock_redis):
    # Setup Pipeline Mock
    pipeline = MagicMock()
    mock_redis.pipeline.return_value = pipeline
    
    manager = RedisSessionManager("redis://localhost")
    session = manager.create_session("Redis Test")
    
    # Verify Pipeline Execution
    assert pipeline.set.called
    assert pipeline.zadd.called
    assert pipeline.execute.called
    assert session["title"] == "Redis Test"

def test_redis_add_message_optimistic_lock(mock_redis):
    manager = RedisSessionManager("redis://localhost")
    sid = "test_session_123"
    
    # Mock existing session data in Redis
    existing_session = {
        "session_id": sid,
        "messages": [],
        "updated_at": 1000
    }
    mock_redis.get.return_value = json.dumps(existing_session)
    
    # Setup Pipeline
    pipeline = MagicMock()
    mock_redis.pipeline.return_value = pipeline
    
    # Action
    manager.add_message(sid, "user", "New Message")
    
    # Assertions
    mock_redis.watch.assert_called_with(manager._session_key(sid))
    assert pipeline.multi.called
    assert pipeline.execute.called