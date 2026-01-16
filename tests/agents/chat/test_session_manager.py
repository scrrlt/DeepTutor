#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the SessionManager.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.agents.chat.session_manager import SessionManager, RedisSessionManager


@pytest.fixture
def session_manager(tmp_path):
    """
    Provides a SessionManager instance with a temporary base directory.
    """
    manager = SessionManager(base_dir=str(tmp_path))
    yield manager


def test_session_manager_initialization(session_manager: SessionManager):
    """
    Tests that the SessionManager can be initialized correctly.
    """
    assert session_manager.base_dir.exists()
    assert session_manager.sessions_file.exists()


def test_create_session(session_manager: SessionManager):
    """
    Tests that the create_session method correctly creates a new session.
    """
    session = session_manager.create_session(title="Test Session")
    assert session["title"] == "Test Session"
    assert len(session_manager.list_sessions()) == 1


def test_get_session(session_manager: SessionManager):
    """
    Tests that the get_session method correctly retrieves a session.
    """
    session = session_manager.create_session(title="Test Session")
    retrieved_session = session_manager.get_session(session["session_id"])
    assert retrieved_session is not None
    assert retrieved_session["session_id"] == session["session_id"]


def test_update_session(session_manager: SessionManager):
    """
    Tests that the update_session method correctly updates a session.
    """
    session = session_manager.create_session(title="Test Session")
    session_manager.update_session(session["session_id"], title="Updated Title")
    updated_session = session_manager.get_session(session["session_id"])
    assert updated_session["title"] == "Updated Title"


def test_add_message(session_manager: SessionManager):
    """
    Tests that the add_message method correctly adds a message to a session.
    """
    session = session_manager.create_session(title="Test Session")
    session_manager.add_message(session["session_id"], role="user", content="Hello")
    updated_session = session_manager.get_session(session["session_id"])
    assert len(updated_session["messages"]) == 1
    assert updated_session["messages"][0]["content"] == "Hello"


def test_list_sessions(session_manager: SessionManager):
    """
    Tests that the list_sessions method correctly lists the sessions.
    """
    session_manager.create_session(title="Session 1")
    session_manager.create_session(title="Session 2")
    sessions = session_manager.list_sessions()
    assert len(sessions) == 2


def test_delete_session(session_manager: SessionManager):
    """
    Tests that the delete_session method correctly deletes a session.
    """
    session = session_manager.create_session(title="Test Session")
    session_manager.delete_session(session["session_id"])
    assert len(session_manager.list_sessions()) == 0


def test_clear_all_sessions(session_manager: SessionManager):
    """
    Tests that the clear_all_sessions method correctly deletes all sessions.
    """
    session_manager.create_session(title="Session 1")
    session_manager.create_session(title="Session 2")
    session_manager.clear_all_sessions()
    assert len(session_manager.list_sessions()) == 0


@patch.dict('os.environ', {'REDIS_URL': 'redis://localhost:6379/0'})
@patch('src.agents.chat.session_manager.Redis')
def test_redis_session_manager(mock_redis):
    """
    Tests that the RedisSessionManager works as expected.
    """
    mock_redis_instance = MagicMock()
    mock_redis.from_url.return_value = mock_redis_instance

    manager = RedisSessionManager(redis_url="redis://localhost:6379/0")

    # Mock Redis methods
    manager.redis.get.return_value = json.dumps({"session_id": "test_session", "title": "Test Session", "messages": [], "updated_at": 0})
    manager.redis.zrevrange.return_value = ["test_session"]

    session = manager.create_session(title="Test Session")
    assert session["title"] == "Test Session"

    retrieved_session = manager.get_session(session["session_id"])
    assert retrieved_session is not None
    
    manager.update_session(session["session_id"], title="Updated Title")
    
    manager.add_message(session["session_id"], role="user", content="Hello")
    
    sessions = manager.list_sessions()
    assert len(sessions) > 0

    manager.delete_session(session["session_id"])
    
    assert manager.redis.pipeline.call_count > 0
