#!/usr/bin/env python

"""
Tests for the Chat API Router.
"""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.routers.chat import router as chat_router


@pytest.fixture
def client():
    """
    Provides a TestClient instance for the chat router.
    """
    app = FastAPI()
    app.include_router(chat_router)
    with TestClient(app) as c:
        yield c


@patch("src.api.routers.chat.session_manager")
def test_list_sessions(mock_session_manager, client):
    """
    Tests the list_sessions endpoint.
    """
    mock_session_manager.list_sessions.return_value = [{"session_id": "s1"}]
    response = client.get("/chat/sessions")
    assert response.status_code == 200
    assert response.json() == [{"session_id": "s1"}]


@patch("src.api.routers.chat.session_manager")
def test_get_session(mock_session_manager, client):
    """
    Tests the get_session endpoint.
    """
    mock_session_manager.get_session.return_value = {"session_id": "s1"}
    response = client.get("/chat/sessions/s1")
    assert response.status_code == 200
    assert response.json() == {"session_id": "s1"}

    mock_session_manager.get_session.return_value = None
    response = client.get("/chat/sessions/s2")
    assert response.status_code == 404


@patch("src.api.routers.chat.session_manager")
def test_delete_session(mock_session_manager, client):
    """
    Tests the delete_session endpoint.
    """
    mock_session_manager.delete_session.return_value = True
    response = client.delete("/chat/sessions/s1")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    mock_session_manager.delete_session.return_value = False
    response = client.delete("/chat/sessions/s2")
    assert response.status_code == 404
