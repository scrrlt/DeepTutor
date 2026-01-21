"""
Chat API Router
================

WebSocket endpoint for lightweight chat with session management.
REST endpoints for session operations.
"""

from pathlib import Path
import sys
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.agents.chat import ChatAgent, SessionManager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.llm.config import get_llm_config

# Initialize logger
import logging as _logging
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
# Ensure we have at least a basic logging configuration in environments where none exists
if not _logging.getLogger().handlers:
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

logger = get_logger("ChatAPI", level="INFO", log_dir=log_dir)

router = APIRouter()

# Initialize session manager
session_manager = SessionManager()


# =============================================================================
# REST Endpoints for Session Management
# =============================================================================


@router.get("/chat/sessions")
async def list_sessions(limit: int = 20):
    """
    List recent chat sessions.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of session summaries
    """
    return session_manager.list_sessions(limit=limit, include_messages=False)


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get a specific chat session with full message history.

    Args:
        session_id: Session identifier

    Returns:
        Complete session data including messages
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session.

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    if session_manager.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# WebSocket Endpoint for Chat
# =============================================================================


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Handle a WebSocket chat connection that manages sessions, optional retrieval (RAG) and web search, and streams agent responses.
    
    Accepts JSON messages with fields:
    - message (str): user message (required)
    - session_id (str|None): existing session id or null to create a new session
    - history (list|None): optional explicit role/content history to override stored session history
    - kb_name (str): knowledge base name for RAG
    - enable_rag (bool): enable knowledge-base retrieval
    - enable_web_search (bool): enable web search
    
    Sends JSON messages to the client with these types:
    - {"type": "session", "session_id": str}: current or newly created session ID
    - {"type": "status", "stage": str, "message": str}: progress/status updates (e.g., "rag", "web", "generating")
    - {"type": "stream", "content": str}: incremental streaming chunks of the assistant response
    - {"type": "sources", "rag": list, "web": list}: citation sources from RAG and/or web search
    - {"type": "result", "content": str}: final complete assistant response
    - {"type": "error", "message": str}: user-facing error message
    
    Behavioral notes:
    - If a provided session_id is missing, a new session is created with a title derived from the message.
    - If explicit history is provided it replaces the session-stored history for the current request.
    - The assistant response is streamed to the client and the final response (and any sources) are persisted to the session.
    - Validation: an empty or missing "message" field results in an error message sent to the client.
    """
    await websocket.accept()

    # Get system language for agent
    language = config.get("system", {}).get("language", "en")

    try:
        llm_config = get_llm_config()
        api_key = llm_config.api_key
        base_url = llm_config.base_url
        api_version = getattr(llm_config, "api_version", None)
    except Exception as e:
        # Log configuration loading failure - system will continue with None values
        # but ChatAgent must handle missing credentials gracefully
        logger.warning("Failed to load LLM config: %s", e)
        api_key = None
        base_url = None
        api_version = None

    agent = ChatAgent(
        language=language,
        config=config,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )

    def _get_or_create_session(session_id: str | None, message: str, kb_name: str, enable_rag: bool, enable_web_search: bool):
        """
        Return an existing session for the given ID or create a new session configured with the provided settings.
        
        Parameters:
            session_id (str | None): Existing session ID, or None to force creation of a new session.
            message (str): User message used to derive a session title when creating a new session.
            kb_name (str): Knowledge-base name to store in the session settings.
            enable_rag (bool): Whether retrieval-augmented generation (RAG) is enabled for the session.
            enable_web_search (bool): Whether web search is enabled for the session.
        
        Returns:
            tuple: (session, session_id) where `session` is the session object and `session_id` is its ID.
        """
        if session_id:
            session = session_manager.get_session(session_id)
            if not session:
                # Session not found, create new one
                session = session_manager.create_session(
                    title=message[:50] + ("..." if len(message) > 50 else ""),
                    settings={
                        "kb_name": kb_name,
                        "enable_rag": enable_rag,
                        "enable_web_search": enable_web_search,
                    },
                )
                session_id = session["session_id"]
        else:
            # Create new session
            session = session_manager.create_session(
                title=message[:50] + ("..." if len(message) > 50 else ""),
                settings={
                    "kb_name": kb_name,
                    "enable_rag": enable_rag,
                    "enable_web_search": enable_web_search,
                },
            )
            session_id = session["session_id"]

        return session, session_id

    try:
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                message = data.get("message", "").strip()
                session_id = data.get("session_id")
                explicit_history = data.get("history")  # Optional override
                kb_name = data.get("kb_name", "")
                enable_rag = data.get("enable_rag", False)
                enable_web_search = data.get("enable_web_search", False)

                if not message:
                    await websocket.send_json({"type": "error", "message": "Message is required"})
                    continue

                logger.info(
                    f"Chat request: session={session_id}, "
                    f"message={message[:50]}..., rag={enable_rag}, web={enable_web_search}"
                )

                agent.refresh_config()

                session, session_id = _get_or_create_session(session_id, message, kb_name, enable_rag, enable_web_search)

                # Send session ID to frontend
                await websocket.send_json(
                    {
                        "type": "session",
                        "session_id": session_id,
                    }
                )

                # Build history from session or explicit override
                if explicit_history is not None:
                    history = explicit_history
                else:
                    # Get history from session messages
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in session.get("messages", [])
                    ]

                # Add user message to session
                session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                )

                # Send status updates
                if enable_rag and kb_name:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "stage": "rag",
                            "message": f"Searching knowledge base: {kb_name}...",
                        }
                    )

                if enable_web_search:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "stage": "web",
                            "message": "Searching the web...",
                        }
                    )

                await websocket.send_json(
                    {
                        "type": "status",
                        "stage": "generating",
                        "message": "Generating response...",
                    }
                )

                # Process with streaming
                full_response = ""
                sources: dict[str, list[Any]] = {"rag": [], "web": []}

                stream_generator = await agent.process(
                    message=message,
                    history=history,
                    kb_name=kb_name,
                    enable_rag=enable_rag,
                    enable_web_search=enable_web_search,
                    stream=True,
                )

                # Ensure stream_generator is iterable (handle both dict and AsyncGenerator return types)
                if hasattr(stream_generator, "__aiter__"):
                    async for chunk_data in stream_generator:
                        if chunk_data["type"] == "chunk":
                            await websocket.send_json(
                                {
                                    "type": "stream",
                                    "content": chunk_data["content"],
                                }
                            )
                            full_response += chunk_data["content"]
                        elif chunk_data["type"] == "complete":
                            full_response = chunk_data["response"]
                            sources = chunk_data.get("sources", {"rag": [], "web": []})

                # Send sources if any
                if sources.get("rag") or sources.get("web"):
                    await websocket.send_json({"type": "sources", **sources})

                # Send final result
                await websocket.send_json(
                    {
                        "type": "result",
                        "content": full_response,
                    }
                )

                # Save assistant message to session
                session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    sources=sources if (sources.get("rag") or sources.get("web")) else None,
                )

                logger.info(f"Chat completed: session={session_id}, {len(full_response)} chars")
            except Exception as exc:
                # Log internal exception with traceback for server-side diagnostics
                logger.warning("Chat message processing failed", exc_info=True)
                # Send a user-friendly error message to clients
                await websocket.send_json({"type": "error", "message": "An error occurred while processing your message. Please try again."})

    except WebSocketDisconnect:
        logger.debug("Client disconnected from chat")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass