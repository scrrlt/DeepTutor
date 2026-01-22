#!/usr/bin/env python
"""
LightRAG Log Forwarder
======================

Forwards LightRAG and RAG-Anything logs to DeepTutor's unified logging system.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

from .._stdlib_logging import stdlib_logging


class LightRAGLogForwarder(stdlib_logging.Handler):
    """
    Handler that forwards LightRAG logger messages to DeepTutor logger.
    """

    def __init__(self, ai_tutor_logger: logging.Logger, add_prefix: bool = True):
        """
        Args:
            ai_tutor_logger: DeepTutor Logger instance.
            add_prefix: Whether to add [LightRAG] prefix to messages.
        """
        super().__init__()
        self.ai_tutor_logger = ai_tutor_logger
        self.add_prefix = add_prefix
        self.setLevel(stdlib_logging.DEBUG)

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        """Forward log record to DeepTutor logger with proper level mapping."""
        try:
            message = record.getMessage()
            level = record.levelno
            if level >= logging.ERROR:
                self.ai_tutor_logger.error(message)
            elif level >= logging.WARNING:
                self.ai_tutor_logger.warning(message)
            elif level >= logging.INFO:
                self.ai_tutor_logger.info(message)
            else:
                self.ai_tutor_logger.debug(message)
        except Exception:
            self.handleError(record)


def get_lightrag_forwarding_config() -> dict[str, Any]:
    """
    Load LightRAG forwarding configuration from config/main.yaml.

    Returns:
        Configuration dictionary with defaults if not found.
    """
    try:
        from src.services.config import load_config_with_main

        from ..config import get_global_log_level

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config = load_config_with_main("solve_config.yaml", project_root)
        logging_config = config.get("logging", {})
        level = get_global_log_level()

        return {
            "enabled": True,
            "min_level": level,
            "logger_names": logging_config.get(
                "rag_logger_names",
                {"knowledge_init": "RAG-Init", "rag_tool": "RAG"},
            ),
        }
    except Exception:
        return {
            "enabled": True,
            "min_level": "DEBUG",
            "logger_names": {"knowledge_init": "RAG-Init", "rag_tool": "RAG"},
        }


@contextmanager
def LightRAGLogContext(
    logger_name: str | None = None,
    scene: str | None = None,
):
    """
    Context manager for LightRAG log forwarding.

    Args:
        logger_name: Explicit logger name (overrides scene-based lookup).
        scene: Scene name ('knowledge_init' or 'rag_tool') for logger name lookup.
    """
    from ..logger import get_logger

    config = get_lightrag_forwarding_config()
    if not config.get("enabled", True):
        yield
        return

    try:
        debug_logger = get_logger("RAGForward")
        debug_logger.debug(
            "Setting up LightRAG log forwarding (scene=%s, logger_name=%s)",
            scene,
            logger_name,
        )
    except Exception:
        pass

    if logger_name is None:
        if scene:
            logger_names = config.get("logger_names", {})
            logger_name = logger_names.get(scene, "Main")
        else:
            logger_name = "Main"

    ai_tutor_logger = get_logger(logger_name)
    handler = LightRAGLogForwarder(ai_tutor_logger)

    target_logger = logging.getLogger("lightrag")
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)

    try:
        yield
    finally:
        target_logger.removeHandler(handler)
        handler.close()
