"""
Unified Logging System for DeepTutor
=====================================

A clean, consistent logging system with:
- Unified format: [Module] Symbol Message
- English-only output
- File output to data/user/logs/
- WebSocket streaming support
- Color-coded console output
- LLM usage statistics tracking
- External library log forwarding (LightRAG, LlamaIndex)

Usage:
    from src.logging import get_logger, LLMStats

    logger = get_logger("Solver")
    logger.info("Processing started")
    logger.success("Task completed in 2.3s")
    logger.error("Something went wrong")

    # Track LLM usage
    stats = LLMStats("Solver")
    stats.add_call(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
    stats.print_summary()
"""

# Core logging
# Configuration
from .config import (
    LoggingConfig,
    get_default_log_dir,
    load_logging_config,
)

# Handlers
from .handlers import (
    ConsoleHandler,
    FileHandler,
    JSONFileHandler,
    LogInterceptor,
    RotatingFileHandler,
    WebSocketLogHandler,
)
from .logger import (
    ConsoleFormatter,
    FileFormatter,
    Logger,
    LogLevel,
    get_logger,
    reset_logger,
)

# Statistics tracking
from .stats import (
    MODEL_PRICING,
    LLMCall,
    LLMStats,
    estimate_tokens,
    get_pricing,
)

__all__ = [
    # Core
    "Logger",
    "LogLevel",
    "get_logger",
    "reset_logger",
    "ConsoleFormatter",
    "FileFormatter",
    # Handlers
    "ConsoleHandler",
    "FileHandler",
    "JSONFileHandler",
    "RotatingFileHandler",
    "WebSocketLogHandler",
    "LogInterceptor",
    # Adapters
    "LightRAGLogContext",
    "LightRAGLogForwarder",
    "get_lightrag_forwarding_config",
    "LlamaIndexLogContext",
    "LlamaIndexLogForwarder",
    # Stats
    "LLMStats",
    "LLMCall",
    "get_pricing",
    "estimate_tokens",
    "MODEL_PRICING",
    # Config
    "LoggingConfig",
    "load_logging_config",
    "get_default_log_dir",
]


def __getattr__(name):
    """Lazy import adapters to avoid circular import issues."""
    if name in ("LightRAGLogContext", "LightRAGLogForwarder", "get_lightrag_forwarding_config"):
        from .adapters import LightRAGLogContext, LightRAGLogForwarder, get_lightrag_forwarding_config
        if name == "LightRAGLogContext":
            return LightRAGLogContext
        elif name == "LightRAGLogForwarder":
            return LightRAGLogForwarder
        elif name == "get_lightrag_forwarding_config":
            return get_lightrag_forwarding_config
    elif name in ("LlamaIndexLogContext", "LlamaIndexLogForwarder"):
        from .adapters import LlamaIndexLogContext, LlamaIndexLogForwarder
        if name == "LlamaIndexLogContext":
            return LlamaIndexLogContext
        elif name == "LlamaIndexLogForwarder":
            return LlamaIndexLogForwarder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
