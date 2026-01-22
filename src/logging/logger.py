"""
Core Logger Implementation
==========================

Unified logging with consistent format across all modules.
Format: [Module] Symbol Message

Implementation uses a non-blocking QueueHandler to prevent event loop stalls
under high-load scenarios. The main thread pushes LogRecords to a queue, and a
separate listener thread handles actual I/O (file writes, console output).

Logging is configured once at application startup via configure_logging().
All subsequent get_logger() calls share the same listener and output handlers.
This prevents silent configuration failures and ensures clean lifecycle management.

Example outputs:
    [INFO]     [Solver]        Ready in 2.3s
    [INFO]     [Research]      Starting deep research...
    [INFO]     [Guide]         Compiling knowledge points
    [INFO]     [Knowledge]     Indexed 150 documents
    [ERROR]    [EmbeddingClient]  Embedding request failed
"""

import atexit
from datetime import datetime
from enum import Enum
import json
import logging
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
import sys
from typing import Any

from src.config.constants import PROJECT_ROOT

from ._stdlib_logging import stdlib_logging

# Module-level queue for non-blocking log submission (bounded to prevent OOM)
_log_queue: Queue = Queue(maxsize=10000)  # Max 10k records to prevent memory explosion
_listener: QueueListener | None = None
_configured: bool = False  # Track if logging is configured
_lazy_configured: bool = False  # Track if configured via lazy initialization
_default_service_prefix: str | None = None


def configure_logging(
    console_output: bool = True,
    file_output: bool = True,
    log_level: str = "INFO",
    log_dir: str | Path | None = None,
) -> None:
    """Configure logging once at application startup.

    MUST be called before any get_logger() calls. Subsequent calls are ignored.

    Args:
        console_output: Enable console output
        file_output: Enable file output (to log_dir)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Log directory (default: PROJECT_ROOT/data/user/logs)

    Raises:
        RuntimeError: If called more than once
    """
    global _listener, _configured, _lazy_configured

    # Allow configuration if it was previously only lazy-configured
    if _configured and not _lazy_configured:
        raise RuntimeError("logging.configure_logging() already called. Cannot reconfigure.")

    # If already running (from lazy), stop the old one before reconfiguration
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None

    handlers = []

    if console_output:
        console_handler = stdlib_logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(ConsoleFormatter(service_prefix=_default_service_prefix))
        handlers.append(console_handler)

    if file_output:
        log_dir_path: Path
        if log_dir is None:
            log_dir_path = PROJECT_ROOT / "data" / "user" / "logs"
        else:
            log_dir_path = Path(log_dir) if isinstance(log_dir, str) else log_dir
            if not log_dir_path.is_absolute():
                log_dir_path = PROJECT_ROOT / log_dir_path

        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir_path / f"ai_tutor_{timestamp}.log"

        file_handler = stdlib_logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(FileFormatter())
        handlers.append(file_handler)

    if handlers:
        _listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
        _listener.start()

    _configured = True
    _lazy_configured = False


def _cleanup_logging() -> None:
    """Cleanup handler called by atexit. Flushes and closes the listener."""
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None


# Register cleanup on interpreter exit to ensure logs are flushed
atexit.register(_cleanup_logging)


def set_default_service_prefix(prefix: str | None) -> None:
    """
    Set the default service prefix used by console formatters.

    Args:
        prefix: Prefix string to display before module tags, or None to disable.
    """
    global _default_service_prefix

    _default_service_prefix = prefix
    if _listener is None:
        return

    for handler in _listener.handlers:
        formatter = getattr(handler, "formatter", None)
        if isinstance(formatter, ConsoleFormatter):
            formatter.service_prefix = prefix


class LogLevel(Enum):
    """Log levels with standard tags"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PROGRESS = "PROGRESS"
    COMPLETE = "COMPLETE"


class ConsoleFormatter(stdlib_logging.Formatter):
    """
    Clean console formatter with colors and standard level tags.
    Format: [LEVEL]   [Module]  Message
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[37m",  # White
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "PROGRESS": "\033[36m",  # Cyan
        "COMPLETE": "\033[32m",  # Green
    }
    SYMBOLS = {
        "DEBUG": "·",
        "INFO": "●",
        "SUCCESS": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "CRITICAL": "✗",
        "PROGRESS": "→",
        "COMPLETE": "✓",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, service_prefix: str | None = None):
        """
        Initialize console formatter.

        Args:
            service_prefix: Optional service layer prefix (e.g., "Backend", "Frontend")
        """
        super().__init__()
        self.service_prefix = service_prefix
        # Check TTY status once during initialization
        stdout_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        stderr_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self.use_colors = stdout_tty or stderr_tty

    def format(self, record: stdlib_logging.LogRecord) -> str:
        module = getattr(record, "module_name", record.name)
        module_tag = f"[{module}]".ljust(14)
        display_level = getattr(record, "display_level", record.levelname)
        level_tag = f"[{display_level}]".ljust(10)
        symbol = getattr(
            record,
            "symbol",
            self.SYMBOLS.get(record.levelname, "●"),
        )

        if self.use_colors:
            color = self.COLORS.get(display_level, self.COLORS["INFO"])
            dim = self.DIM
            reset = self.RESET
        else:
            color = ""
            dim = ""
            reset = ""

        message = record.getMessage()

        if self.service_prefix:
            service_tag = f"[{self.service_prefix}]"
            prefix = f"{dim}{service_tag}{reset} "
        else:
            prefix = ""

        return (
            f"{prefix}{dim}{module_tag}{reset} "
            f"{color}{symbol}{reset} "
            f"{color}{level_tag}{reset} {message}"
        )


class FileFormatter(stdlib_logging.Formatter):
    """
    Detailed file formatter for log files.
    Format: TIMESTAMP [LEVEL] [Module] Message
    """

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)-8s] [%(module_name)-12s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: stdlib_logging.LogRecord) -> str:
        # Ensure module_name exists
        if not hasattr(record, "module_name"):
            record.module_name = record.name
        return super().format(record)


class Logger:
    """
    Unified logger for DeepTutor.

    Features:
    - Consistent format across all modules
    - Color-coded console output
    - File logging to user/logs/
    - WebSocket streaming support
    - Success/progress/complete convenience methods
    - Optional service layer prefix (Backend/Frontend)

    Usage:
        logger = Logger("Solver")
        logger.info("Processing...")
        logger.success("Done!", elapsed=2.3)
        logger.progress("Step 1/5")
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
    ):
        """Initialize logger.

        Args:
            name: Module name (e.g., "Solver", "Research", "Guide")
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            RuntimeError: If logging not configured via configure_logging()
        """
        if not _configured:
            raise RuntimeError(
                f"Logger({name}): logging not configured. Call configure_logging() "
                "at application startup before creating loggers."
            )

        self.name = name
        self.level = getattr(logging, level.upper(), stdlib_logging.INFO)

        # Create underlying Python logger
        self.logger = stdlib_logging.getLogger(f"ai_tutor.{name}")
        self.logger.setLevel(stdlib_logging.DEBUG)  # Capture all, filter at handlers
        self.logger.handlers.clear()

        # Use QueueHandler for non-blocking submission
        queue_handler = QueueHandler(_log_queue)
        queue_handler.setLevel(self.level)  # Filter per this logger's level
        self.logger.addHandler(queue_handler)

        # For backwards compatibility with task-specific logging
        self._task_handlers: list[stdlib_logging.Handler] = []

        # Display manager for TUI (optional, used by solve_agents)
        self.display_manager = None

    def add_task_log_handler(
        self,
        task_log_file: str,
        capture_stdout: bool = False,
        capture_stderr: bool = False,
    ):
        """
        Add a task-specific log file handler.
        For backwards compatibility with old SolveAgentLogger.

        Creates a separate QueueListener for this task to avoid blocking the main event loop.

        Args:
            task_log_file: Path to the task log file
            capture_stdout: Ignored (kept for API compatibility)
            capture_stderr: Ignored (kept for API compatibility)
        """
        task_path = Path(task_log_file)
        task_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a task-specific queue and listener to avoid blocking main loop
        task_queue = Queue(maxsize=1000)  # Smaller queue for task logs
        task_handler = stdlib_logging.FileHandler(task_log_file, encoding="utf-8")
        task_handler.setLevel(stdlib_logging.DEBUG)
        task_handler.setFormatter(FileFormatter())

        task_listener = QueueListener(task_queue, task_handler, respect_handler_level=True)
        task_listener.start()

        # Add QueueHandler to this logger that feeds the task-specific queue
        task_queue_handler = QueueHandler(task_queue)
        task_queue_handler.setLevel(stdlib_logging.DEBUG)  # Capture all task logs
        self.logger.addHandler(task_queue_handler)

        # Store both handler and listener for cleanup
        self._task_handlers.append((task_queue_handler, task_listener))

    def remove_task_log_handlers(self):
        """Remove all task-specific log handlers."""
        for handler, listener in self._task_handlers:
            self.logger.removeHandler(handler)
            handler.close()
            try:
                listener.stop()
            except Exception:
                pass
        self._task_handlers.clear()

    def log_stage_progress(self, stage: str, status: str, detail: str | None = None):
        """Backwards compatibility alias for stage()"""
        self.stage(stage, status, detail)

    def section(self, title: str, char: str = "=", length: int = 60):
        """Print a section header"""
        self.info(char * length)
        self.info(title)
        self.info(char * length)

    def _log(
        self,
        level: int,
        message: str,
        *args: object,
        display_level: str | None = None,
        **kwargs: object,
    ):
        """Internal logging method with extra attributes."""
        symbol = kwargs.get(
            "symbol",
            ConsoleFormatter.SYMBOLS.get(
                stdlib_logging.getLevelName(level),
                "●",
            ),
        )
        extra = {
            "module_name": self.name,
            "symbol": symbol,
            "display_level": display_level or stdlib_logging.getLevelName(level),
        }
        # Extract standard logging parameters from kwargs
        log_kwargs = {
            "extra": extra,
            "exc_info": kwargs.get("exc_info", False),
            "stack_info": kwargs.get("stack_info", False),
            "stacklevel": kwargs.get("stacklevel", 1) + 2,
        }
        self.logger.log(level, message, *args, **log_kwargs)

    # Standard logging methods
    def debug(self, message: str, *args: object, **kwargs: object):
        """Debug level log [DEBUG]"""
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args: object, **kwargs: object):
        """Info level log [INFO]"""
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: object, **kwargs: object):
        """Warning level log [WARNING]"""
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: object, **kwargs: object):
        """Error level log [ERROR]"""
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args: object, **kwargs: object):
        """Critical level log [CRITICAL]"""
        self._log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args: object, **kwargs: object):
        """Log exception with traceback"""
        # Ensure exc_info is True to print the stack trace
        kwargs.setdefault("exc_info", True)

        # Forward all kwargs (including stack_info, stacklevel) to the underlying logger
        self._log(logging.ERROR, message, *args, display_level="ERROR", **kwargs)

    # Convenience methods
    def success(
        self,
        message: str,
        *args: object,
        elapsed: float | None = None,
        **kwargs: object,
    ):
        """Success log with checkmark (✓)"""
        if elapsed is not None:
            message = f"{message} in {elapsed:.1f}s"
        self._log(
            stdlib_logging.INFO,
            message,
            *args,
            symbol="✓",
            display_level="SUCCESS",
            **kwargs,
        )

    def progress(self, message: str, *args: object, **kwargs: object):
        """Progress log with arrow (→)"""
        self._log(stdlib_logging.INFO, message, *args, symbol="→", **kwargs)

    def complete(self, message: str, *args: object, **kwargs: object):
        """Completion log with checkmark (✓)"""
        self._log(
            stdlib_logging.INFO,
            message,
            *args,
            symbol="✓",
            display_level="COMPLETE",
            **kwargs,
        )

    def stage(self, stage_name: str, status: str = "start", detail: str | None = None):
        """
        Log stage progress.

        Args:
            stage_name: Name of the stage (e.g., "Analysis", "Synthesis")
            status: One of "start", "running", "complete", "skip", "error"
            detail: Optional detail message
        """
        # Map status to display level
        status_to_level = {
            "start": "PROGRESS",
            "running": "INFO",
            "complete": "SUCCESS",
            "skip": "INFO",
            "error": "ERROR",
            "warning": "WARNING",
        }
        display_level = status_to_level.get(status, "INFO")

        message = f"{stage_name}"
        if status == "complete":
            message += " completed"
        elif status == "start":
            message += " started"
        elif status == "running":
            message += " running"
        elif status == "skip":
            message += " skipped"
        elif status == "error":
            message += " failed"

        if detail:
            message += f" | {detail}"

        level = stdlib_logging.ERROR if status == "error" else stdlib_logging.INFO
        display_level = (
            "ERROR" if status == "error" else ("SUCCESS" if status == "complete" else "INFO")
        )
        symbol = "✓" if status == "complete" else ("✗" if status == "error" else "●")
        self._log(level, message, symbol=symbol, display_level=display_level)

    def tool_call(
        self,
        tool_name: str,
        status: str = "success",
        elapsed_ms: float | None = None,
        **kwargs,
    ):
        """
        Log tool call.

        Args:
            tool_name: Name of the tool
            status: "success", "error", or "running"
            elapsed_ms: Execution time in milliseconds
        """
        symbol = "✓" if status == "success" else ("✗" if status == "error" else "●")
        display_level = (
            "SUCCESS" if status == "success" else ("ERROR" if status == "error" else "INFO")
        )

        message = f"Tool: {tool_name}"
        if elapsed_ms is not None:
            message += f" ({elapsed_ms:.0f}ms)"
        if status == "error":
            message += " [FAILED]"

        self._log(
            stdlib_logging.INFO if status != "error" else stdlib_logging.ERROR,
            message,
            display_level=display_level,
        )

    def llm_call(
        self,
        model: str,
        agent: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        elapsed: float | None = None,
        **kwargs,
    ):
        """
        Log LLM API call.

        Args:
            model: Model name
            agent: Agent making the call
            tokens_in: Input tokens
            tokens_out: Output tokens
            elapsed: Call duration in seconds
        """
        parts = [f"LLM: {model}"]
        if agent:
            parts.append(f"agent={agent}")
        if tokens_in is not None:
            parts.append(f"in={tokens_in}")
        if tokens_out is not None:
            parts.append(f"out={tokens_out}")
        if elapsed is not None:
            parts.append(f"{elapsed:.2f}s")

        message = " | ".join(parts)
        self._log(stdlib_logging.DEBUG, message, symbol="◆")

    def separator(self, char: str = "─", length: int = 50):
        """Print a separator line"""
        self.info(char * length)

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        status: str = "success",
        elapsed_ms: float | None = None,
        **kwargs,
    ):
        """
        Log a tool call with input/output details.
        Backwards compatible with old SolveAgentLogger.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input (logged to file only)
            tool_output: Tool output (logged to file only)
            status: "success", "error", or "running"
            elapsed_ms: Execution time in milliseconds
        """
        symbol = "✓" if status == "success" else ("✗" if status == "error" else "●")
        display_level = (
            "SUCCESS" if status == "success" else ("ERROR" if status == "error" else "INFO")
        )

        # Console message (brief)
        message = f"Tool: {tool_name}"
        if elapsed_ms is not None:
            message += f" ({elapsed_ms:.0f}ms)"
        if status == "error":
            message += " [FAILED]"

        self._log(
            stdlib_logging.INFO if status != "error" else stdlib_logging.ERROR,
            message,
            display_level=display_level,
        )

        # Debug log with full details (file only)
        if tool_input is not None:
            try:
                input_str = (
                    json.dumps(tool_input, ensure_ascii=False, indent=2)
                    if isinstance(tool_input, (dict, list))
                    else str(tool_input)
                )
                self.debug(f"Tool Input: {input_str[:500]}...")
            except Exception:
                pass
        if tool_output is not None:
            try:
                output_str = (
                    json.dumps(tool_output, ensure_ascii=False, indent=2)
                    if isinstance(tool_output, (dict, list))
                    else str(tool_output)
                )
                self.debug(f"Tool Output: {output_str[:500]}...")
            except Exception:
                pass

    def log_llm_input(
        self,
        agent_name: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Log LLM input (debug level, file only)"""
        self.debug(
            f"LLM Input [{agent_name}:{stage}] system={len(system_prompt)}chars, user={len(user_prompt)}chars"
        )

    def log_llm_output(
        self,
        agent_name: str,
        stage: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Log LLM output (debug level, file only)"""
        self.debug(f"LLM Output [{agent_name}:{stage}] response={len(response)}chars")

    def log_llm_call(
        self,
        model: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        agent_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
        level: str = "INFO",
    ):
        """
        Log complete LLM call with formatted output.

        Args:
            model: Model name
            stage: Stage name (e.g., "generate_question", "validate")
            system_prompt: System prompt content
            user_prompt: User prompt content
            response: LLM response content
            agent_name: Agent name (optional)
            input_tokens: Input token count (optional)
            output_tokens: Output token count (optional)
            cost: Estimated cost (optional)
            level: Log level ("DEBUG" for full details, "INFO" for summary)
        """
        # Build header
        header_parts = ["LLM-CALL"]
        if agent_name:
            header_parts.append(f"Agent: {agent_name}")
        header_parts.append(f"Stage: {stage}")
        header_parts.append(f"Model: {model}")
        header = " | ".join(header_parts)

        # Log at appropriate level
        log_level = stdlib_logging.DEBUG if level == "DEBUG" else stdlib_logging.INFO

        if level == "DEBUG":
            # Full detailed output
            self._log(log_level, header, symbol="◆")
            self._log(
                log_level,
                "┌─ Input ──────────────────────────────────────────────",
                symbol=" ",
            )
            self._log(
                log_level,
                (
                    f"System: {system_prompt[:200]}..."
                    if len(system_prompt) > 200
                    else f"System: {system_prompt}"
                ),
            )
            self._log(
                log_level,
                (
                    f"User: {user_prompt[:500]}..."
                    if len(user_prompt) > 500
                    else f"User: {user_prompt}"
                ),
            )
            self._log(
                log_level,
                "└──────────────────────────────────────────────────────",
                symbol=" ",
            )
            self._log(
                log_level,
                "┌─ Output ─────────────────────────────────────────────",
                symbol=" ",
            )
            self._log(
                log_level,
                f"{response[:1000]}..." if len(response) > 1000 else response,
                symbol=" ",
            )
            self._log(
                log_level,
                "└──────────────────────────────────────────────────────",
                symbol=" ",
            )

            # Token and cost info
            token_info_parts = []
            if input_tokens is not None:
                token_info_parts.append(f"in={input_tokens}")
            if output_tokens is not None:
                token_info_parts.append(f"out={output_tokens}")
            if input_tokens is not None and output_tokens is not None:
                token_info_parts.append(f"total={input_tokens + output_tokens}")
            if cost is not None:
                token_info_parts.append(f"cost=${cost:.6f}")

            if token_info_parts:
                self._log(
                    log_level,
                    f"[Tokens: {' '.join(token_info_parts)}]",
                    symbol=" ",
                )
        else:
            # Summary output
            token_info = ""  # nosec B105
            if input_tokens is not None and output_tokens is not None:
                token_info = f" | Tokens: in={input_tokens}, out={output_tokens}, total={input_tokens + output_tokens}"
            if cost is not None:
                token_info += f" | Cost: ${cost:.6f}"

            message = f"{header}{token_info}"
            self._log(log_level, message)

    def update_token_stats(self, summary: dict[str, Any]):
        """Update token statistics (for display manager compatibility)"""
        # Log token stats at debug level
        if summary:
            total_tokens = summary.get("total_tokens", 0)
            self.debug(f"Token Stats: {total_tokens} tokens")

    def shutdown(self):
        """
        Shut down this logger by cleaning up **all** attached handlers.
        Also stops the global QueueListener if this is the last logger.
        """
        global _listener
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        # Clean up task handlers
        self.remove_task_log_handlers()

        # If no more loggers exist, stop the listener
        if _listener and len(_loggers) <= 1:
            try:
                _listener.stop()
            except Exception:
                pass
            _listener = None


# Global logger registry - key is tuple of (name, level)
_loggers: dict[tuple[str, str], "Logger"] = {}


def get_logger(
    name: str = "Main",
    level: str = "INFO",
    log_dir: str | Path | None = None,
) -> Logger:
    """
    Get or create a logger instance.

    Args:
        name: Module name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Optional log directory override (ignored; use configure_logging).

    Returns:
        Logger instance

    Note:
        If logging is not configured via configure_logging(), defaults will be used.
        For production, always call configure_logging() at application startup.
    """
    global _loggers, _default_service_prefix

    # Simple cache key - only name and level matter since config is centralized
    cache_key = (name, level)

    if cache_key not in _loggers:
        # Lazy configuration for backwards compatibility
        if not _configured:
            configure_logging()  # Use defaults if not configured
            _lazy_configured = True

        _loggers[cache_key] = Logger(
            name=name,
            level=level,
        )

    # Return the underlying stdlib logger for compatibility with tests that
    # expect a logging.Logger instance. Attach convenience methods from our
    # Logger wrapper to keep the richer API available.
    wrapper = _loggers[cache_key]
    std_logger = wrapper.logger

    # Use the user-facing name for tests (keeps tests stable and readable)
    try:
        std_logger.name = name
    except Exception:
        pass

    # Attach convenience methods if not already present
    method_names = [
        "success",
        "progress",
        "complete",
        "stage",
        "tool_call",
        "llm_call",
        "separator",
        "log_tool_call",
        "add_task_log_handler",
        "remove_task_log_handlers",
        "shutdown",
    ]

    for m in method_names:
        if not hasattr(std_logger, m) and hasattr(wrapper, m):
            setattr(std_logger, m, getattr(wrapper, m))

    return std_logger


def reset_logger(name: str | None = None):
    """
    Reset logger(s).

    Args:
        name: Logger name to reset, or None to reset all
    """
    global _loggers

    if name is None:
        keys_to_remove = list(_loggers.keys())
    else:
        # Remove all loggers with the given name
        keys_to_remove = [key for key in _loggers.keys() if key[0] == name]

    for key in keys_to_remove:
        _loggers.pop(key, None)


def reload_loggers():
    """
    Reload configuration for all cached loggers.

    This method clears the logger cache, forcing recreation with current config
    on next get_logger() calls.
    """
    global _loggers
    _loggers.clear()
