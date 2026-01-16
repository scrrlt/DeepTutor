import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.logging import get_logger
from src.services.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = get_logger("API")


def get_safe_detail(exc: Exception) -> str | None:
    """
    Return a safe detail string for non-production environments.

    In production, return None to avoid leaking sensitive details.
    """
    environment = os.getenv("ENVIRONMENT", "").strip().lower()
    allowed = {"development", "dev", "local", "test", "testing"}
    if environment not in allowed:
        return None
    detail = str(exc).strip()
    return detail or None


def _normalize_status_code(value: object) -> int:
    """Normalize and validate an HTTP status code, defaulting to 500."""
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return 500
    if status_code < 100 or status_code > 599:
        return 500
    return status_code


def _include_routers(app: FastAPI) -> None:
    from src.api.routers import (
        agent_config,
        chat,
        co_writer,
        config,
        dashboard,
        guide,
        ideagen,
        knowledge,
        notebook,
        question,
        research,
        settings,
        solve,
        system,
    )

    app.include_router(solve.router, prefix="/api/v1", tags=["solve"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(
        question.router,
        prefix="/api/v1/question",
        tags=["question"],
    )
    app.include_router(
        research.router,
        prefix="/api/v1/research",
        tags=["research"],
    )
    app.include_router(
        knowledge.router,
        prefix="/api/v1/knowledge",
        tags=["knowledge"],
    )
    app.include_router(
        dashboard.router,
        prefix="/api/v1/dashboard",
        tags=["dashboard"],
    )
    app.include_router(
        co_writer.router,
        prefix="/api/v1/co_writer",
        tags=["co_writer"],
    )
    app.include_router(
        notebook.router,
        prefix="/api/v1/notebook",
        tags=["notebook"],
    )
    app.include_router(
        guide.router,
        prefix="/api/v1/guide",
        tags=["guide"],
    )
    app.include_router(
        ideagen.router,
        prefix="/api/v1/ideagen",
        tags=["ideagen"],
    )
    app.include_router(
        settings.router,
        prefix="/api/v1/settings",
        tags=["settings"],
    )
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["system"],
    )
    app.include_router(
        config.router,
        prefix="/api/v1/config",
        tags=["config"],
    )
    app.include_router(
        agent_config.router,
        prefix="/api/v1/agent-config",
        tags=["agent-config"],
    )

CONFIG_DRIFT_ERROR_TEMPLATE = (
    "Configuration Drift Detected: Tools {drift} found in agents.yaml "
    "investigate.valid_tools but missing from main.yaml solve.valid_tools. "
    "Add these tools to main.yaml solve.valid_tools or remove them from "
    "agents.yaml investigate.valid_tools."
)


def validate_tool_consistency() -> None:
    """
    Validate that the tools configured for agents are consistent with the main application
    configuration.

    This function loads the main configuration (``main.yaml``) and the agents configuration
    (``agents.yaml``) from the project root and compares:

    * ``solve.valid_tools`` in ``main.yaml``
    * ``investigate.valid_tools`` in ``agents.yaml``

    All tools referenced by agents must be present in the main configuration. If any tools are
    defined for agents that are not listed in the main configuration, a ``RuntimeError`` is
    raised describing the drift. The error is logged and re-raised, which causes the FastAPI
    application startup to fail when this function is called from the ``lifespan`` handler.

    Impact on startup
    ------------------
    This validation runs during application startup. Any configuration drift will:

    * Be logged as an error with details about the unknown tools.
    * Prevent the API from starting until the configuration is corrected.

    How to resolve configuration drift
    ----------------------------------
    If startup fails with a configuration drift error:

    1. Inspect the set of tools reported in the error message.
    2. Either:
       * Add the missing tools to ``solve.valid_tools`` in ``main.yaml``, **or**
       * Remove or rename the offending tools from ``investigate.valid_tools`` in ``agents.yaml``.
    3. Restart the application after updating the configuration files.

    Example of aligned configuration
    --------------------------------
    ``main.yaml``::

        solve:
          valid_tools:
            - web_search
            - code_execution

    ``agents.yaml``::

        investigate:
          valid_tools:
            - web_search

    In this case, validation passes because ``investigate.valid_tools`` is a subset of
    ``solve.valid_tools``.

    Example of configuration drift
    ------------------------------
    ``agents.yaml``::

        investigate:
          valid_tools:
            - web_search
            - unknown_tool

    Here, ``unknown_tool`` is not present in ``solve.valid_tools`` in ``main.yaml``, so
    validation will fail and prevent the application from starting until the configurations
    are aligned.
    """
    try:
        from src.services.config import load_config_with_main

        project_root = Path(__file__).parent.parent.parent
        main_config = load_config_with_main("main.yaml", project_root)
        agent_config_data = load_config_with_main("agents.yaml", project_root)

        main_tools = set(
            main_config.get("solve", {}).get("valid_tools", [])
        )
        agent_tools = set(
            agent_config_data.get("investigate", {}).get("valid_tools", [])
        )

        if not agent_tools.issubset(main_tools):
            drift = agent_tools - main_tools
            raise RuntimeError(CONFIG_DRIFT_ERROR_TEMPLATE.format(drift=drift))
    except RuntimeError:
        logger.exception("Configuration validation failed")
        raise
    except Exception:
        logger.exception("Failed to load configuration for validation")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle management
    Gracefully handle startup and shutdown events, avoid CancelledError
    """
    # Execute on startup
    logger.info("Application startup")

    # Validate configuration consistency
    validate_tool_consistency()

    _include_routers(app)

    yield

    # Execute on shutdown
    logger.info("Application shutdown")

    # Close shared HTTP client to release connections
    try:
        from src.services.llm.http_client import close_shared_http_client

        await close_shared_http_client()
        logger.info("Shared HTTP client closed successfully")
    except Exception as e:
        logger.error(
            f"Error closing shared HTTP client: {e}"
        )


app = FastAPI(
    title="DeepTutor API",
    version="1.0.0",
    lifespan=lifespan,
    # Disable automatic trailing slash redirects to prevent protocol downgrade issues
    # when deployed behind HTTPS reverse proxies (e.g., nginx).
    # Without this, FastAPI's 307 redirects may change HTTPS to HTTP.
    # See: https://github.com/HKUDS/DeepTutor/issues/112
    redirect_slashes=False,
)


# Global exception handlers for proper HTTP status codes
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """
    Handle LLM-related errors with appropriate HTTP status codes.

    Circuit breaker errors return 503 Service Unavailable to allow clients
    to handle backpressure correctly. Other LLM errors are mapped based on
    their type (rate limit -> 429, auth -> 401, etc.).

    Args:
        request: The incoming HTTP request.
        exc: The raised LLMError instance.

    Returns:
        JSONResponse describing the error with the appropriate status code.

    Raises:
        LLMError: Re-raised when the error type is not handled explicitly.
    """
    # Circuit breaker - return 503 Service Unavailable
    if isinstance(exc, LLMError) and getattr(exc, "is_circuit_breaker", False):
        logger.warning(f"Circuit breaker triggered: {exc}")
        content = {
            "error": "service_unavailable",
            "message": (
                "Circuit breaker is open; the LLM service is temporarily "
                "unavailable. Please try again later."
            ),
        }
        safe_detail = get_safe_detail(exc)
        if safe_detail:
            content["detail"] = safe_detail
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=content,
        )
    
    # Rate limit errors - return 429 Too Many Requests
    if isinstance(exc, LLMRateLimitError):
        logger.warning(f"Rate limit exceeded: {exc}")
        content = {
            "error": "rate_limit_exceeded",
            "message": "Rate limit exceeded. Please try again later.",
        }
        safe_detail = get_safe_detail(exc)
        if safe_detail:
            content["detail"] = safe_detail
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
        )
    
    # Authentication errors - return 401 Unauthorized
    if isinstance(exc, LLMAuthenticationError):
        logger.error(f"Authentication error: {exc}")
        content = {
            "error": "authentication_failed",
            "message": (
                "LLM authentication failed. "
                "Please check your API credentials."
            ),
        }
        safe_detail = get_safe_detail(exc)
        if safe_detail:
            content["detail"] = safe_detail
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=content,
        )
    
    # Timeout errors - return 504 Gateway Timeout
    if isinstance(exc, LLMTimeoutError):
        logger.warning(f"LLM timeout: {exc}")
        content = {
            "error": "gateway_timeout",
            "message": "The LLM request timed out. Please try again.",
        }
        safe_detail = get_safe_detail(exc)
        if safe_detail:
            content["detail"] = safe_detail
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=content,
        )
    
    # Generic LLM API errors - map based on status code
    if isinstance(exc, LLMAPIError):
        status_code = _normalize_status_code(getattr(exc, "status_code", None))
        logger.error(
            f"LLM API error (status {status_code}): {exc}"
        )
        content = {
            "error": "llm_api_error",
            "message": "An error occurred while communicating with the LLM.",
        }
        safe_detail = get_safe_detail(exc)
        if safe_detail:
            content["detail"] = safe_detail
        return JSONResponse(
            status_code=status_code,
            content=content,
        )
    
    # Re-raise for other exceptions to be handled by default handler
    raise exc


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle uncaught exceptions with a sanitized 500 response.

    HTTPException instances preserve their status code and detail while
    other exceptions return a generic internal server error message.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        JSONResponse carrying a sanitized error payload.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    logger.error(f"Unhandled exception: {exc}")
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "An unexpected error occurred. Please try again later."

    if isinstance(exc, HTTPException):
        status_code = _normalize_status_code(exc.status_code)
        message = str(exc.detail or message)

    content = {
        "error": "internal_server_error",
        "message": message,
    }
    safe_detail = get_safe_detail(exc)
    if safe_detail:
        content["detail"] = safe_detail

    return JSONResponse(status_code=status_code, content=content)


_original_add_exception_handler = FastAPI.add_exception_handler


def _add_exception_handler_with_default(
    self: FastAPI,
    exc_class_or_status_code: object,
    handler: object,
) -> object:
    """
    Ensure a default exception handler is always registered.

    Args:
        exc_class_or_status_code: Exception class or HTTP status code.
        handler: The handler callable to register.

    Returns:
        The result from FastAPI.add_exception_handler.
    """
    if Exception not in self.exception_handlers:
        self.exception_handlers[Exception] = generic_exception_handler
    if ValueError not in self.exception_handlers:
        self.exception_handlers[ValueError] = generic_exception_handler
    return _original_add_exception_handler(
        self,
        exc_class_or_status_code,
        handler,
    )


if getattr(
    FastAPI.add_exception_handler,
    "_deep_tutor_patched",
    False,
) is False:
    FastAPI.add_exception_handler = _add_exception_handler_with_default
    FastAPI.add_exception_handler._deep_tutor_patched = True


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount user directory as static root for generated artifacts
# This allows frontend to access generated artifacts (images, PDFs, etc.)
# URL: /api/outputs/solve/solve_xxx/artifacts/image.png
# Physical Path: DeepTutor/data/user/solve/solve_xxx/artifacts/image.png
project_root = Path(__file__).parent.parent.parent
user_dir = project_root / "data" / "user"

# Initialize user directories on startup
try:
    from src.services.setup import init_user_directories

    init_user_directories(project_root)
except Exception:
    # Fallback: just create the main directory if it doesn't exist
    if not user_dir.exists():
        user_dir.mkdir(parents=True)

app.mount("/api/outputs", StaticFiles(directory=str(user_dir)), name="outputs")

@app.get("/")
async def root():
    return {"message": "Welcome to DeepTutor API"}


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    # Ensure project root is in Python path
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Get port from configuration
    from src.services.setup import get_backend_port

    backend_port = get_backend_port(project_root)

    # Configure reload_excludes with absolute paths to properly exclude directories
    venv_dir = project_root / "venv"
    data_dir = project_root / "data"
    reload_excludes = [
        str(d)
        for d in [
            venv_dir,
            project_root / ".venv",
            data_dir,
            project_root / "web" / "node_modules",
            project_root / "web" / ".next",
            project_root / ".git",
        ]
        if d.exists()
    ]

    # Bind to localhost only for the built-in development server to avoid
    # unintentionally exposing the API on the local network. Production
    # deployments should configure the host/interface explicitly.
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=backend_port,
        reload=True,
        reload_excludes=reload_excludes,
    )
