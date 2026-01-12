from contextlib import asynccontextmanager
import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    agent_config,
    chat,
    co_writer,
    dashboard,
    embedding_provider,
    guide,
    ideagen,
    knowledge,
    llm_provider,
    notebook,
    question,
    research,
    settings,
    solve,
    system,
)
from src.logging import get_logger

logger = get_logger("API")


def validate_cors_origin(origin: str) -> bool:
    """
    Validate that a CORS origin is a well-formed origin URL:
    - Must use http:// or https://
    - Must have a hostname
    - Must not include userinfo, query string, fragment, or non-root path
    """
    if not origin:
        return False

    try:
        # Strip whitespace to avoid accepting values with leading/trailing spaces
        origin = origin.strip()
        parsed = urlparse(origin)

        # Require http/https scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Require a hostname (netloc might include port, which is allowed)
        if not parsed.hostname:
            return False

        # Disallow userinfo (username:password@host)
        if parsed.username is not None or parsed.password is not None:
            return False

        # Disallow query, fragment, and params
        if parsed.query or parsed.fragment or parsed.params:
            return False

        # For CORS origins, path must be empty or root
        if parsed.path not in ("", "/"):
            return False

        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management
    Gracefully handle startup and shutdown events, avoid CancelledError
    """
    # Execute on startup
    logger.info("Application startup")
    yield
    # Execute on shutdown
    logger.info("Application shutdown")


app = FastAPI(title="DeepTutor API", version="1.0.0", lifespan=lifespan)

# Configure CORS
# The `ALLOWED_ORIGINS` environment variable controls which origins are allowed to
# call this API from a browser via CORS. It should be a comma-separated list of
# origins, for example:
# In production, always set `ALLOWED_ORIGINS` explicitly to the exact origins
# that should be able to access the API.
origins_str = os.getenv("ALLOWED_ORIGINS", "")
if origins_str.strip():
    origins_list = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    # Validate each origin
    valid_origins = []
    invalid_origins = []
    for origin in origins_list:
        if validate_cors_origin(origin):
            valid_origins.append(origin)
        else:
            invalid_origins.append(origin)

    if invalid_origins:
        logger.warning(
            f"Invalid CORS origins found and ignored: {', '.join(invalid_origins)}. "
            "Origins must be valid URLs starting with http:// or https://"
        )

    allow_origins = valid_origins
    logger.info("CORS configured from ALLOWED_ORIGINS for %d valid origin(s)", len(allow_origins))
else:
    # Restrictive default for local development: allow only localhost origins when
    # ALLOWED_ORIGINS is not set. For non-local deployments, set ALLOWED_ORIGINS
    # to the appropriate domain(s) instead of relying on this default.
    allow_origins = [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    logger.warning(
        "ALLOWED_ORIGINS is not set; using restrictive localhost-only CORS defaults. "
        "Set ALLOWED_ORIGINS to a comma-separated list of allowed origins for your "
        "deployment environment."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
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

# Include routers
app.include_router(solve.router, prefix="/api/v1", tags=["solve"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(question.router, prefix="/api/v1/question", tags=["question"])
app.include_router(research.router, prefix="/api/v1/research", tags=["research"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(co_writer.router, prefix="/api/v1/co_writer", tags=["co_writer"])
app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["notebook"])
app.include_router(guide.router, prefix="/api/v1/guide", tags=["guide"])
app.include_router(ideagen.router, prefix="/api/v1/ideagen", tags=["ideagen"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(llm_provider.router, prefix="/api/v1/config/llm", tags=["config"])
app.include_router(embedding_provider.router, prefix="/api/v1/config/embedding", tags=["config"])
app.include_router(agent_config.router, prefix="/api/v1/config", tags=["config"])


@app.get("/")
async def root():
    return {"message": "Welcome to DeepTutor API"}


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

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

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=backend_port,
        reload=True,
        reload_excludes=reload_excludes,
    )
