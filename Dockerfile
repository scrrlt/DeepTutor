# ============================================
# DeepTutor Multi-Stage Dockerfile
# ============================================
# This Dockerfile builds a production-ready image for DeepTutor
# containing both the FastAPI backend and Next.js frontend
#
# Build: docker compose build
# Run:   docker compose up -d
#
# Prerequisites:
#   1. Copy .env.example to .env and configure your API keys
#   2. Optionally customize config/main.yaml
# ============================================

# ============================================
# Stage 1: Frontend Builder
# ============================================
FROM node:22-slim AS frontend-builder

WORKDIR /app/web

# Accept build argument for backend port
ARG BACKEND_PORT=8001

# Copy package files first for better caching
COPY web/package.json web/package-lock.json* ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy frontend source code
COPY web/ ./

# Create .env.local with a sensible default
# Runtime config will override this via publicRuntimeConfig during `next start`
RUN echo "NEXT_PUBLIC_API_BASE=http://localhost:${BACKEND_PORT}" > .env.local

# Build Next.js for production with standalone output
# This allows runtime environment variable injection
RUN npm run build

# ============================================
# Stage 2: Python Base with Dependencies
# ============================================
FROM python:3.11-slim AS python-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
# Note: libgl1 and libglib2.0-0 are required for OpenCV (used by mineru)
# libmagic1 is for file type detection, poppler-utils for PDF processing
# Rust is required for building tiktoken and other packages without pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    poppler-utils \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================
# Stage 3: Production Image
# ============================================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="DeepTutor Team" \
      description="DeepTutor: AI-Powered Personalized Learning Assistant" \
      version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    NODE_ENV=production \
    # Default ports (can be overridden)
    BACKEND_PORT=8001 \
    FRONTEND_PORT=3782

WORKDIR /app

# Install system dependencies
# Note: libgl1 and libglib2.0-0 are required for OpenCV (used by mineru)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bash \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js from frontend-builder stage (avoids re-downloading from NodeSource)
COPY --from=frontend-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend-builder /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version && npm --version

# Copy Python packages from builder stage
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-base /usr/local/bin /usr/local/bin

# Copy built frontend from frontend-builder stage
COPY --from=frontend-builder /app/web/.next ./web/.next
COPY --from=frontend-builder /app/web/public ./web/public
COPY --from=frontend-builder /app/web/package.json ./web/package.json
COPY --from=frontend-builder /app/web/next.config.js ./web/next.config.js
COPY --from=frontend-builder /app/web/node_modules ./web/node_modules

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY pyproject.toml ./
COPY requirements.txt ./

# Create necessary directories (these will be overwritten by volume mounts)
RUN mkdir -p \
    data/user/solve \
    data/user/question \
    data/user/research/cache \
    data/user/research/reports \
    data/user/guide \
    data/user/notebook \
    data/user/co-writer/audio \
    data/user/co-writer/tool_calls \
    data/user/logs \
    data/user/run_code_workspace \
    data/user/performance \
    data/knowledge_bases

# Copy startup script (normalize line endings and make executable)
COPY start.sh ./start.sh
RUN sed -i 's/\r$//' ./start.sh && chmod +x ./start.sh

# Create supervisord configuration for running both services
# Log output goes to stdout/stderr so docker logs can capture them
RUN mkdir -p /etc/supervisor/conf.d

RUN cat > /etc/supervisor/conf.d/deeptutor.conf <<'EOF'
[supervisord]
nodaemon=true
logfile=/dev/null
logfile_maxbytes=0
pidfile=/var/run/supervisord.pid

[program:backend]
command=/bin/bash /app/start-backend.sh
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=PYTHONPATH="/app",PYTHONUNBUFFERED="1"

[program:frontend]
command=/bin/bash /app/start-frontend.sh
directory=/app/web
autostart=false
autorestart=false
startsecs=5
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=NODE_ENV="production"
EOF

RUN sed -i 's/\r$//' /etc/supervisor/conf.d/deeptutor.conf

# Create backend startup script
RUN cat > /app/start-backend.sh <<'EOF'
#!/bin/bash
set -e

BACKEND_PORT=${PORT:-${BACKEND_PORT:-8001}}

echo "[Backend]  🚀 Starting FastAPI backend on port ${BACKEND_PORT}..."

# Run uvicorn via bash wrapper to ensure consistent environment across platforms
exec /bin/bash -c "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${BACKEND_PORT}"
EOF

RUN sed -i 's/\r$//' /app/start-backend.sh && chmod +x /app/start-backend.sh

# Create frontend startup script
# This script handles runtime environment variable injection for Next.js
RUN cat > /app/start-frontend.sh <<'EOF'
#!/bin/bash
set -e

# Get the backend port (default to 8001)
BACKEND_PORT=${BACKEND_PORT:-8001}
FRONTEND_PORT=${FRONTEND_PORT:-3782}

# Determine the API base URL with multiple fallback options
# Priority: NEXT_PUBLIC_API_BASE_EXTERNAL > NEXT_PUBLIC_API_BASE > auto-detect
if [ -n "$NEXT_PUBLIC_API_BASE_EXTERNAL" ]; then
    # Explicit external URL for cloud deployments
    API_BASE="$NEXT_PUBLIC_API_BASE_EXTERNAL"
    echo "[Frontend] 📌 Using external API URL: ${API_BASE}"
elif [ -n "$NEXT_PUBLIC_API_BASE" ]; then
    # Custom API base URL
    API_BASE="$NEXT_PUBLIC_API_BASE"
    echo "[Frontend] 📌 Using custom API URL: ${API_BASE}"
else
    # Default: localhost with configured backend port
    # Note: This only works for local development, not cloud deployments
    API_BASE="http://localhost:${BACKEND_PORT}"
    echo "[Frontend] 📌 Using default API URL: ${API_BASE}"
    echo "[Frontend] ⚠️  For cloud deployment, set NEXT_PUBLIC_API_BASE_EXTERNAL to your server's public URL"
    echo "[Frontend]    Example: -e NEXT_PUBLIC_API_BASE_EXTERNAL=https://your-server.com:${BACKEND_PORT}"
fi

echo "[Frontend] 🚀 Starting Next.js frontend on port ${FRONTEND_PORT}..."

export NEXT_PUBLIC_API_BASE="${API_BASE}"
echo "NEXT_PUBLIC_API_BASE=${API_BASE}" > /app/web/.env.local

# Start Next.js with runtime config sourced from environment
cd /app/web && exec NEXT_PUBLIC_API_BASE="${API_BASE}" \
    node node_modules/next/dist/bin/next start -H 0.0.0.0 -p ${FRONTEND_PORT}
EOF

RUN sed -i 's/\r$//' /app/start-frontend.sh && chmod +x /app/start-frontend.sh


# Expose ports
EXPOSE 8001 3782

# Override ENTRYPOINT for Cloud Run (single process on $PORT)
ENTRYPOINT []

# Start backend (Cloud Run expects the container to listen on $PORT)
CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]

# ============================================
# Stage 4: Development Image (Optional)
# ============================================
FROM production AS development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pre-commit \
    black \
    ruff

# Override supervisord config for development (with reload)
# Log output goes to stdout/stderr so docker logs can capture them
RUN cat > /etc/supervisor/conf.d/deeptutor.conf <<'EOF'
[supervisord]
nodaemon=true
logfile=/dev/null
logfile_maxbytes=0
pidfile=/var/run/supervisord.pid

[program:backend]
command=python -m uvicorn src.api.main:app --host 0.0.0.0 --port %(ENV_BACKEND_PORT)s --reload
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=PYTHONPATH="/app",PYTHONUNBUFFERED="1"

[program:frontend]
command=/bin/bash -c "cd /app/web && node node_modules/next/dist/bin/next dev -H 0.0.0.0 -p ${FRONTEND_PORT:-3782}"
directory=/app/web
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=NODE_ENV="development"
EOF

RUN sed -i 's/\r$//' /etc/supervisor/conf.d/deeptutor.conf

# Development ports
EXPOSE 8001 3782
=======
# Start the application
CMD ["bash", "./start.sh"]
>>>>>>> 9dfce02 (fix(cloud): normalize start.sh line endings + run via bash for Cloud Run)
