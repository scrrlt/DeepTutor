###############
# Builder stage
###############
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies required for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency descriptors and optional wheel cache
COPY requirements.txt pyproject.toml ./
RUN mkdir -p build_cache/pip_wheels
COPY build_cache/pip_wheels/ ./build_cache/pip_wheels/

# Create a virtual environment with cached wheels when available
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir --prefer-binary --find-links build_cache/pip_wheels -r requirements.txt

###############
# Runtime stage
###############
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy pre-built dependencies from builder stage
COPY --from=builder /opt/venv /opt/venv

# Install runtime system packages only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Configure environment
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:$PYTHONPATH

# Copy application source and assets
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml ./
COPY requirements.txt ./

# Copy startup script and set permissions
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Create data directory and non-root user
RUN mkdir -p /app/data/user \
    && useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

USER app

# Expose service port
EXPOSE 8001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Default command
CMD ["./start.sh"]
