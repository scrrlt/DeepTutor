# ===================================
# Stage 1: Build dependencies
# ===================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building wheels
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies into a target directory
# This keeps the final image clean
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# ===================================
# Stage 2: Final application image
# ===================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set Python path to include installed dependencies and project root.
# IMPORTANT: Do NOT add /app/src directly, otherwise src/logging shadows stdlib logging.
ENV PYTHONPATH=/app/deps:/app:$PYTHONPATH

# Copy installed dependencies from the builder stage
COPY --from=builder /app/deps /app/deps

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy startup script and make it executable
COPY start.sh ./start.sh
RUN sed -i 's/\r$//' ./start.sh && chmod +x ./start.sh

# Create data directory and non-root user
RUN mkdir -p /app/data/user \
    && useradd --create-home --shell /bin/sh app \
    && chown -R app:app /app
USER app

# Expose the application port
EXPOSE 8001

# Health check to ensure the application starts correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /bin/sh -c "python -c \"import os, urllib.request; port=os.environ.get('PORT','8001'); urllib.request.urlopen(f'http://localhost:{port}/health')\"" || exit 1

# Set the command to run the application
CMD ["sh", "./start.sh"]