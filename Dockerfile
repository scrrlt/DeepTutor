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

# Set Python path to include installed dependencies and src
ENV PYTHONPATH=/app/deps:/app/src:$PYTHONPATH

# Copy installed dependencies from the builder stage
COPY --from=builder /app/deps /app/deps

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy startup script and make it executable
COPY start.sh ./start.sh
RUN sed -i 's/\r$//' ./start.sh && chmod +x ./start.sh

# Create data directory
RUN mkdir -p /app/data/user

# Create a non-root user and switch to it
# This is a security best practice
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose the application port
EXPOSE 8001

# Health check to ensure the application starts correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Set the command to run the application
CMD ["sh", "./start.sh"]