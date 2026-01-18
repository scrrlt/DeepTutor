# tests/performance/oom_repro/Dockerfile
#
# Purpose: Minimal, safe image for OOM reproduction harness.
# Notes:
#  - Only required runtime files are copied into the image to keep it small and deterministic.
#  - Installs lightweight system deps and Python packages needed by the harness and CI synthetic PDF generator.
#  - Runs as non-root user and creates a writable logs directory.
#  - Healthcheck verifies Python/psutil availability.

FROM python:3.10-slim

LABEL app="deeptutor-oom-repro"
LABEL version="1.0"
LABEL description="Environment for reproducing OOM issues safely (ingest harness + monitor)."

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user for safety
RUN useradd -m -u 1000 tester && chown -R tester:tester /app

# Copy only the files needed for the harness and tests to keep the image small and deterministic.
# - requirements.txt controls Python deps for the application
# - src/ contains the minimal runtime code required by the harness
# - tests/performance/oom_repro/ contains harness scripts, monitor, and CI helpers
COPY requirements.txt ./
COPY src/ src/
COPY tests/performance/oom_repro/ tests/performance/oom_repro/

# Install Python dependencies.
# Keep psutil/pdfplumber/reportlab/Pillow pinned in requirements.txt where possible.
# We include them here as a fallback to ensure the image can run the harness and generate the synthetic PDF.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir psutil pdfplumber reportlab Pillow

# Prepare logs directory and ensure tester owns it
RUN mkdir -p /app/tests/performance/oom_repro/logs \
    && chown -R tester:tester /app/tests/performance/oom_repro

USER tester

# Lightweight healthcheck to ensure Python and psutil are available
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import psutil; print('OK')" || exit 1

# Default entrypoint shows help for the harness; callers override this to run a specific test
CMD ["python3", "tests/performance/oom_repro/ingest_isolated.py", "--help"]
