# tests/performance/oom_repro/Dockerfile
#
# Purpose:
#   Minimal, safe image for OOM reproduction and ingestion diagnostics.
#
# IMPORTANT DEPENDENCY POLICY:
# ------------------------------------------------------------
# This image intentionally EXCLUDES the following libraries:
#   - Pillow
#   - reportlab
#
# These libraries are used ONLY for CI tooling (synthetic PDF generation)
# and MUST NOT be present in production or runtime images.
#
# Why?
#   - They introduce native memory allocators (fonts, images, caches)
#   - They make RSS behavior non-deterministic
#   - They obscure ingestion-related memory signals
#   - They increase attack surface and CVE churn
#
# If you believe one of these libraries is required at runtime,
# STOP and document the exact production code path that needs it.
# Otherwise, add it to requirements-dev.txt ONLY.
# ------------------------------------------------------------

FROM python:3.10-slim

LABEL app="deeptutor-oom-repro"
LABEL version="1.0"
LABEL description="OOM reproduction harness (runtime-only dependencies)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ------------------------------------------------------------
# System dependencies (minimal)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------
# Non-root user for safety
# ------------------------------------------------------------
RUN useradd -m -u 1000 tester && chown -R tester:tester /app

# ------------------------------------------------------------
# Copy runtime files only
# ------------------------------------------------------------
COPY requirements.txt ./
COPY src/ src/
COPY tests/performance/oom_repro/ tests/performance/oom_repro/

# ------------------------------------------------------------
# Install runtime Python dependencies ONLY
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir psutil pdfplumber

# ------------------------------------------------------------
# Prepare writable logs directory
# ------------------------------------------------------------
RUN mkdir -p /app/tests/performance/oom_repro/logs \
    && chown -R tester:tester /app/tests/performance/oom_repro

USER tester

# ------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import psutil; print('OK')" || exit 1

# ------------------------------------------------------------
# Default entrypoint
# ------------------------------------------------------------
CMD ["python3", "tests/performance/oom_repro/ingest_isolated.py", "--help"]
