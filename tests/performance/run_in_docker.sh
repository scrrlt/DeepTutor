#!/usr/bin/env bash
# tests/performance/run_in_docker.sh
#
# Robust Docker runner for OOM reproduction.
# Fixes pathing for flat directory structure.

set -euo pipefail

# -----------------------------
# Preconditions
# -----------------------------
command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 1; }
# Check for uuidgen (some systems might use python fallback if missing, but let's stick to bash tools)
command -v uuidgen >/dev/null 2>&1 || { echo "uuidgen is required"; exit 1; }

# -----------------------------
# Args
# -----------------------------
# Usage: ./run_in_docker.sh <path_to_pdf> [memory_limit]
PDF_PATH=${1:-}
MEM_LIMIT=${2:-6g}
MEM_LIMIT=${OVERRIDE_MEM_LIMIT:-$MEM_LIMIT}

if [[ -z "$PDF_PATH" ]]; then
  echo "Usage: $0 <path_to_pdf> [memory_limit]"
  exit 1
fi

if [[ ! -f "$PDF_PATH" ]]; then
  echo "Error: File $PDF_PATH not found."
  exit 1
fi

# -----------------------------
# Paths & names
# -----------------------------
ROOT_DIR="$(pwd)"
# Fixed: Paths aligned to tests/performance/ structure
LOG_DIR="$ROOT_DIR/tests/performance/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MANIFEST="$LOG_DIR/manifest_$TIMESTAMP.json"

FILENAME="$(basename "$PDF_PATH")"
IMAGE_NAME="deeptutor-oom-repro:latest"
CONTAINER_NAME="oom_repro_$(uuidgen | cut -c1-8)"
CID_FILE="$(mktemp)"

mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR"

cleanup() {
  rm -f "$CID_FILE"
}
trap cleanup EXIT

# -----------------------------
# Info
# -----------------------------
echo "=== OOM Reproduction Setup ==="
echo "Target File: $FILENAME"
echo "Memory Cap:  $MEM_LIMIT"
echo "Logs Dir:    $LOG_DIR"
echo "Container:   $CONTAINER_NAME"
echo "=============================="

# -----------------------------
# Build image
# -----------------------------
echo "Building Docker image..."
# Fixed: Dockerfile is in tests/performance/Dockerfile
docker build -f tests/performance/Dockerfile -t "$IMAGE_NAME" .

# -----------------------------
# Command inside container
# -----------------------------
# Fixed: ingest_isolated.py is in /app/tests/performance/
CMD_STR="python3 tests/performance/ingest_isolated.py \
  --file /data/$FILENAME \
  --batch-size 10 \
  --output-dir /app/tests/performance/logs"

echo "Running Command inside Container:"
echo "$CMD_STR"

# -----------------------------
# Run container (NO --rm)
# -----------------------------
set +e
docker run \
  --name "$CONTAINER_NAME" \
  --cidfile "$CID_FILE" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEM_LIMIT" \
  --cpus="1.0" \
  -v "$(realpath "$PDF_PATH"):/data/$FILENAME:ro" \
  -v "$LOG_DIR:/app/tests/performance/logs" \
  "$IMAGE_NAME" \
  bash -c "$CMD_STR"
EXIT_CODE=$?
set -e

# -----------------------------
# Inspect container state
# -----------------------------
OOM_KILLED="unknown"
if [[ -f "$CID_FILE" ]]; then
  OOM_KILLED="$(docker inspect --format='{{.State.OOMKilled}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")"
fi

# -----------------------------
# Aggregate peak RSS across batches
# -----------------------------
PEAK_RSS="unknown"
if ls "$LOG_DIR"/batch_*/summary.json >/dev/null 2>&1; then
  PEAK_RSS="$(jq -s '[.[].peak_rss_mb] | max' "$LOG_DIR"/batch_*/summary.json)"
fi

# -----------------------------
# Write manifest
# -----------------------------
cat <<EOF > "$MANIFEST"
{
  "timestamp": "$TIMESTAMP",
  "pdf_file": "$FILENAME",
  "pdf_size_mb": $(du -m "$PDF_PATH" | cut -f1),
  "memory_cap": "$MEM_LIMIT",
  "exit_code": $EXIT_CODE,
  "oom_killed": "$OOM_KILLED",
  "peak_rss_mb": "$PEAK_RSS",
  "container_name": "$CONTAINER_NAME",
  "log_dir": "$LOG_DIR"
}
EOF

# -----------------------------
# Cleanup container
# -----------------------------
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Run complete."
echo "Manifest written to: $MANIFEST"