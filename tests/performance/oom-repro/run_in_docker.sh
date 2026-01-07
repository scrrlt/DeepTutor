#!/bin/bash
# tests/performance/oom_repro/run_in_docker.sh

set -euo pipefail

# Usage: ./run_in_docker.sh <path_to_pdf> [memory_limit]
PDF_PATH=$1
MEM_LIMIT=${2:-6g}
LOG_DIR="$(pwd)/tests/performance/oom_repro/logs"
MEM_LIMIT=${OVERRIDE_MEM_LIMIT:-$MEM_LIMIT}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MANIFEST="$LOG_DIR/manifest_$TIMESTAMP.json"

if [ -z "$PDF_PATH" ]; then
    echo "Usage: $0 <path_to_pdf> [memory_limit]"
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File $PDF_PATH not found."
    exit 1
fi

FILENAME=$(basename "$PDF_PATH")
IMAGE_NAME="deeptutor-oom-repro:latest"
CONTAINER_NAME="oom_repro_$(uuidgen | cut -c1-8)"

mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR"

echo "=== OOM Reproduction Setup ==="
echo "Target File: $FILENAME"
echo "Memory Cap:  $MEM_LIMIT"
echo "Logs Dir:    $LOG_DIR"
echo "=============================="

echo "Building Docker image..."
docker build -f tests/performance/oom_repro/Dockerfile -t $IMAGE_NAME .

CMD_STR="python3 tests/performance/oom_repro/ingest_isolated.py \
        --kb_name repro_run \
        --file /data/$FILENAME \
        --batch-size 10 \
        --max-bytes 5000000 \
        --output-dir /app/tests/performance/oom_repro/logs"

echo "Running Command inside Container:"
echo "$CMD_STR"

# Run container with memory cap and log mounts
docker run --rm --name "$CONTAINER_NAME" \
    --memory="$MEM_LIMIT" \
    --memory-swap="$MEM_LIMIT" \
    --cpus="1.0" \
    -v "$(realpath "$PDF_PATH"):/data/$FILENAME:ro" \
    -v "$LOG_DIR:/app/tests/performance/oom_repro/logs" \
    $IMAGE_NAME \
    bash -c "$CMD_STR"

EXIT_CODE=$?

# Post-run classification
OOM_KILLED=$(docker inspect --format='{{.State.OOMKilled}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
SUMMARY_JSON="$LOG_DIR/summary.json"

if [ -f "$SUMMARY_JSON" ]; then
    PEAK_RSS=$(jq -r '.peak_rss_mb' "$SUMMARY_JSON")
else
    PEAK_RSS="unknown"
fi

# Write manifest
cat <<EOF > "$MANIFEST"
{
  "timestamp": "$TIMESTAMP",
  "pdf_file": "$FILENAME",
  "pdf_size_mb": $(du -m "$PDF_PATH" | cut -f1),
  "memory_cap": "$MEM_LIMIT",
  "exit_code": $EXIT_CODE,
  "oom_killed": "$OOM_KILLED",
  "peak_rss_mb": "$PEAK_RSS",
  "log_dir": "$LOG_DIR"
}
EOF

echo "Run complete. Manifest written to $MANIFEST"