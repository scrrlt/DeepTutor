#!/usr/bin/env bash
set -euo pipefail

# Run pip-audit against requirements.txt and emit JSON to stdout
# Usage: scripts/deps/pip_audit.sh > pip-audit.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$ROOT_DIR/requirements.txt"

if ! command -v pip-audit >/dev/null 2>&1; then
  python -m pip install --upgrade pip >/dev/null
  python -m pip install pip-audit >/dev/null
fi

python -m pip_audit -r "$REQ_FILE" -f json
