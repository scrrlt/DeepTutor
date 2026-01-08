#!/usr/bin/env bash
set -euo pipefail

# Run npm audit in web/ and emit JSON to stdout
# Usage: scripts/deps/npm_audit.sh > npm-audit.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_DIR="$ROOT_DIR/web"

pushd "$WEB_DIR" >/dev/null
npm audit --json
popd >/dev/null
