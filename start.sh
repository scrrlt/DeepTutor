#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting DeepTutor Backend..."

# Cloud Run provides $PORT. Default to 8001 for local/dev.
PORT_VALUE="${PORT:-8001}"

# Test imports first
echo "Testing imports..."
python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from src.api.main import app
    print('✅ App imported successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "Starting uvicorn server on port ${PORT_VALUE}..."
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT_VALUE}"
