#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting DeepTutor Backend..."

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

APP_PORT="${PORT:-8001}"
echo "Starting uvicorn server on port ${APP_PORT}..."
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${APP_PORT}"
echo "Uvicorn server command finished. This line should not be reached if 'exec' worked."
