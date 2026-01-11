#!/bin/bash
# Startup script for DeepTutor

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
    exit 1
"

# Start the server
echo "Starting uvicorn server..."
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
