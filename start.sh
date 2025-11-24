#!/bin/bash
set -e

# Get port from environment variable, default to 8080
# Cloud Run automatically sets PORT environment variable
PORT=${PORT:-8080}

# Ensure PORT is a valid integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: PORT must be a number, got: $PORT"
    exit 1
fi

# Verify main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in current directory"
    exit 1
fi

# Start uvicorn server
echo "Starting server on port $PORT..."
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version 2>&1 || echo 'not found')"

# Use exec to replace shell process with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info

