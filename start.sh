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

# Decompress models if compressed file exists and models are not already extracted
# Note: This runs BEFORE server starts, so keep it fast for Cloud Run cold starts
if [ -d "models" ] && [ -f "models/models_compressed.tar.gz" ]; then
    echo "📦 Checking for compressed models..."
    cd models
    
    # Check if any .pkl or .parquet files exist
    pkl_count=$(ls *.pkl 2>/dev/null | wc -l)
    parquet_count=$(ls *.parquet 2>/dev/null | wc -l)
    
    if [ "$pkl_count" -eq 0 ] && [ "$parquet_count" -eq 0 ]; then
        echo "📦 Decompressing model files (this may take 10-30 seconds)..."
        # Use timeout to prevent hanging, and continue even if it fails
        timeout 60 tar -xzf models_compressed.tar.gz || {
            echo "⚠️ Warning: Model decompression failed or timed out. Server will start anyway."
            echo "⚠️ Predictive analysis endpoints may not be available until models are loaded."
        }
        if [ $? -eq 0 ] || [ $? -eq 124 ]; then
            echo "✅ Models decompressed successfully (or timeout reached, continuing)"
        fi
    else
        echo "✅ Model files already extracted ($pkl_count .pkl files, $parquet_count .parquet files)"
    fi
    cd ..
fi

# Start uvicorn server
echo "Starting server on port $PORT..."
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Uvicorn version: $(uvicorn --version 2>&1 || echo 'not found')"

# Use exec to replace shell process with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info

