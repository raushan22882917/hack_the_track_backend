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
if [ -d "models" ] && [ -f "models/models_compressed.tar.gz" ]; then
    echo "📦 Checking for compressed models..."
    cd models
    
    # Check if any .pkl or .parquet files exist
    pkl_count=$(ls *.pkl 2>/dev/null | wc -l)
    parquet_count=$(ls *.parquet 2>/dev/null | wc -l)
    
    if [ "$pkl_count" -eq 0 ] && [ "$parquet_count" -eq 0 ]; then
        echo "📦 Decompressing model files..."
        tar -xzf models_compressed.tar.gz
        if [ $? -eq 0 ]; then
            echo "✅ Models decompressed successfully"
        else
            echo "⚠️ Warning: Failed to decompress models. Continuing anyway..."
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
echo "PORT environment variable: $PORT"

# Verify Python can import the app
echo "Testing app import..."
python -c "import main; print('✅ App imported successfully')" || {
    echo "❌ Failed to import app module"
    exit 1
}

# Use exec to replace shell process with uvicorn
# Use --timeout-keep-alive to prevent connection timeouts
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info --timeout-keep-alive 30

