#!/bin/bash

# Start Backend Server for Predictive Analysis
echo "🚀 Starting Backend Server..."
echo "================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import fastapi, joblib, pyarrow" 2>/dev/null || {
    echo "⚠️  Missing required packages. Installing dependencies..."
    pip3 install -r requirements.txt
}

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Models directory not found!"
    echo "   Make sure model files are in the /models directory"
    exit 1
fi

# Start the server
echo ""
echo "✅ Starting server on http://127.0.0.1:8000"
echo "   Press Ctrl+C to stop"
echo ""
echo "================================"
echo ""

python3 main.py

