#!/bin/bash
# Script to decompress model files
# Usage: ./decompress_models.sh

cd "$(dirname "$0")"

if [ ! -f models_compressed.tar.gz ]; then
    echo "❌ models_compressed.tar.gz not found!"
    exit 1
fi

echo "Decompressing model files..."
tar -xzf models_compressed.tar.gz

if [ $? -eq 0 ]; then
    echo "✅ Decompression complete!"
    echo "   All model files (.pkl and .parquet) have been extracted."
    ls -lh *.pkl *.parquet 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
else
    echo "❌ Decompression failed!"
    exit 1
fi

