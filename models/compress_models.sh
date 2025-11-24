#!/bin/bash
# Script to compress model files for GitHub storage
# Usage: ./compress_models.sh

cd "$(dirname "$0")"

echo "Compressing model files..."
tar -czf models_compressed.tar.gz *.pkl *.parquet 2>/dev/null

if [ -f models_compressed.tar.gz ]; then
    original_size=$(du -sh *.pkl *.parquet 2>/dev/null | awk '{sum+=$1} END {print sum}')
    compressed_size=$(du -sh models_compressed.tar.gz | awk '{print $1}')
    echo "✅ Compression complete!"
    echo "   Compressed file: models_compressed.tar.gz ($compressed_size)"
    echo "   Original size: ~51MB"
    echo "   Compressed size: $compressed_size"
else
    echo "❌ Compression failed!"
    exit 1
fi



