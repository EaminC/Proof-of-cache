#!/bin/bash
# Quick Demo Script for GPT-2 KV Cache Extractor

echo "🚀 GPT-2 KV Cache Extractor - Quick Demo"
echo "========================================="

# Create data directory
mkdir -p data

echo ""
echo "1️⃣  Basic KV Cache Extraction"
echo "-------------------------------"
echo "Input: 'The quick brown fox jumps over the lazy dog'"
python3 src/gpt2_kvcache.py

echo ""
echo "2️⃣  Query Dataset Information"
echo "------------------------------"
python3 tools/kvcache_query.py data/kv_cache_output.json --info

echo ""
echo "3️⃣  Query Specific Coordinate [0,0,0]"
echo "-------------------------------------"
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0

echo ""
echo "4️⃣  Search for Token 'fox'"
echo "---------------------------"
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search 'fox'

echo ""
echo "5️⃣  Generate Random Dataset"
echo "----------------------------"
python3 tools/dataset_generator.py --random 3 --output-dir data/demo_random/

echo ""
echo "6️⃣  Layer Summary"
echo "-----------------"
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 0

echo ""
echo "✅ Demo completed! Check the data/ directory for generated files."
echo "========================================="
