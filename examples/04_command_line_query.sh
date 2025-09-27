#!/bin/bash
# Example 4: Command Line Query
# Show how to use the command line query tool

echo "Example 4: Command Line Query"
echo "============================"

# First generate a sample file if it doesn't exist
if [ ! -f "../data/sample_for_query.json" ]; then
    echo "1. Generating sample file..."
    python3 ../src/gpt2_kvcache.py
    cp ../data/kv_cache_output.json ../data/sample_for_query.json
    echo "   Generated: ../data/sample_for_query.json"
fi

echo ""
echo "2. Query examples:"

echo ""
echo "   a) Show dataset information:"
echo "   python3 ../tools/kvcache_query.py ../data/sample_for_query.json --info"
python3 ../tools/kvcache_query.py ../data/sample_for_query.json --info

echo ""
echo "   b) Query specific coordinate [0,0,0]:"
echo "   python3 ../tools/kvcache_query.py ../data/sample_for_query.json --coordinate 0 0 0"
python3 ../tools/kvcache_query.py ../data/sample_for_query.json --coordinate 0 0 0

echo ""
echo "   c) Search for token 'fox':"
echo "   python3 ../tools/kvcache_query.py ../data/sample_for_query.json --token-search fox"
python3 ../tools/kvcache_query.py ../data/sample_for_query.json --token-search fox

echo ""
echo "   d) Get layer 0 summary:"
echo "   python3 ../tools/kvcache_query.py ../data/sample_for_query.json --layer-summary 0"
python3 ../tools/kvcache_query.py ../data/sample_for_query.json --layer-summary 0

echo ""
echo "✅ Example 4 completed!"
