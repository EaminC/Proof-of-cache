#!/bin/bash
# Example 7: Search and Export
# Show how to search by different criteria and export results

echo "Example 7: Search and Export"
echo "============================"

# Use existing file or generate one
if [ ! -f "../data/search_example.json" ]; then
    echo "1. Generating sample file for search..."
    python3 ../src/gpt2_kvcache.py
    cp ../data/kv_cache_output.json ../data/search_example.json
    echo "   Generated: ../data/search_example.json"
fi

echo ""
echo "2. Search examples:"

echo ""
echo "   a) Search by token text 'brown':"
echo "   python3 ../tools/kvcache_query.py ../data/search_example.json --token-search brown"
python3 ../tools/kvcache_query.py ../data/search_example.json --token-search brown

echo ""
echo "   b) Search by norm range (0.5 to 1.0) and export to CSV:"
echo "   python3 ../tools/kvcache_query.py ../data/search_example.json --search-norm 0.5 1.0 key --export-csv ../data/norm_search.csv"
python3 ../tools/kvcache_query.py ../data/search_example.json --search-norm 0.5 1.0 key --export-csv ../data/norm_search.csv

echo ""
echo "   c) Get head summary [0,0]:"
echo "   python3 ../tools/kvcache_query.py ../data/search_example.json --head-summary 0 0"
python3 ../tools/kvcache_query.py ../data/search_example.json --head-summary 0 0

echo ""
echo "   d) Get token summary for token 0:"
echo "   python3 ../tools/kvcache_query.py ../data/search_example.json --token-summary 0"
python3 ../tools/kvcache_query.py ../data/search_example.json --token-summary 0

echo ""
echo "✅ Example 7 completed!"
echo "Check ../data/norm_search.csv for exported results."
