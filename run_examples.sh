#!/bin/bash
# Run examples script

echo "GPT-2 KV Cache Extractor - Examples"
echo "===================================="

# Make tools executable
chmod +x tools/*.py
chmod +x examples/*.py

echo "1. Running basic usage examples..."
python3 examples/basic_usage.py

echo -e "\n2. Generating sample dataset..."
python3 tools/dataset_generator.py --text-file data/sample_prompts.txt --output-dir data/sample_datasets/

echo -e "\n3. Querying generated data..."
if [ -f "data/kv_cache_output.json" ]; then
    echo "Querying main output file:"
    python3 tools/kvcache_query.py data/kv_cache_output.json --info
    echo -e "\nCoordinate [0,0,0] query:"
    python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0
else
    echo "No main output file found, using sample dataset..."
    if [ -f "data/sample_datasets/dataset_index.json" ]; then
        echo "Sample dataset generated successfully!"
    else
        echo "No sample dataset found. Run the dataset generator first."
    fi
fi

echo -e "\nExamples completed!"
echo "Check the data/ directory for generated files."
