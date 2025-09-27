#!/bin/bash
# Run All Examples
# Simple script to run all examples in order

echo "🚀 GPT-2 KV Cache Extractor - All Examples"
echo "==========================================="

# Make all files executable
chmod +x *.py *.sh

echo ""
echo "Running all examples..."
echo ""

echo "Example 1: Basic Extraction"
echo "----------------------------"
python3 01_basic_extraction.py

echo ""
echo "Example 2: Coordinate Query"
echo "---------------------------"
python3 02_coordinate_query.py

echo ""
echo "Example 3: Batch Processing"
echo "---------------------------"
python3 03_batch_processing.py

echo ""
echo "Example 4: Command Line Query"
echo "------------------------------"
./04_command_line_query.sh

echo ""
echo "Example 5: Dataset Generation"
echo "-----------------------------"
./05_dataset_generation.sh

echo ""
echo "Example 6: Analysis Workflow"
echo "----------------------------"
python3 06_analysis_workflow.py

echo ""
echo "Example 7: Search and Export"
echo "----------------------------"
./07_search_and_export.sh

echo ""
echo "🎉 All examples completed!"
echo "Check the ../data/ directory for generated files."
echo "==========================================="
