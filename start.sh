#!/bin/bash

# VLLM Intermediate State Inspector startup script

echo "VLLM Intermediate State Inspector"
echo "================================="

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
if ! python3 -c "import torch, transformers" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Create output directory
mkdir -p outputs

echo "Dependency check completed!"
echo ""

# Show usage options
echo "Usage options:"
echo "1. Run examples"
echo "2. Basic analysis"
echo "3. Custom analysis"
echo "4. Exit"
echo ""

read -p "Please select (1-4): " choice

case $choice in
    1)
        echo "Running all examples..."
        python3 examples.py
        ;;
    2)
        echo "Performing basic analysis..."
        read -p "Enter text to analyze: " text
        python3 main.py --text "$text" --export-formats json csv
        ;;
    3)
        echo "Custom analysis..."
        read -p "Model name (default: gpt2): " model
        model=${model:-gpt2}
        
        read -p "Enter text: " text
        
        read -p "Target tokens (space separated, leave empty to analyze all): " tokens
        
        read -p "Output directory (default: ./outputs): " output_dir
        output_dir=${output_dir:-./outputs}
        
        if [ -z "$tokens" ]; then
            python3 main.py --model "$model" --text "$text" --output-dir "$output_dir"
        else
            python3 main.py --model "$model" --text "$text" --target-tokens $tokens --output-dir "$output_dir"
        fi
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "Analysis completed! Check output directory for results."
echo "For more options, run: python3 main.py --help"