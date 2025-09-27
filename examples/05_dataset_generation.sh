#!/bin/bash
# Example 5: Dataset Generation
# Show how to generate multiple datasets

echo "Example 5: Dataset Generation"
echo "============================="

echo "1. Generate dataset from text file:"
echo "   python3 ../tools/dataset_generator.py --text-file ../data/sample_prompts.txt --output-dir ../data/generated_from_file/"
python3 ../tools/dataset_generator.py --text-file ../data/sample_prompts.txt --output-dir ../data/generated_from_file/

echo ""
echo "2. Generate dataset from prompts:"
echo "   python3 ../tools/dataset_generator.py --prompts 'Hello world' 'Machine learning' --output-dir ../data/generated_from_prompts/"
python3 ../tools/dataset_generator.py --prompts 'Hello world' 'Machine learning' --output-dir ../data/generated_from_prompts/

echo ""
echo "3. Generate random dataset:"
echo "   python3 ../tools/dataset_generator.py --random 3 --output-dir ../data/random_dataset/"
python3 ../tools/dataset_generator.py --random 3 --output-dir ../data/random_dataset/

echo ""
echo "✅ Example 5 completed!"
echo "Check the ../data/ directory for generated files."
