# Complete Examples Guide

This guide provides comprehensive examples of how to use the GPT-2 KV Cache Extractor.

## 🚀 Quick Start Examples

### 1. Basic Usage
```bash
# Generate single KV cache dataset
python3 src/gpt2_kvcache.py
```

### 2. Query Generated Data
```bash
# Show dataset information
python3 tools/kvcache_query.py data/kv_cache_output.json --info

# Query specific coordinate [layer, head, token]
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0

# Search for token containing "fox"
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search "fox"
```

### 3. Generate Multiple Datasets
```bash
# From text file
python3 tools/dataset_generator.py --text-file data/sample_prompts.txt --output-dir data/datasets/

# Random dataset
python3 tools/dataset_generator.py --random 10 --output-dir data/random/
```

## 📋 Complete Example Workflows

### Example 1: Single Dataset Analysis

```bash
# Step 1: Generate dataset
python3 src/gpt2_kvcache.py

# Step 2: View basic information
python3 tools/kvcache_query.py data/kv_cache_output.json --info

# Step 3: Query specific coordinates
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 5 5 2

# Step 4: Search by token
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search "brown"

# Step 5: Get layer summary
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 0

# Step 6: Search by norm range and export
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key --export-csv results.csv
```

### Example 2: Batch Dataset Generation

```bash
# Step 1: Create prompts file
echo -e "Machine learning is\nDeep learning models\nNatural language processing" > data/my_prompts.txt

# Step 2: Generate datasets
python3 tools/dataset_generator.py --text-file data/my_prompts.txt --output-dir data/batch_results/

# Step 3: Analyze each generated file
for file in data/batch_results/*.json; do
    echo "Analyzing: $file"
    python3 tools/kvcache_query.py "$file" --info
done
```

### Example 3: Random Dataset Analysis

```bash
# Step 1: Generate random dataset
python3 tools/dataset_generator.py --random 5 --output-dir data/random_analysis/

# Step 2: Analyze the dataset index
python3 tools/kvcache_query.py data/random_analysis/dataset_index.json --info

# Step 3: Query specific files
python3 tools/kvcache_query.py data/random_analysis/*.json --coordinate 0 0 0
```

## 🔍 Advanced Query Examples

### Coordinate Queries
```bash
# Get specific coordinate
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0

# Get head summary
python3 tools/kvcache_query.py data/kv_cache_output.json --head-summary 0 0

# Get token summary
python3 tools/kvcache_query.py data/kv_cache_output.json --token-summary 0
```

### Search Queries
```bash
# Search by token text
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search "quick"

# Search by norm range
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key

# Search and export to CSV
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key --export-csv search_results.csv
```

### Summary Queries
```bash
# Layer summaries
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 0
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 11

# List all tokens
python3 tools/kvcache_query.py data/kv_cache_output.json --list-tokens
```

## 🛠️ Dataset Generation Examples

### From Text File
```bash
# Create input file
cat > data/tech_prompts.txt << EOF
Artificial intelligence will
Machine learning algorithms
Deep neural networks
Computer vision systems
Natural language processing
EOF

# Generate datasets
python3 tools/dataset_generator.py --text-file data/tech_prompts.txt --output-dir data/tech_datasets/
```

### From Command Line Prompts
```bash
python3 tools/dataset_generator.py \
    --prompts "Hello world" "Machine learning" "Deep learning" \
    --output-dir data/prompt_datasets/
```

### Random Dataset
```bash
python3 tools/dataset_generator.py \
    --random 20 \
    --output-dir data/random_20/ \
    --random-min 3 \
    --random-max 8
```

## 📊 Analysis Examples

### Complete Analysis Workflow
```bash
# 1. Generate dataset
python3 src/gpt2_kvcache.py

# 2. Basic analysis
python3 tools/kvcache_query.py data/kv_cache_output.json --info

# 3. Coordinate analysis
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 11 11 8

# 4. Layer analysis
for layer in {0..11}; do
    echo "Layer $layer:"
    python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary $layer
done

# 5. Token analysis
for token in {0..8}; do
    echo "Token $token:"
    python3 tools/kvcache_query.py data/kv_cache_output.json --token-summary $token
done

# 6. Search and export
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key --export-csv analysis_results.csv
```

### Batch Analysis
```bash
# Generate multiple datasets
python3 tools/dataset_generator.py --random 5 --output-dir data/batch_analysis/

# Analyze all files
for file in data/batch_analysis/*.json; do
    echo "=== Analyzing $file ==="
    python3 tools/kvcache_query.py "$file" --info
    python3 tools/kvcache_query.py "$file" --coordinate 0 0 0
done
```

## 🎯 Python API Examples

### Basic Python Usage
```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

# Create extractor
extractor = GPT2KVCacheExtractor(model_name="gpt2")

# Extract KV cache
result = extractor.extract_kv_cache("The quick brown fox")

# Save result
extractor.save_to_json(result, "my_output.json")

# Query specific coordinate
kv_data = extractor.get_coordinate_kv(result, layer=0, head=0, token=0)
print(f"Key norm: {kv_data['key_norm']}")
print(f"Value norm: {kv_data['value_norm']}")
```

### Batch Processing in Python
```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

extractor = GPT2KVCacheExtractor()

inputs = [
    "Machine learning is",
    "Deep learning models",
    "Natural language processing"
]

for i, input_text in enumerate(inputs):
    result = extractor.extract_kv_cache(input_text)
    extractor.save_to_json(result, f"batch_{i}.json")
    print(f"Input: {input_text}")
    print(f"Predicted: {result['predicted_next_token']['token']}")
```

## 🔧 Command Line Tool Reference

### KV Cache Query Tool
```bash
python3 tools/kvcache_query.py <json_file> [options]

Options:
  --info                    Show dataset information
  --list-tokens            List all tokens
  --coordinate L H T       Get specific coordinate [Layer, Head, Token]
  --token-search TEXT      Search for token containing TEXT
  --layer-summary L        Get layer L summary
  --head-summary L H       Get head [L,H] summary
  --token-summary T        Get token T summary
  --search-norm MIN MAX TYPE Search by norm range (TYPE: key or value)
  --export-csv FILE        Export results to CSV file
```

### Dataset Generator Tool
```bash
python3 tools/dataset_generator.py [options]

Options:
  --model MODEL            GPT-2 model name (default: gpt2)
  --output-dir DIR         Output directory
  --text-file FILE         Input text file
  --prompts TEXT [TEXT...] Input prompts
  --random NUM             Generate random dataset
  --random-min MIN         Minimum prompt length
  --random-max MAX         Maximum prompt length
  --temperature TEMP       Generation temperature
  --top-k K               Top-k candidates
```

## 📁 Output File Structure

Generated files include:
- `kv_cache_output.json` - Main output file
- `dataset_index.json` - Dataset index for batch generation
- `*.csv` - Exported analysis results
- `batch_*.json` - Batch processing results

## 🎉 Quick Demo

Run the complete demo:
```bash
./quick_demo.sh
```

This will demonstrate all major features with real examples.
