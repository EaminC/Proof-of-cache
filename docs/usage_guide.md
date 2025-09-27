# Usage Guide

This guide explains how to use the GPT-2 KV Cache extraction tools.

## Project Structure

```
poc/
├── src/                    # Source code
│   └── gpt2_kvcache.py    # Main KV cache extractor
├── tools/                  # Command-line tools
│   ├── kvcache_query.py   # Query tool for analyzing JSON data
│   └── dataset_generator.py # Dataset generation tool
├── examples/               # Usage examples
│   └── basic_usage.py     # Basic usage examples
├── data/                   # Data directory
│   ├── sample_prompts.txt  # Sample input prompts
│   └── *.json             # Generated KV cache files
├── docs/                   # Documentation
│   └── usage_guide.md     # This file
├── requirements.txt        # Python dependencies
└── README.md              # Main documentation
```

## Quick Start

### 1. Basic Usage

```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

# Create extractor
extractor = GPT2KVCacheExtractor(model_name="gpt2")

# Extract KV cache
result = extractor.extract_kv_cache("Hello world")

# Save to JSON
extractor.save_to_json(result, "output.json")
```

### 2. Command Line Usage

```bash
# Generate single dataset
python3 src/gpt2_kvcache.py

# Generate dataset from text file
python3 tools/dataset_generator.py --text-file data/sample_prompts.txt --output-dir data/datasets/

# Generate random dataset
python3 tools/dataset_generator.py --random 10 --output-dir data/random/

# Query KV cache data
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0
```

## Detailed Usage

### KV Cache Extractor

The main extractor class provides these methods:

- `extract_kv_cache(input_text, temperature=1.0, top_k=10)`: Extract KV cache and predict next token
- `get_coordinate_kv(data, layer, head, token)`: Get KV cache for specific coordinate
- `save_to_json(data, output_path)`: Save results to JSON file
- `analyze_kv_cache(data)`: Get statistics about KV cache

### Query Tool

The query tool (`tools/kvcache_query.py`) provides various ways to analyze generated JSON data:

#### Basic Information
```bash
# Show dataset info
python3 tools/kvcache_query.py data/kv_cache_output.json --info

# List all tokens
python3 tools/kvcache_query.py data/kv_cache_output.json --list-tokens
```

#### Coordinate Queries
```bash
# Get specific coordinate
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0

# Search by token text
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search "fox"
```

#### Summary Queries
```bash
# Layer summary
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 0

# Head summary
python3 tools/kvcache_query.py data/kv_cache_output.json --head-summary 0 0

# Token summary
python3 tools/kvcache_query.py data/kv_cache_output.json --token-summary 0
```

#### Search and Export
```bash
# Search by norm range
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key

# Export results to CSV
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key --export-csv results.csv
```

### Dataset Generator

The dataset generator (`tools/dataset_generator.py`) can create multiple datasets:

#### From Text File
```bash
python3 tools/dataset_generator.py \
    --text-file data/sample_prompts.txt \
    --output-dir data/datasets/ \
    --temperature 0.8 \
    --top-k 10
```

#### From Prompts
```bash
python3 tools/dataset_generator.py \
    --prompts "Hello world" "Machine learning" "Deep learning" \
    --output-dir data/datasets/
```

#### Random Dataset
```bash
python3 tools/dataset_generator.py \
    --random 100 \
    --output-dir data/random/ \
    --random-min 3 \
    --random-max 8
```

## Output Format

The generated JSON files contain:

```json
{
  "model": "gpt2",
  "input_text": "Input text",
  "input_tokens": ["token1", "token2", ...],
  "predicted_next_token": {
    "token": "predicted_token",
    "token_id": 123,
    "top_k_candidates": [...]
  },
  "kv_cache": {
    "layer_0": {
      "head_0": {
        "tokens": [
          {
            "token_idx": 0,
            "key": [64-dimensional vector],
            "value": [64-dimensional vector]
          }
        ]
      }
    }
  },
  "metadata": {
    "n_layers": 12,
    "n_heads": 12,
    "d_head": 64,
    "sequence_length": 9
  }
}
```

## Coordinate System

- **Layer**: 0-11 (12 Transformer layers)
- **Head**: 0-11 (12 attention heads per layer)
- **Token**: 0 to sequence_length-1 (input tokens)

Example: `[5, 3, 2]` means Layer 6, Head 4, Token 3.

## Examples

### Example 1: Extract and Query
```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

extractor = GPT2KVCacheExtractor()
result = extractor.extract_kv_cache("The quick brown fox")

# Get specific coordinate
kv_data = extractor.get_coordinate_kv(result, layer=0, head=0, token=0)
print(f"Key norm: {kv_data['key_norm']}")
```

### Example 2: Batch Processing
```python
inputs = ["Hello world", "Machine learning", "Deep learning"]
for i, text in enumerate(inputs):
    result = extractor.extract_kv_cache(text)
    extractor.save_to_json(result, f"output_{i}.json")
```

### Example 3: Command Line Analysis
```bash
# Generate dataset
python3 tools/dataset_generator.py --random 5 --output-dir data/test/

# Analyze results
python3 tools/kvcache_query.py data/test/dataset_index.json --info
```

## Tips

1. **Memory Usage**: Large sequences generate large JSON files. Monitor disk space.
2. **Performance**: Use GPU if available for faster processing.
3. **Batch Processing**: Use the dataset generator for multiple inputs.
4. **Analysis**: Use the query tool to explore generated data efficiently.
5. **Export**: Use CSV export for further analysis in other tools.
