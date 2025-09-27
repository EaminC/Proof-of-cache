# GPT-2 KV Cache Extractor

A comprehensive tool for extracting and analyzing Key-Value cache from GPT-2 model for each layer, attention head, and token position.

## Features

- 🚀 Deploy GPT-2 model using Transformers library
- 📊 Extract complete KV cache from 12 layers × 12 attention heads
- 🎯 Predict next token with top-k candidates
- 💾 Save results in structured JSON format
- 🔍 Command-line query tool for analyzing generated data
- 📈 Dataset generation tools for batch processing
- 🛠️ Comprehensive visualization and analysis tools

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
│   └── usage_guide.md     # Detailed usage guide
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

# Create extractor
extractor = GPT2KVCacheExtractor(model_name="gpt2")

# Extract KV cache
result = extractor.extract_kv_cache("The quick brown fox jumps over the lazy dog")

# Save to JSON
extractor.save_to_json(result, "output.json")
```

### Command Line Usage

```bash
# Generate single dataset
python3 src/gpt2_kvcache.py

# Generate dataset from text file
python3 tools/dataset_generator.py --text-file data/sample_prompts.txt --output-dir data/datasets/

# Query KV cache data
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0
```

## Command Line Tools

### KV Cache Query Tool

Analyze and query generated JSON data:

```bash
# Show dataset information
python3 tools/kvcache_query.py data/kv_cache_output.json --info

# Get specific coordinate
python3 tools/kvcache_query.py data/kv_cache_output.json --coordinate 0 0 0

# Search by token text
python3 tools/kvcache_query.py data/kv_cache_output.json --token-search "fox"

# Layer summary
python3 tools/kvcache_query.py data/kv_cache_output.json --layer-summary 0

# Search by norm range and export to CSV
python3 tools/kvcache_query.py data/kv_cache_output.json --search-norm 0.5 1.0 key --export-csv results.csv
```

### Dataset Generator

Generate multiple datasets:

```bash
# From text file
python3 tools/dataset_generator.py --text-file data/sample_prompts.txt --output-dir data/datasets/

# From prompts
python3 tools/dataset_generator.py --prompts "Hello world" "Machine learning" --output-dir data/datasets/

# Random dataset
python3 tools/dataset_generator.py --random 100 --output-dir data/random/
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
    "top_k_candidates": [
      {"token": "candidate1", "probability": 0.25},
      {"token": "candidate2", "probability": 0.15}
    ]
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

KV Cache uses a 3D coordinate system:
- **Layer**: 0-11, 12 Transformer blocks
- **Head**: 0-11, 12 attention heads per layer  
- **Token**: 0 to sequence_length-1, corresponding to each input token

For example, coordinate `[5, 3, 2]` means:
- Layer 6 (0-indexed)
- Attention head 4
- Token 3

## Examples

### Example 1: Basic Extraction
```python
from src.gpt2_kvcache import GPT2KVCacheExtractor

extractor = GPT2KVCacheExtractor()
result = extractor.extract_kv_cache("Hello world")
print(f"Predicted next token: {result['predicted_next_token']['token']}")
```

### Example 2: Coordinate Query
```python
# Get KV cache for specific coordinate
kv_data = extractor.get_coordinate_kv(result, layer=0, head=0, token=0)
print(f"Key vector: {kv_data['key'][:5]}")
print(f"Value vector: {kv_data['value'][:5]}")
```

### Example 3: Batch Processing
```bash
# Generate multiple datasets
python3 tools/dataset_generator.py --random 10 --output-dir data/batch/

# Analyze all generated files
for file in data/batch/*.json; do
    python3 tools/kvcache_query.py "$file" --info
done
```

## Model Variants

You can use different GPT-2 model sizes:
- `gpt2`: 124M parameters (default)
- `gpt2-medium`: 355M parameters
- `gpt2-large`: 774M parameters
- `gpt2-xl`: 1.5B parameters

```python
# Use larger model
extractor = GPT2KVCacheExtractor(model_name="gpt2-medium")
```

## Performance Notes

- GPT-2 model parameters: ~124M
- KV cache size per token: ~0.6MB (float32)
- Supports both CPU and GPU (auto-detected)
- First run downloads model (~500MB)

## Documentation

- [Usage Guide](docs/usage_guide.md) - Detailed usage instructions
- [Examples](examples/) - Code examples
- [Tools](tools/) - Command-line tools

## License

MIT