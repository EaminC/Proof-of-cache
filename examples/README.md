# Examples

Simple, one-file examples for the GPT-2 KV Cache Extractor.

## 📋 Available Examples

| File | Description | Type |
|------|-------------|------|
| `01_basic_extraction.py` | Basic KV cache extraction | Python |
| `02_coordinate_query.py` | Query specific coordinates | Python |
| `03_batch_processing.py` | Process multiple inputs | Python |
| `04_command_line_query.sh` | Command line query examples | Shell |
| `05_dataset_generation.sh` | Generate multiple datasets | Shell |
| `06_analysis_workflow.py` | Complete analysis workflow | Python |
| `07_search_and_export.sh` | Search and export results | Shell |
| `run_all_examples.sh` | Run all examples | Shell |

## 🚀 Quick Start

### Run Individual Examples
```bash
# Python examples
python3 01_basic_extraction.py
python3 02_coordinate_query.py
python3 03_batch_processing.py
python3 06_analysis_workflow.py

# Shell examples
./04_command_line_query.sh
./05_dataset_generation.sh
./07_search_and_export.sh
```

### Run All Examples
```bash
./run_all_examples.sh
```

## 📖 Example Descriptions

### 01_basic_extraction.py
- Shows basic KV cache extraction
- Input: "The quick brown fox jumps over the lazy dog"
- Output: JSON file with complete KV cache data

### 02_coordinate_query.py
- Demonstrates coordinate-based queries
- Queries specific [layer, head, token] coordinates
- Shows how to access KV data for specific positions

### 03_batch_processing.py
- Processes multiple inputs in batch
- Shows how to handle multiple texts
- Saves each result to separate files

### 04_command_line_query.sh
- Command line query examples
- Shows `--info`, `--coordinate`, `--token-search` options
- Demonstrates basic query tool usage

### 05_dataset_generation.sh
- Dataset generation examples
- From text file, prompts, and random generation
- Shows different ways to create datasets

### 06_analysis_workflow.py
- Complete analysis workflow
- From generation to analysis
- Shows how to analyze multiple files

### 07_search_and_export.sh
- Search and export examples
- Token search, norm range search
- CSV export functionality

## 🎯 Key Features Demonstrated

- ✅ Basic KV cache extraction
- ✅ Coordinate-based queries
- ✅ Batch processing
- ✅ Command line tools
- ✅ Dataset generation
- ✅ Search and analysis
- ✅ Export functionality

## 📁 Output Files

Examples generate files in `../data/`:
- `example_1_output.json` - Basic extraction result
- `batch_example_*.json` - Batch processing results
- `analysis_*.json` - Analysis workflow results
- `*.csv` - Exported search results
- Various generated datasets

## 🔧 Requirements

- Python 3.7+
- Dependencies from `../requirements.txt`
- GPT-2 model (downloaded automatically)

## 💡 Tips

1. **Start with Example 1** - Basic extraction
2. **Try Example 2** - Coordinate queries
3. **Use Example 4** - Command line tools
4. **Run Example 6** - Complete workflow
5. **Check Example 7** - Search and export

Each example is self-contained and can be run independently!
