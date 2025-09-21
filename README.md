# Transformer Intermediate State Inspector

A powerful tool for analyzing intermediate states and token trajectories in Transformer models. Enables deep exploration of the internal workings of models and tracking the evolution of specific tokens across layers.

## 🌟 Main Features

### 🎯 Core Features
- **Token Tracking**: Track specific input/output tokens' intermediate states across model layers
- **Coordinate Positioning**: Precisely locate intermediate variables by position (layer, position, head, feature dimension, etc.)
- **Multi-format Export**: Support for JSON, NPZ, CSV, HDF5, and other data export formats
- **Visualization Analysis**: Automatically generate trajectory plots, attention heatmaps, and other visualizations
- **Model Compatibility**: Support for GPT-2, BERT, RoBERTa, and other Transformer models

### 📊 Analysis Capabilities
- **Hidden State Trajectories**: Analyze token hidden state changes across layers
- **Attention Patterns**: Capture and analyze the evolution of self-attention weights
- **Token Comparison**: Compare internal representations and similarities between different tokens
- **Layer Analysis**: Statistical analysis of activation patterns across layers
- **Attention Flow**: Track attention flow direction and intensity changes

## 🚀 Quick Start

### Install Dependencies

```bash
cd poc
pip install -r requirements.txt
```

### Basic Usage

```python
from main import TransformerInspector

# Create inspector
inspector = TransformerInspector()

# Load model
inspector.load_model("gpt2")

# Analyze text
result = inspector.analyze_text(
    text="Hello world! How are you today?",
    target_tokens=["Hello", "world"],
    export_formats=["json", "csv"]
)

# Generate visualizations
inspector.visualize_results(result)
```

### Command Line Usage

```bash
# Basic analysis
python main.py --model gpt2 --text "Hello world!"

# Specify target tokens
python main.py --model gpt2 --text "The cat sat on the mat" --target-tokens cat sat

# Custom output
python main.py --model gpt2 --text "AI is amazing" --output-dir ./my_analysis --export-formats json npz csv
```

## 📁 Project Structure

```
poc/
├── main.py                    # Main program and integration interface
├── model_loader.py           # Model loading and initialization
├── hook_system.py            # Hook system (capture intermediate activations)
├── token_tracker.py          # Token tracker
├── coordinate_system.py      # Coordinate system and data export
├── examples.py               # Usage examples
├── config.json               # Configuration file
├── requirements.txt          # Dependencies list
└── README.md                 # Documentation
```

## 💡 Usage Examples

### Example 1: Basic Token Analysis

```python
from main import TransformerInspector

inspector = TransformerInspector()
inspector.load_model("gpt2")

# Analyze keywords in sentence
result = inspector.analyze_text(
    text="The quick brown fox jumps over the lazy dog.",
    target_tokens=["quick", "fox", "jumps"]
)

# View analysis report
report = result['analysis_report']
print(f"Tracked {len(report['summary']['tracked_positions'])} tokens")
```

### Example 2: Attention Analysis

```python
from token_tracker import TokenTracker
from model_loader import ModelLoader

loader = ModelLoader()
loader.load_model("gpt2")

tracker = TokenTracker(loader.model, loader.tokenizer)

# Analyze attention patterns
text = "I love natural language processing."
inputs = loader.tokenize_text(text)
result = tracker.trace_tokens(inputs, trace_all=True)

# Get attention flow for specific position
attention_flow = tracker.get_attention_flow(2)  # position of "natural"
```

### Example 3: Token Comparison

```python
# Compare internal representations of synonyms
text = "The cat and the kitten are playing together."
inputs = loader.tokenize_text(text)
result = tracker.trace_tokens(inputs, trace_all=True)

# Compare similarity between "cat" and "kitten"
comparison = tracker.compare_tokens(1, 4, metric='cosine')
print("Cosine similarity across layers:", comparison['similarities'])
```

### Example 4: Custom Coordinate System

```python
from coordinate_system import CoordinateManager

manager = CoordinateManager("gpt2")

# Create precise coordinates
coord = manager.create_coordinate(
    batch_idx=0,
    sequence_idx=5,      # 5th token
    layer_idx=3,         # 3rd layer
    head_idx=2,          # 2nd attention head
    module_type="attention"
)

# Add intermediate variable
value = torch.randn(768)  # hidden state vector
manager.add_variable(coord, value, "Layer 3 attention output")

# Export data
exported = manager.export_all("my_analysis", formats=["json", "hdf5"])
```

## 🔧 Configuration

Edit `config.json` to customize configuration:

```json
{
    "model_name": "gpt2",
    "device": "auto",
    "max_length": 128,
    "output_dir": "./outputs",
    "export_formats": ["json", "npz", "csv"],
    "capture_layers": {
        "attention": true,
        "mlp": true,
        "embeddings": true,
        "layer_norm": true,
        "final_logits": true
    },
    "layer_indices": "all",
    "attention_heads": "all",
    "position_range": "all"
}
```

## 📊 Coordinate System

Our coordinate system provides precise intermediate variable positioning:

```python
# Coordinate format
CoordinateSystem(
    batch_idx=0,          # Batch dimension
    sequence_idx=5,       # Sequence position
    layer_idx=3,          # Layer index
    head_idx=2,           # Attention head index
    feature_idx=100,      # Feature dimension index
    module_type="attention",  # Module type
    module_name="transformer.h.3.attn"  # Specific module name
)

# Coordinate string representation
"B0_S5_L3_H2_F100_(attention)"
```

## 🎨 Visualization

Automatically generate various visualization charts:

1. **Token Trajectory Plots**: Show token hidden state changes across layers
2. **Attention Heatmaps**: Display attention weight distributions
3. **Layer Analysis Charts**: Statistics of variable distribution across layers
4. **Similarity Plots**: Compare similarity evolution between different tokens

## 📤 Data Export

Support for multiple data export formats:

- **JSON**: Human-readable, includes complete metadata
- **NPZ**: NumPy native format, efficient storage
- **CSV**: Tabular format, suitable for Excel/pandas analysis
- **HDF5**: Scientific computing standard, supports big data
- **Pickle**: Python native serialization

## 🔬 Advanced Features

### Hook System

Direct access to model internals:

```python
from hook_system import IntermediateInspector

inspector = IntermediateInspector(model)
inspector.register_hooks(layer_patterns=["transformer.h.0", "transformer.h.1"])

# Capture forward pass
result = inspector.capture_forward_pass(inputs)
```

### Batch Analysis

```python
texts = [
    "The cat sat on the mat.",
    "A dog ran in the park.",
    "Birds fly in the sky."
]

for i, text in enumerate(texts):
    result = inspector.analyze_text(text, export_formats=["json"])
    print(f"Analyzed text {i+1}/3")
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce sequence length
   inspector.loader.config["max_length"] = 64
   
   # Only analyze specific layers
   inspector.register_hooks(layer_patterns=["transformer.h.0", "transformer.h.1"])
   ```

2. **Model Loading Failed**
   ```python
   # Specify device
   inspector.loader.config["device"] = "cpu"  # or "cuda"
   ```

3. **Export Files Too Large**
   ```python
   # Use compressed formats
   exported = manager.export_all("analysis", formats=["npz"], compress=True)
   ```

## 📚 API Documentation

### Main Classes

#### `TransformerInspector`
Main integration interface providing end-to-end analysis workflow.

#### `TokenTracker`
Token tracker specifically for tracking state changes of specific tokens.

#### `CoordinateManager`
Coordinate manager providing precise variable positioning and management.

#### `IntermediateInspector`
Hook system for capturing internal model activations.

### Core Methods

```python
# Load model
inspector.load_model(model_name, model_type="causal_lm")

# Analyze text
result = inspector.analyze_text(text, target_tokens=None, export_formats=["json"])

# Track tokens
tracking_result = tracker.trace_tokens(inputs, trace_all=False)

# Get trajectory
trajectory = tracker.get_token_trajectory(position, layer_range=None)

# Compare tokens
comparison = tracker.compare_tokens(pos1, pos2, metric='cosine')
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

Built on Transformers library and PyTorch.

---

**Start exploring the internal world of Transformer models!** 🚀