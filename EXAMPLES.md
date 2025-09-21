# Transformer 中间状态分析器 - 使用示例

这个文档展示了如何使用 Transformer 中间状态分析器的各种功能。从基础分析到高级自定义，提供了完整的使用案例。

## 快速开始

### 示例 1: 基础文本分析

```python
from main import TransformerInspector

# 创建分析器
inspector = TransformerInspector()

# 加载模型
inspector.load_model("gpt2")

# 分析文本
text = "The cat sits on the mat"
result = inspector.analyze_text(
    text=text,
    target_tokens=["cat", "sits", "mat"],
    export_formats=["json", "csv"]
)

# 查看结果
print(f"分析了 {result['analysis_report']['summary']['total_tokens']} 个 tokens")
print(f"追踪了 {len(result['analysis_report']['token_analysis'])} 个目标 tokens")
```

### 示例 2: 指定位置分析

```python
# 按位置索引分析 tokens
result = inspector.analyze_text(
    text="Hello world from transformer analysis",
    target_positions=[0, 1, 4],  # 分析第 0, 1, 4 个位置的 tokens
    export_formats=["npz"]
)
```

## 进阶功能

### 示例 3: 注意力流分析

```python
from token_tracker import TokenTracker
from model_loader import ModelLoader

# 设置模型和追踪器
loader = ModelLoader()
loader.load_model("gpt2")
tracker = TokenTracker(loader.model, loader.tokenizer)

# 分析句子
text = "The programmer writes clean and efficient code"
inputs = loader.tokenize_text(text)

# 追踪 "programmer" 这个词的注意力流
result = tracker.trace_tokens(inputs, trace_all=True)
token_info = loader.get_token_info(inputs['input_ids'])

# 找到 "programmer" 的位置
programmer_pos = None
for i, token in enumerate(token_info['tokens']):
    if 'programmer' in token.lower():
        programmer_pos = i
        break

if programmer_pos is not None:
    # 获取注意力流
    attention_flow = tracker.get_attention_flow(programmer_pos)
    
    print(f"Token 'programmer' (位置 {programmer_pos}) 的注意力分析:")
    
    # 分析出向注意力
    for layer, attn_info in list(attention_flow['outgoing_attention'].items())[:5]:
        attended_tokens = [token_info['tokens'][pos] for pos in attn_info['top_attended']]
        print(f"  层 {layer}: 最关注 → {attended_tokens}")
    
    # 分析入向注意力  
    for layer, attn_info in list(attention_flow['incoming_attention'].items())[:5]:
        attending_tokens = [token_info['tokens'][pos] for pos in attn_info['top_attending']]
        print(f"  层 {layer}: 被关注 ← {attending_tokens}")
```

### 示例 4: Token 语义比较

```python
# 比较语义相似的词汇
text = "The dog and the puppy are both canines"
inputs = loader.tokenize_text(text)
result = tracker.trace_tokens(inputs, trace_all=True)
token_info = loader.get_token_info(inputs['input_ids'])

# 找到 "dog" 和 "puppy" 的位置
dog_pos = next((i for i, t in enumerate(token_info['tokens']) if 'dog' in t.lower()), None)
puppy_pos = next((i for i, t in enumerate(token_info['tokens']) if 'puppy' in t.lower()), None)

if dog_pos is not None and puppy_pos is not None:
    # 计算各层的相似度
    comparison = tracker.compare_tokens(dog_pos, puppy_pos, metric='cosine')
    
    print(f"'dog' vs 'puppy' 各层语义相似度:")
    for layer in sorted(comparison['similarities'].keys())[:8]:
        sim = comparison['similarities'][layer]
        print(f"  层 {layer}: {sim:.4f}")
    
    # 找到最相似的层
    max_sim_layer = max(comparison['similarities'].items(), key=lambda x: x[1])
    print(f"最高相似度: 层 {max_sim_layer[0]} = {max_sim_layer[1]:.4f}")
```

## 坐标系统与数据管理

### 示例 5: 精确坐标定位

```python
from coordinate_system import CoordinateManager, DataExporter

# 创建坐标管理器
manager = CoordinateManager("gpt2")

# 模拟捕获特定位置的中间变量
text = "Natural language processing"
inputs = loader.tokenize_text(text)

# 为每个 token 在每层创建坐标
for layer_idx in range(6):  # 前6层
    for seq_pos in range(len(inputs['input_ids'][0])):
        # 隐藏状态坐标
        coord = manager.create_coordinate(
            batch_idx=0,
            sequence_idx=seq_pos,
            layer_idx=layer_idx,
            module_type="hidden_state",
            module_name=f"transformer.h.{layer_idx}"
        )
        
        # 存储模拟的隐藏状态
        hidden_state = torch.randn(768)  # GPT-2 hidden dimension
        manager.add_variable(
            coord, 
            hidden_state, 
            f"Hidden state for token {seq_pos} at layer {layer_idx}"
        )

# 数据检索示例
print("坐标系统数据管理:")
summary = manager.summary()
print(f"总变量数: {summary['total_variables']}")
print(f"层分布: {summary['layer_distribution']}")

# 按条件过滤
layer_2_vars = manager.get_layer_variables(2)
print(f"第2层变量数: {len(layer_2_vars)}")

position_1_vars = manager.get_position_variables(1)
print(f"位置1的变量数: {len(position_1_vars)}")

# 多格式导出
exported = manager.export_all(
    "nlp_analysis", 
    formats=["json", "csv", "npz", "hdf5"]
)
print("导出文件:", list(exported.keys()))
```

## 自定义分析流程

### 示例 6: Hook 系统直接使用

```python
from hook_system import IntermediateInspector

# 创建 Hook 检查器
hook_inspector = IntermediateInspector(loader.model, capture_attention=True)

# 只监控特定层模式
target_patterns = [
    "transformer.h.0",    # 第一层
    "transformer.h.5",    # 第六层  
    "transformer.h.11"    # 最后一层
]

hook_inspector.register_hooks(layer_patterns=target_patterns)

# 执行分析
text = "Artificial intelligence transforms our world"
inputs = loader.tokenize_text(text)
result = hook_inspector.capture_forward_pass(inputs)

print(f"捕获的层数: {len(result['activations'])}")
print(f"注意力权重: {len(result['attention_weights'])}")

# 分析激活统计
print("\n各层激活统计:")
for layer_name, activations in result['activations'].items():
    for key, tensor in activations.items():
        if tensor is not None and hasattr(tensor, 'mean'):
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            stats = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'shape': list(tensor.shape)
            }
            print(f"  {layer_name}.{key}: shape={stats['shape']}, mean={stats['mean']:.4f}")

# 清理 hooks
hook_inspector.remove_hooks()
```

## 批量处理与比较

### 示例 7: 多文本批量分析

```python
# 批量分析多个句子
texts = [
    "The scientist discovered a new species",
    "The researcher found a novel organism", 
    "The explorer identified an unknown creature"
]

results = []
for i, text in enumerate(texts):
    print(f"\n分析句子 {i+1}: {text}")
    
    result = inspector.analyze_text(
        text=text,
        target_tokens=["scientist", "researcher", "explorer", "discovered", "found", "identified"],
        export_formats=["json"],
        output_prefix=f"batch_analysis_{i}"
    )
    
    results.append(result)
    
    # 简要统计
    report = result['analysis_report']
    print(f"  Tokens: {len(report['summary']['tokens'])}")
    print(f"  追踪目标: {len(report['token_analysis'])}")

# 比较不同句子中相似词汇的表示
print("\n跨句子词汇比较分析...")
```

### 示例 8: 长文本滑动窗口分析

```python
# 处理长文本的滑动窗口分析
long_text = """
Machine learning has revolutionized artificial intelligence. 
Deep neural networks can process vast amounts of data efficiently.
Natural language processing enables computers to understand human communication.
"""

# 分句处理
sentences = [s.strip() for s in long_text.strip().split('.') if s.strip()]

window_results = []
for i, sentence in enumerate(sentences):
    if sentence:
        print(f"\n窗口 {i+1}: {sentence}")
        
        result = inspector.analyze_text(
            text=sentence,
            target_positions=[0, -1],  # 分析首尾 tokens
            export_formats=["json"],
            output_prefix=f"window_{i}"
        )
        
        window_results.append(result)

print(f"\n完成 {len(window_results)} 个窗口的分析")
```

## 可视化与报告

### 示例 9: 生成分析报告

```python
# 生成详细的可视化报告
text = "Deep learning models understand complex patterns in data"
result = inspector.analyze_text(
    text=text,
    target_tokens=["Deep", "learning", "models", "understand", "patterns"],
    export_formats=["json", "csv", "npz"]
)

# 生成可视化
inspector.visualize_results(result, save_plots=True)

# 打印详细报告
report = result['analysis_report']
print("\n=== 详细分析报告 ===")
print(f"输入文本: {report['summary']['input_text']}")
print(f"总 tokens: {report['summary']['total_tokens']}")
print(f"追踪目标: {len(report['token_analysis'])}")

print("\n--- Token 轨迹分析 ---")
for pos_key, analysis in report['token_analysis'].items():
    print(f"\n{pos_key} → '{analysis['token_text']}':")
    print(f"  轨迹层数: {len(analysis['trajectory_layers'])}")
    print(f"  隐藏状态维度: {analysis['hidden_state_shapes']}")
    if 'hidden_state_norms' in analysis:
        norms = analysis['hidden_state_norms'][:5]
        print(f"  前5层范数: {[f'{n:.3f}' for n in norms]}")
```

## 性能优化与调试

### 示例 10: 内存和性能监控

```python
import psutil
import time
from token_tracker import TokenTracker

def monitor_analysis(text, target_tokens):
    """监控分析过程的性能"""
    
    # 记录开始状态
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"开始分析: {text[:50]}...")
    print(f"初始内存: {start_memory:.1f} MB")
    
    # 执行分析
    result = inspector.analyze_text(
        text=text,
        target_tokens=target_tokens,
        export_formats=["json"]
    )
    
    # 记录结束状态
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"分析完成!")
    print(f"用时: {end_time - start_time:.2f} 秒")
    print(f"内存增长: {end_memory - start_memory:.1f} MB")
    print(f"最终内存: {end_memory:.1f} MB")
    
    return result

# 测试不同长度文本的性能
test_cases = [
    ("Short text analysis", ["text"]),
    ("This is a medium length sentence for testing the analysis performance", ["medium", "testing", "performance"]),
    ("This is a much longer text that contains many more words and should provide a good test case for analyzing the performance characteristics of our transformer analysis tool when processing longer sequences", ["longer", "words", "performance", "characteristics", "sequences"])
]

for text, targets in test_cases:
    print("\n" + "="*60)
    monitor_analysis(text, targets)
```

## 运行示例

要运行这些示例，您可以：

### 运行单个示例
```bash
cd /home/cc/poc
python examples.py 1  # 运行示例1
python examples.py 2  # 运行示例2
# ... 等等
```

### 运行所有示例
```bash
python examples.py
```

### 在交互式环境中
```python
# 在 Python REPL 或 Jupyter 中
from examples import *

# 运行特定示例
example_1_basic_analysis()
example_2_attention_analysis()

# 或运行所有示例
run_all_examples()
```

## 输出文件

所有分析结果会保存在 `./outputs/` 目录中，包括：

- **JSON 文件**: 完整的分析数据和元数据
- **CSV 文件**: 表格格式的 token 轨迹数据  
- **NPZ 文件**: NumPy 数组格式的张量数据
- **HDF5 文件**: 分层数据格式（如果启用）
- **PNG 图像**: 可视化图表（注意力热图、轨迹图等）

## 自定义扩展

这个工具设计为模块化，您可以：

1. **添加新的模型支持**: 在 `ModelLoader` 中添加模型类型
2. **自定义分析指标**: 扩展 `TokenTracker` 的比较功能  
3. **新的导出格式**: 在 `DataExporter` 中添加格式支持
4. **自定义可视化**: 扩展 `TransformerInspector` 的绘图功能

每个模块都有清晰的接口，便于扩展和自定义！