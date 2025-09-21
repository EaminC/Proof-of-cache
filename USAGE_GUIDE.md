# 🚀 Transformer 中间状态分析器 - 完整示例集合

您现在拥有一个功能强大的 Transformer 模型中间状态分析工具！以下是各种使用示例和实际应用场景。

## 📚 可用示例文件

### 1. **基础示例** (`examples.py`)
包含6个核心功能演示：

```bash
python3 examples.py 1    # 基础文本分析
python3 examples.py 2    # 注意力分析  
python3 examples.py 3    # Token 比较
python3 examples.py 4    # 坐标系统
python3 examples.py 5    # 自定义分析
python3 examples.py 6    # 可视化
python3 examples.py      # 运行所有示例
```

### 2. **实际应用演示** (`real_world_demos.py`)
包含6个真实场景应用：

```bash
python3 real_world_demos.py 1    # 情感分析
python3 real_world_demos.py 2    # 词汇相似性
python3 real_world_demos.py 3    # 注意力模式
python3 real_world_demos.py 4    # 语言现象
python3 real_world_demos.py 5    # 多语言分析
python3 real_world_demos.py 6    # 性能分析
python3 real_world_demos.py      # 运行所有演示
```

## 🎯 快速开始示例

### 最简单的使用方式
```python
from main import TransformerInspector

# 创建分析器并加载模型
inspector = TransformerInspector()
inspector.load_model("gpt2")

# 分析文本
result = inspector.analyze_text(
    text="Hello transformer analysis world!",
    target_tokens=["transformer", "analysis"],
    export_formats=["json", "csv"]
)

# 查看结果
print(f"分析了 {result['analysis_report']['summary']['total_tokens']} 个 tokens")
```

### 刚才运行的情感分析示例结果
```
🎭 情感分析结果:
正面句子: "I absolutely love this amazing product!"
- ' love' (正面): 均值范数 = 147.721
- ' amazing' (正面): 均值范数 = 137.936

负面句子: "I completely hate this terrible product!"  
- ' hate' (负面): 均值范数 = 143.799
- ' terrible' (负面): 均值范数 = 129.263
```

## 🔥 推荐使用场景

### 1. **NLP 研究**
- 分析模型如何处理不同类型的词汇
- 研究注意力模式和语法关系
- 比较同义词在不同上下文中的表示

### 2. **模型调试**
- 定位模型在处理特定输入时的问题
- 分析中间层的激活模式
- 监控模型性能和资源使用

### 3. **教育和学习**
- 可视化 Transformer 的内部工作原理
- 理解注意力机制的实际效果
- 探索词汇语义的向量表示

### 4. **模型比较**
- 对比不同模型对相同输入的处理
- 分析模型版本间的差异
- 评估模型在特定任务上的表现

## 📊 输出文件说明

运行示例后，在 `./outputs/` 目录中您会找到：

```
outputs/
├── analysis_unknown.json.gz    # 完整分析结果 (JSON压缩格式)
├── analysis_unknown.csv        # Token轨迹数据 (表格格式)  
├── analysis_unknown.npz        # 张量数据 (NumPy格式)
├── attention_heatmap.png        # 注意力热图 (如果生成可视化)
└── token_trajectory.png        # Token轨迹图 (如果生成可视化)
```

### JSON 文件结构
```json
{
  "metadata": {
    "model_type": "gpt2",
    "total_tokens": 7,
    "analysis_timestamp": "..."
  },
  "token_analysis": {
    "position_2": {
      "token_text": " love",
      "trajectory_layers": [0, 1, 2, ...],
      "hidden_state_norms": [55.68, 55.99, ...],
      "attention_weights": {...}
    }
  },
  "coordinate_data": {...}
}
```

## 🛠️ 高级自定义

### 自定义 Hook 系统
```python
from hook_system import IntermediateInspector

# 只监控特定层
inspector = IntermediateInspector(model, capture_attention=True)
inspector.register_hooks(layer_patterns=["transformer.h.0", "transformer.h.5"])

# 执行分析
result = inspector.capture_forward_pass(inputs)
```

### 精确坐标管理
```python
from coordinate_system import CoordinateManager

manager = CoordinateManager("gpt2")

# 为特定位置创建坐标
coord = manager.create_coordinate(
    batch_idx=0, sequence_idx=2, layer_idx=5,
    module_type="attention", head_idx=3
)

# 存储变量
manager.add_variable(coord, tensor_data, "描述")
```

### 批量分析
```python
texts = [
    "Positive sentiment text",
    "Negative sentiment text", 
    "Neutral sentiment text"
]

results = []
for text in texts:
    result = inspector.analyze_text(text=text, target_tokens=["sentiment"])
    results.append(result)
```

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 使用较短的文本进行测试
   - 减少 `export_formats` 选项
   - 确保有足够的 GPU 内存

2. **找不到目标 Token**
   - 检查 token 的确切拼写（注意空格）
   - 使用 `target_tokens=None` 分析所有 tokens
   - 查看 token 分解结果：`token_info['tokens']`

3. **导出文件过大**
   - 使用 `["json"]` 而不是所有格式
   - 限制分析的 token 数量
   - 检查 `.gitignore` 避免提交大文件

### 性能优化建议

```python
# 使用较小的模型进行测试
inspector.load_model("distilgpt2")  # 更小更快

# 只分析关键 tokens
result = inspector.analyze_text(
    text=long_text,
    target_tokens=["关键词1", "关键词2"],  # 而不是所有 tokens
    export_formats=["json"]  # 只使用必要的格式
)
```

## 📈 扩展功能

这个工具设计为模块化，您可以轻松扩展：

1. **添加新模型**: 在 `ModelLoader` 中支持更多模型类型
2. **自定义指标**: 扩展 `TokenTracker` 的分析功能
3. **新导出格式**: 在 `DataExporter` 中添加格式支持  
4. **可视化增强**: 扩展绘图和报告功能

## 🎉 开始探索！

现在您有了完整的工具和示例集合！建议从以下步骤开始：

1. **运行基础示例**: `python3 examples.py 1`
2. **尝试实际应用**: `python3 real_world_demos.py 1`  
3. **分析您自己的文本**: 修改示例中的 `text` 变量
4. **探索输出文件**: 查看 `./outputs/` 目录
5. **自定义分析**: 根据您的需求修改参数

祝您使用愉快！🚀