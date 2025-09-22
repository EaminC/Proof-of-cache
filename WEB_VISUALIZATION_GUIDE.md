# 🌐 Web可视化使用指南

## 快速启动

### 方法1: 使用启动脚本（推荐）
```bash
cd /home/cc/poc
chmod +x start_visualizer.sh
./start_visualizer.sh
```

### 方法2: 手动启动
```bash
cd /home/cc/poc
pip install flask flask-cors plotly
python3 web_visualizer.py
```

启动后打开浏览器访问: **http://localhost:5000**

## 📱 界面功能

### 主界面布局
```
┌─────────────────┬─────────────────┐
│   输入区域      │   模型结构区    │
│ (可点击tokens)  │  (层级可视化)   │
├─────────────────┼─────────────────┤
│   控制面板      │   分析结果区    │
│ (参数设置)      │ (图表和数据)    │
└─────────────────┴─────────────────┘
```

### 🎯 核心交互功能

#### 1. Token点击高亮
- **操作**: 点击输入文本中的任意token
- **效果**: 
  - 该token会高亮显示
  - 模型结构中相关的层会被标记
  - 右侧显示该token的中间变量轨迹
  - 注意力连接线会显示

#### 2. 模型结构交互
- **操作**: 点击模型层级图中的任意层
- **效果**:
  - 显示该层的详细信息
  - 高亮显示与当前选中token的连接
  - 显示该层的激活热图

#### 3. 动态分析
- **操作**: 在输入框中输入新文本
- **效果**: 实时分析并更新所有可视化

## 🛠️ 使用步骤

### 步骤1: 启动服务
```bash
./start_visualizer.sh
```
看到以下输出表示启动成功：
```
🚀 启动Transformer可视化服务...
📦 检查依赖...
✅ 所有依赖已满足
🌐 启动Web服务器...
 * Running on http://127.0.0.1:5000
```

### 步骤2: 打开浏览器
访问 http://localhost:5000

### 步骤3: 输入文本
在文本框中输入要分析的句子，例如：
```
"The cat sits on the mat"
```

### 步骤4: 设置分析参数
- **目标Token**: 选择要重点分析的词汇
- **模型**: 选择使用的Transformer模型
- **分析深度**: 选择要分析的层数

### 步骤5: 点击"分析"按钮
系统会开始分析并生成可视化

### 步骤6: 交互探索
- 点击tokens查看中间状态
- 点击模型层查看详细信息
- 鼠标悬停查看提示信息

## 🎨 可视化组件说明

### 1. Token序列显示
```html
[The] [cat] [sits] [on] [the] [mat]
 ↑     ↑     ↑     ↑     ↑     ↑
点击任意token查看其在模型中的处理过程
```

### 2. 模型结构图
```
输入层 → Embedding → 
Layer 0 → Layer 1 → ... → Layer 11 → 
输出层
```
每一层都可以点击查看详情

### 3. 注意力热图
显示tokens之间的注意力权重，颜色越深表示注意力越强

### 4. 隐藏状态轨迹
显示选中token在各层的隐藏状态变化曲线

### 5. 3D可视化
提供token在高维空间中的位置展示

## 📊 数据展示功能

### 实时数据面板
- **Token信息**: 位置、文本、ID
- **层级信息**: 当前层、激活值、权重
- **注意力数据**: 注意力分布、权重矩阵
- **相似度分析**: token间的相似度计算

### 导出功能
- 点击"导出数据"可下载分析结果
- 支持JSON、CSV、PNG格式
- 包含完整的中间状态数据

## 🔧 高级功能

### 1. 对比分析
```javascript
// 可以同时分析多个句子进行对比
输入1: "I love this product"
输入2: "I hate this product"
```

### 2. 自定义可视化
在控制面板中可以：
- 调整颜色方案
- 选择显示的层级范围
- 设置注意力阈值
- 自定义图表类型

### 3. 实时更新
修改输入文本时，界面会实时更新，无需重新点击分析

## 🐛 故障排除

### 常见问题

#### 1. 启动失败
```bash
# 检查Python版本
python3 --version  # 需要 >= 3.7

# 安装缺失依赖
pip install flask flask-cors plotly transformers torch
```

#### 2. 浏览器无法访问
- 检查防火墙设置
- 确认端口5000未被占用
- 尝试使用 http://127.0.0.1:5000

#### 3. 分析速度慢
- 使用较短的文本进行测试
- 选择较小的模型（如distilgpt2）
- 确保有足够的内存

#### 4. 界面显示异常
- 刷新页面（F5）
- 清除浏览器缓存
- 检查JavaScript控制台错误

### 性能优化

#### 服务器端
```bash
# 使用GPU加速（如果可用）
export CUDA_VISIBLE_DEVICES=0
python3 web_visualizer.py
```

#### 浏览器端
- 使用现代浏览器（Chrome、Firefox、Safari）
- 确保JavaScript已启用
- 关闭不必要的浏览器扩展

## 📱 移动端使用

虽然主要为桌面设计，但也支持移动端：
- 响应式布局自动适配
- 触摸操作支持
- 简化的交互界面

## 🔗 API接口

如果需要编程访问，可以直接调用API：

```python
import requests

# 分析文本
response = requests.post('http://localhost:5000/api/analyze', json={
    'text': 'Hello world',
    'target_tokens': ['Hello', 'world']
})

result = response.json()
```

### 可用API端点

- `POST /api/analyze` - 分析文本
- `GET /api/models` - 获取可用模型列表
- `POST /api/compare` - 对比分析
- `GET /api/health` - 服务健康检查

## 🎉 开始使用

现在就开始探索Transformer的内部世界吧！

1. 启动服务: `./start_visualizer.sh`
2. 打开浏览器: http://localhost:5000
3. 输入文本，点击tokens，探索模型！

享受可视化分析的乐趣！ 🚀