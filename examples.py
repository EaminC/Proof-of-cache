"""
使用示例
展示如何使用 VLLM 中间状态检查器的各种功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入自定义模块
from model_loader import ModelLoader
from token_tracker import TokenTracker
from coordinate_system import CoordinateManager, DataExporter
from main import VLLMInspector


def example_1_basic_analysis():
    """示例 1: 基本分析流程"""
    print("="*60)
    print("示例 1: 基本的 Token 分析")
    print("="*60)
    
    # 创建检查器
    inspector = VLLMInspector()
    
    # 加载模型
    inspector.load_model("gpt2")
    
    # 分析文本
    text = "The quick brown fox jumps over the lazy dog."
    target_tokens = ["quick", "fox", "jumps"]
    
    result = inspector.analyze_text(
        text=text,
        target_tokens=target_tokens,
        export_formats=["json", "csv"]
    )
    
    # 显示结果
    report = result['analysis_report']
    print(f"输入文本: {report['summary']['input_text']}")
    print(f"Token 列表: {report['summary']['tokens']}")
    print(f"追踪的位置: {report['summary']['tracked_positions']}")
    
    # 显示 token 分析
    for pos_key, analysis in report['token_analysis'].items():
        print(f"\n{pos_key} ('{analysis['token_text']}'):")
        print(f"  轨迹层数: {len(analysis['trajectory_layers'])}")
        print(f"  隐藏状态范数: {analysis['hidden_state_norms'][:3]}...")
    
    return result


def example_2_attention_analysis():
    """示例 2: 注意力分析"""
    print("\n" + "="*60)
    print("示例 2: 注意力模式分析")
    print("="*60)
    
    # 创建加载器和追踪器
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    tracker = TokenTracker(loader.model, loader.tokenizer)
    
    # 分析一个句子的注意力模式
    text = "I love natural language processing."
    inputs = loader.tokenize_text(text)
    
    # 追踪所有 token
    result = tracker.trace_tokens(inputs, trace_all=True)
    
    token_info = loader.get_token_info(inputs['input_ids'])
    print(f"分析文本: {text}")
    print(f"Token 分解: {token_info['tokens']}")
    
    # 分析特定位置的注意力流
    target_pos = 2  # "natural" 的位置
    if target_pos < len(token_info['tokens']):
        attention_flow = tracker.get_attention_flow(target_pos)
        
        print(f"\nToken '{token_info['tokens'][target_pos]}' (位置 {target_pos}) 的注意力流:")
        
        # 显示出向注意力（该 token 注意其他 token）
        for layer, attn_info in list(attention_flow['outgoing_attention'].items())[:3]:
            print(f"  层 {layer} - 最关注的位置: {attn_info['top_attended']}")
        
        # 显示入向注意力（其他 token 注意该 token）
        for layer, attn_info in list(attention_flow['incoming_attention'].items())[:3]:
            print(f"  层 {layer} - 最被关注的来源: {attn_info['top_attending']}")


def example_3_token_comparison():
    """示例 3: Token 比较分析"""
    print("\n" + "="*60)
    print("示例 3: Token 相似性比较")
    print("="*60)
    
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    tracker = TokenTracker(loader.model, loader.tokenizer)
    
    # 比较同义词的内部表示
    text = "The cat and the kitten are playing together."
    inputs = loader.tokenize_text(text)
    
    result = tracker.trace_tokens(inputs, trace_all=True)
    token_info = loader.get_token_info(inputs['input_ids'])
    
    print(f"分析文本: {text}")
    print(f"Token 列表: {token_info['tokens']}")
    
    # 找到 "cat" 和 "kitten" 的位置
    cat_pos = None
    kitten_pos = None
    
    for i, token in enumerate(token_info['tokens']):
        if 'cat' in token.lower():
            cat_pos = i
        elif 'kitten' in token.lower():
            kitten_pos = i
    
    if cat_pos is not None and kitten_pos is not None:
        # 比较这两个 token
        comparison = tracker.compare_tokens(cat_pos, kitten_pos, metric='cosine')
        
        print(f"\n比较 'cat' (位置 {cat_pos}) 和 'kitten' (位置 {kitten_pos}):")
        print("各层的余弦相似度:")
        
        for layer in sorted(comparison['similarities'].keys())[:5]:
            sim = comparison['similarities'][layer]
            print(f"  层 {layer}: {sim:.4f}")
    
    return result


def example_4_coordinate_system():
    """示例 4: 坐标系统和数据导出"""
    print("\n" + "="*60)
    print("示例 4: 坐标系统和数据导出")
    print("="*60)
    
    # 创建坐标管理器
    manager = CoordinateManager("gpt2")
    
    # 模拟一些中间变量
    print("创建模拟的中间变量...")
    
    for layer in range(3):
        for seq_pos in range(4):
            # 隐藏状态
            coord = manager.create_coordinate(
                batch_idx=0,
                sequence_idx=seq_pos,
                layer_idx=layer,
                module_type="hidden_state",
                module_name=f"transformer.h.{layer}"
            )
            
            value = torch.randn(768)  # GPT-2 hidden size
            manager.add_variable(
                coord, 
                value, 
                f"Hidden state at layer {layer}, position {seq_pos}"
            )
            
            # 注意力权重（每个头）
            for head in range(2):  # 简化，只用2个头
                coord = manager.create_coordinate(
                    batch_idx=0,
                    sequence_idx=seq_pos,
                    layer_idx=layer,
                    head_idx=head,
                    module_type="attention",
                    module_name=f"transformer.h.{layer}.attn"
                )
                
                value = torch.softmax(torch.randn(4), dim=0)  # attention to 4 positions
                manager.add_variable(
                    coord,
                    value,
                    f"Attention weights from layer {layer}, head {head}, position {seq_pos}"
                )
    
    # 显示摘要
    summary = manager.summary()
    print(f"总变量数: {summary['total_variables']}")
    print(f"层分布: {summary['layer_distribution']}")
    print(f"模块类型分布: {summary['module_type_distribution']}")
    
    # 过滤操作示例
    print("\n过滤操作示例:")
    
    # 获取第0层的所有变量
    layer_0_vars = manager.get_layer_variables(0)
    print(f"第0层变量数: {len(layer_0_vars)}")
    
    # 获取位置1的所有变量
    pos_1_vars = manager.get_position_variables(1)
    print(f"位置1变量数: {len(pos_1_vars)}")
    
    # 获取注意力相关变量
    attn_vars = manager.get_attention_variables()
    print(f"注意力变量数: {len(attn_vars)}")
    
    # 导出数据
    print("\n导出数据...")
    exported = manager.export_all(
        "coordinate_example",
        formats=["json", "npz", "csv", "hdf5"]
    )
    
    print("导出的文件:")
    for fmt, path in exported.items():
        print(f"  {fmt.upper()}: {path}")
    
    return manager


def example_5_custom_analysis():
    """示例 5: 自定义分析"""
    print("\n" + "="*60)
    print("示例 5: 自定义分析流程")
    print("="*60)
    
    # 手动控制分析流程
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    # 准备数据
    text = "Machine learning is revolutionizing artificial intelligence."
    inputs = loader.tokenize_text(text)
    token_info = loader.get_token_info(inputs['input_ids'])
    
    print(f"分析文本: {text}")
    
    # 使用 Hook 系统直接捕获激活
    from hook_system import IntermediateInspector
    
    inspector = IntermediateInspector(loader.model, capture_attention=True)
    
    # 只监控特定层（比如前3层）
    target_layers = [f"transformer.h.{i}" for i in range(3)]
    inspector.register_hooks(layer_patterns=target_layers)
    
    # 执行前向传播
    result = inspector.capture_forward_pass(inputs)
    
    print(f"捕获了 {len(result['activations'])} 个层的激活")
    print(f"捕获了 {len(result['attention_weights'])} 个注意力权重")
    
    # 分析激活值的统计信息
    print("\n激活值统计:")
    for layer_name, activations in result['activations'].items():
        if f"{layer_name}_output" in activations:
            output_tensor = activations[f"{layer_name}_output"]
            if isinstance(output_tensor, list):
                output_tensor = output_tensor[0]
            
            if output_tensor is not None and output_tensor.numel() > 0:
                stats = {
                    'mean': output_tensor.mean().item(),
                    'std': output_tensor.std().item(),
                    'min': output_tensor.min().item(),
                    'max': output_tensor.max().item()
                }
                print(f"  {layer_name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    
    # 清理
    inspector.remove_hooks()
    
    return result


def example_6_visualization():
    """示例 6: 可视化分析"""
    print("\n" + "="*60)
    print("示例 6: 数据可视化")
    print("="*60)
    
    # 使用主检查器
    inspector = VLLMInspector()
    inspector.load_model("gpt2")
    
    # 分析一个有趣的句子
    text = "The scientist discovered a new planet orbiting a distant star."
    result = inspector.analyze_text(
        text=text,
        target_tokens=["scientist", "discovered", "planet"],
        export_formats=["json"]
    )
    
    # 生成可视化
    inspector.visualize_results(result, save_plots=True)
    
    print("可视化图表已生成并保存！")
    
    return result


def run_all_examples():
    """运行所有示例"""
    print("VLLM 中间状态检查器 - 使用示例")
    print("="*80)
    
    try:
        # 确保输出目录存在
        Path("./outputs").mkdir(exist_ok=True)
        
        # 运行示例
        example_1_basic_analysis()
        example_2_attention_analysis()
        example_3_token_comparison()
        example_4_coordinate_system()
        example_5_custom_analysis()
        example_6_visualization()
        
        print("\n" + "="*80)
        print("所有示例已完成！")
        print("检查 ./outputs 目录查看生成的文件")
        print("="*80)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        raise


if __name__ == "__main__":
    # 可以运行单个示例或所有示例
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_basic_analysis,
            2: example_2_attention_analysis,
            3: example_3_token_comparison,
            4: example_4_coordinate_system,
            5: example_5_custom_analysis,
            6: example_6_visualization
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"示例 {example_num} 不存在")
    else:
        # 运行所有示例
        run_all_examples()