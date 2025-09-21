"""
中间层 Hook 系统
用于捕获 Transformer 模型各层的中间激活值、注意力权重等
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ActivationHook:
    """
    单个激活值捕获器
    """
    
    def __init__(self, name: str, capture_input: bool = True, capture_output: bool = True):
        self.name = name
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.activations = {}
        self.handles = []
        
    def __call__(self, module, input_data, output_data):
        """Hook 函数，在前向传播时被调用"""
        if self.capture_input and input_data is not None:
            # 处理输入（可能是元组）
            if isinstance(input_data, tuple):
                input_tensors = []
                for i, inp in enumerate(input_data):
                    if isinstance(inp, torch.Tensor):
                        input_tensors.append(inp.detach().cpu().clone())
                self.activations[f"{self.name}_input"] = input_tensors
            elif isinstance(input_data, torch.Tensor):
                self.activations[f"{self.name}_input"] = input_data.detach().cpu().clone()
        
        if self.capture_output and output_data is not None:
            # 处理输出（可能是元组或自定义对象）
            if isinstance(output_data, tuple):
                output_tensors = []
                for i, out in enumerate(output_data):
                    if isinstance(out, torch.Tensor):
                        output_tensors.append(out.detach().cpu().clone())
                self.activations[f"{self.name}_output"] = output_tensors
            elif isinstance(output_data, torch.Tensor):
                self.activations[f"{self.name}_output"] = output_data.detach().cpu().clone()
            elif hasattr(output_data, 'last_hidden_state'):
                # 处理 Transformer 输出
                self.activations[f"{self.name}_output"] = output_data.last_hidden_state.detach().cpu().clone()
                if hasattr(output_data, 'attentions') and output_data.attentions is not None:
                    self.activations[f"{self.name}_attentions"] = [
                        attn.detach().cpu().clone() for attn in output_data.attentions
                    ]
    
    def clear(self):
        """清空捕获的激活值"""
        self.activations.clear()


class AttentionHook:
    """
    专门用于捕获注意力权重的 Hook
    """
    
    def __init__(self, name: str, layer_idx: int):
        self.name = name
        self.layer_idx = layer_idx
        self.attention_weights = {}
        
    def __call__(self, module, input_data, output_data):
        """捕获注意力权重"""
        if hasattr(output_data, 'attentions') and output_data.attentions is not None:
            # 对于返回注意力权重的层
            if len(output_data.attentions) > self.layer_idx:
                self.attention_weights[f"{self.name}_layer_{self.layer_idx}"] = \
                    output_data.attentions[self.layer_idx].detach().cpu().clone()
        elif isinstance(output_data, tuple) and len(output_data) > 1:
            # 有些模型直接返回注意力权重作为第二个元素
            if isinstance(output_data[1], torch.Tensor):
                self.attention_weights[f"{self.name}_layer_{self.layer_idx}"] = \
                    output_data[1].detach().cpu().clone()


class IntermediateInspector:
    """
    中间状态检查器主类
    管理所有的 Hook 并提供统一的接口
    """
    
    def __init__(self, model, capture_attention: bool = True):
        self.model = model
        self.capture_attention = capture_attention
        self.hooks = {}
        self.attention_hooks = {}
        self.handles = []
        self.layer_mapping = self._build_layer_mapping()
        
    def _build_layer_mapping(self) -> Dict[str, nn.Module]:
        """构建层名到模块的映射"""
        layer_mapping = {}
        
        def add_module_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                layer_mapping[full_name] = child
                add_module_recursive(child, full_name)
        
        add_module_recursive(self.model)
        return layer_mapping
    
    def get_available_layers(self) -> List[str]:
        """获取所有可用的层名"""
        return list(self.layer_mapping.keys())
    
    def register_hooks(self, 
                      layer_patterns: Optional[List[str]] = None,
                      capture_types: Optional[List[str]] = None) -> None:
        """
        注册 Hook 到指定的层
        
        Args:
            layer_patterns: 要监控的层模式列表，如 ['transformer.h.0', 'transformer.h.1']
                          如果为 None，则监控所有层
            capture_types: 要捕获的类型，如 ['attention', 'mlp', 'ln']
        """
        if layer_patterns is None:
            # 默认监控所有 Transformer 层
            layer_patterns = []
            for layer_name in self.layer_mapping.keys():
                if any(pattern in layer_name.lower() for pattern in 
                      ['transformer', 'layer', 'block', 'encoder', 'decoder']):
                    layer_patterns.append(layer_name)
        
        if capture_types is None:
            capture_types = ['all']
        
        logger.info(f"注册 Hook 到 {len(layer_patterns)} 个层")
        
        for layer_name in layer_patterns:
            if layer_name in self.layer_mapping:
                module = self.layer_mapping[layer_name]
                
                # 判断是否需要捕获这个层
                should_capture = 'all' in capture_types
                if not should_capture:
                    layer_lower = layer_name.lower()
                    should_capture = any(
                        (ctype == 'attention' and any(pattern in layer_lower for pattern in ['attn', 'attention'])) or
                        (ctype == 'mlp' and any(pattern in layer_lower for pattern in ['mlp', 'feed_forward', 'ffn'])) or
                        (ctype == 'ln' and any(pattern in layer_lower for pattern in ['ln', 'norm', 'layer_norm'])) or
                        (ctype == 'embedding' and any(pattern in layer_lower for pattern in ['embed', 'wte', 'wpe']))
                        for ctype in capture_types
                    )
                
                if should_capture:
                    # 注册激活值 Hook
                    hook = ActivationHook(layer_name)
                    handle = module.register_forward_hook(hook)
                    self.hooks[layer_name] = hook
                    self.handles.append(handle)
                    
                    # 如果需要捕获注意力权重
                    if self.capture_attention and 'attn' in layer_name.lower():
                        layer_idx = self._extract_layer_index(layer_name)
                        if layer_idx is not None:
                            attn_hook = AttentionHook(layer_name, layer_idx)
                            attn_handle = module.register_forward_hook(attn_hook)
                            self.attention_hooks[layer_name] = attn_hook
                            self.handles.append(attn_handle)
    
    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """从层名中提取层索引"""
        import re
        match = re.search(r'\.(\d+)\.', layer_name)
        if match:
            return int(match.group(1))
        return None
    
    def capture_forward_pass(self, 
                           inputs: Dict[str, torch.Tensor],
                           output_attentions: bool = True) -> Dict[str, Any]:
        """
        执行前向传播并捕获中间状态
        
        Args:
            inputs: 模型输入，通常包含 input_ids, attention_mask 等
            output_attentions: 是否输出注意力权重
        
        Returns:
            包含模型输出和所有捕获的中间状态的字典
        """
        # 清空之前的激活值
        self.clear_activations()
        
        # 设置模型输出注意力权重
        original_output_attentions = getattr(self.model.config, 'output_attentions', False)
        if output_attentions:
            self.model.config.output_attentions = True
        
        try:
            # 执行前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 收集激活值
            activations = self.get_all_activations()
            
            # 收集注意力权重
            attention_weights = self.get_all_attention_weights()
            
            # 恢复原始设置
            self.model.config.output_attentions = original_output_attentions
            
            return {
                'model_outputs': outputs,
                'activations': activations,
                'attention_weights': attention_weights,
                'layer_mapping': list(self.hooks.keys())
            }
            
        except Exception as e:
            self.model.config.output_attentions = original_output_attentions
            raise e
    
    def get_all_activations(self) -> Dict[str, Any]:
        """获取所有捕获的激活值"""
        all_activations = {}
        for layer_name, hook in self.hooks.items():
            all_activations[layer_name] = hook.activations.copy()
        return all_activations
    
    def get_all_attention_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有注意力权重"""
        all_attention = {}
        for layer_name, hook in self.attention_hooks.items():
            all_attention.update(hook.attention_weights)
        return all_attention
    
    def clear_activations(self) -> None:
        """清空所有激活值"""
        for hook in self.hooks.values():
            hook.clear()
        for hook in self.attention_hooks.values():
            hook.attention_weights.clear()
    
    def remove_hooks(self) -> None:
        """移除所有 Hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hooks.clear()
        self.attention_hooks.clear()
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有层的详细信息"""
        layer_info = {}
        
        for layer_name, module in self.layer_mapping.items():
            info = {
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'has_bias': any(hasattr(module, attr) and getattr(module, attr) is not None 
                              for attr in ['bias']),
                'children': len(list(module.children()))
            }
            
            # 添加特定模块的信息
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                info['input_dim'] = module.in_features
                info['output_dim'] = module.out_features
            
            if hasattr(module, 'num_heads'):
                info['num_heads'] = module.num_heads
            
            if hasattr(module, 'head_dim'):
                info['head_dim'] = module.head_dim
            
            layer_info[layer_name] = info
        
        return layer_info
    
    def __del__(self):
        """析构函数，确保 Hook 被清理"""
        self.remove_hooks()


# 辅助函数
def find_transformer_layers(model) -> List[str]:
    """自动查找 Transformer 层"""
    transformer_layers = []
    
    for name, module in model.named_modules():
        # 常见的 Transformer 层模式
        if any(pattern in name.lower() for pattern in [
            'transformer.h.', 'encoder.layer.', 'decoder.layer.',
            'layers.', 'block.', 'transformer_blocks.'
        ]):
            transformer_layers.append(name)
    
    return transformer_layers


def print_model_structure(model, max_depth: int = 3):
    """打印模型结构"""
    def print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print("  " * depth + f"├── {name}: {type(child).__name__}")
            
            # 添加一些有用的信息
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                print("  " * depth + f"    ({child.in_features} → {child.out_features})")
            elif hasattr(child, 'num_heads'):
                print("  " * depth + f"    (heads: {child.num_heads})")
            
            if depth < max_depth:
                print_module(child, full_name, depth + 1)
    
    print(f"模型结构 (最大深度: {max_depth}):")
    print(f"根模块: {type(model).__name__}")
    print_module(model)


if __name__ == "__main__":
    # 测试代码
    from model_loader import ModelLoader
    
    # 加载模型
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    # 创建检查器
    inspector = IntermediateInspector(loader.model)
    
    # 打印模型结构
    print_model_structure(loader.model, max_depth=2)
    
    # 显示可用层
    print(f"\n可用层 ({len(inspector.get_available_layers())} 个):")
    for layer in inspector.get_available_layers()[:10]:  # 只显示前10个
        print(f"  - {layer}")
    
    # 注册 Hook
    inspector.register_hooks()
    
    # 测试捕获
    text = "Hello world!"
    inputs = loader.tokenize_text(text)
    
    result = inspector.capture_forward_pass(inputs)
    print(f"\n捕获的层数: {len(result['activations'])}")
    print(f"注意力权重层数: {len(result['attention_weights'])}")
    
    # 清理
    inspector.remove_hooks()