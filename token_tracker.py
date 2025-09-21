"""
Token 追踪器
专门用于追踪特定输入/输出 token 在模型各层的中间状态
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TokenCoordinate:
    """
    Token 坐标，用于精确定位 token 在模型中的位置
    """
    batch_idx: int = 0      # batch 中的索引
    sequence_idx: int = 0   # 序列中的位置
    layer_idx: Optional[int] = None     # 层索引
    head_idx: Optional[int] = None      # 注意力头索引
    token_id: Optional[int] = None      # Token ID
    token_text: Optional[str] = None    # Token 文本


@dataclass
class TokenState:
    """
    Token 在某一层的状态信息
    """
    coordinate: TokenCoordinate
    hidden_state: torch.Tensor          # 隐藏状态向量
    attention_weights: Optional[torch.Tensor] = None  # 注意力权重
    key_vector: Optional[torch.Tensor] = None      # Key 向量
    query_vector: Optional[torch.Tensor] = None    # Query 向量
    value_vector: Optional[torch.Tensor] = None    # Value 向量
    layer_norm_output: Optional[torch.Tensor] = None  # LayerNorm 输出
    mlp_output: Optional[torch.Tensor] = None      # MLP 输出
    residual_connection: Optional[torch.Tensor] = None  # 残差连接


class TokenTracker:
    """
    Token 追踪器主类
    跟踪特定 token 在整个模型前向传播过程中的状态变化
    """
    
    def __init__(self, 
                 model,
                 tokenizer,
                 target_tokens: Optional[List[Union[int, str]]] = None):
        """
        初始化 Token 追踪器
        
        Args:
            model: Transformer 模型
            tokenizer: 分词器
            target_tokens: 要追踪的目标 token（ID 或文本）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens or []
        self.token_states = defaultdict(list)  # {token_coordinate: [TokenState]}
        self.layer_outputs = {}
        self.attention_patterns = {}
        
    def set_target_tokens(self, tokens: List[Union[int, str]]) -> None:
        """设置要追踪的目标 token"""
        self.target_tokens = tokens
        
    def add_target_token(self, token: Union[int, str]) -> None:
        """添加要追踪的 token"""
        if token not in self.target_tokens:
            self.target_tokens.append(token)
    
    def trace_tokens(self, 
                    inputs: Dict[str, torch.Tensor],
                    trace_all: bool = False) -> Dict[str, Any]:
        """
        追踪 token 在模型中的状态变化
        
        Args:
            inputs: 模型输入
            trace_all: 是否追踪所有 token（而不仅仅是目标 token）
        
        Returns:
            包含所有追踪信息的字典
        """
        from hook_system import IntermediateInspector
        
        # 清空之前的状态
        self.token_states.clear()
        self.layer_outputs.clear()
        self.attention_patterns.clear()
        
        # 创建检查器并注册 Hook
        inspector = IntermediateInspector(self.model, capture_attention=True)
        inspector.register_hooks()
        
        try:
            # 执行前向传播并捕获中间状态
            result = inspector.capture_forward_pass(inputs, output_attentions=True)
            
            # 解析输入 token
            input_ids = inputs['input_ids'].squeeze(0)  # 假设 batch_size=1
            token_texts = [self.tokenizer.decode([tid.item()]) for tid in input_ids]
            
            # 确定要追踪的 token 位置
            target_positions = self._find_target_positions(input_ids, token_texts, trace_all)
            
            # 分析每一层的状态
            self._analyze_layer_states(result['activations'], target_positions, token_texts)
            
            # 分析注意力模式
            self._analyze_attention_patterns(result['attention_weights'], target_positions, token_texts)
            
            return {
                'token_states': dict(self.token_states),
                'layer_outputs': self.layer_outputs,
                'attention_patterns': self.attention_patterns,
                'input_tokens': token_texts,
                'target_positions': target_positions,
                'model_outputs': result['model_outputs']
            }
            
        finally:
            # 清理 Hook
            inspector.remove_hooks()
    
    def _find_target_positions(self, 
                              input_ids: torch.Tensor,
                              token_texts: List[str],
                              trace_all: bool) -> List[int]:
        """找到目标 token 的位置"""
        if trace_all:
            return list(range(len(input_ids)))
        
        target_positions = []
        
        for pos, (token_id, token_text) in enumerate(zip(input_ids, token_texts)):
            # 检查是否是目标 token
            for target in self.target_tokens:
                if isinstance(target, int) and token_id.item() == target:
                    target_positions.append(pos)
                    break
                elif isinstance(target, str) and target.lower() in token_text.lower():
                    target_positions.append(pos)
                    break
        
        logger.info(f"找到 {len(target_positions)} 个目标 token 位置: {target_positions}")
        return target_positions
    
    def _analyze_layer_states(self, 
                             activations: Dict[str, Any],
                             target_positions: List[int],
                             token_texts: List[str]) -> None:
        """分析每层的激活状态"""
        for layer_name, layer_activations in activations.items():
            # 处理输出激活
            if f"{layer_name}_output" in layer_activations:
                output_tensor = layer_activations[f"{layer_name}_output"]
                
                # 确保是正确的形状 [batch_size, seq_length, hidden_size]
                if isinstance(output_tensor, list):
                    output_tensor = output_tensor[0]  # 取第一个元素
                
                if output_tensor is not None and output_tensor.dim() >= 3:
                    batch_size, seq_length, hidden_size = output_tensor.shape[:3]
                    
                    # 存储层输出用于后续分析
                    self.layer_outputs[layer_name] = output_tensor
                    
                    # 为每个目标位置创建 TokenState
                    for pos in target_positions:
                        if pos < seq_length:
                            # 创建坐标
                            coordinate = TokenCoordinate(
                                batch_idx=0,
                                sequence_idx=pos,
                                layer_idx=self._extract_layer_index(layer_name),
                                token_text=token_texts[pos] if pos < len(token_texts) else None
                            )
                            
                            # 创建状态
                            token_state = TokenState(
                                coordinate=coordinate,
                                hidden_state=output_tensor[0, pos, :].clone()
                            )
                            
                            # 添加其他相关信息
                            self._enrich_token_state(token_state, layer_activations, pos)
                            
                            # 存储状态
                            key = f"layer_{coordinate.layer_idx}_pos_{pos}"
                            self.token_states[key].append(token_state)
    
    def _analyze_attention_patterns(self, 
                                   attention_weights: Dict[str, torch.Tensor],
                                   target_positions: List[int],
                                   token_texts: List[str]) -> None:
        """分析注意力模式"""
        for layer_name, attn_tensor in attention_weights.items():
            if attn_tensor is None:
                continue
                
            layer_idx = self._extract_layer_index(layer_name)
            
            # 注意力权重形状：[batch_size, num_heads, seq_length, seq_length]
            if attn_tensor.dim() == 4:
                batch_size, num_heads, seq_length, _ = attn_tensor.shape
                
                for pos in target_positions:
                    if pos < seq_length:
                        # 获取该位置作为 query 时的注意力权重
                        query_attention = attn_tensor[0, :, pos, :]  # [num_heads, seq_length]
                        
                        # 获取该位置作为 key 时被注意的权重
                        key_attention = attn_tensor[0, :, :, pos]    # [num_heads, seq_length]
                        
                        self.attention_patterns[f"layer_{layer_idx}_pos_{pos}"] = {
                            'as_query': query_attention.clone(),
                            'as_key': key_attention.clone(),
                            'num_heads': num_heads,
                            'token_text': token_texts[pos] if pos < len(token_texts) else None
                        }
    
    def _enrich_token_state(self, 
                           token_state: TokenState,
                           layer_activations: Dict[str, Any],
                           position: int) -> None:
        """为 TokenState 添加额外的信息"""
        # 这里可以添加更多的激活值信息
        # 例如 LayerNorm 输出、MLP 输出等
        pass
    
    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """从层名中提取层索引"""
        import re
        # 尝试多种模式
        patterns = [
            r'\.(\d+)\.',      # .0. 格式
            r'layer_(\d+)',    # layer_0 格式  
            r'h\.(\d+)',       # h.0 格式（GPT-2）
            r'layers\.(\d+)',  # layers.0 格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, layer_name)
            if match:
                return int(match.group(1))
        
        return None
    
    def get_token_trajectory(self, 
                           token_position: int,
                           layer_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        获取特定 token 在所有层中的轨迹
        
        Args:
            token_position: token 位置
            layer_range: 层范围 (start, end)，如果为 None 则包含所有层
        
        Returns:
            该 token 的完整轨迹信息
        """
        trajectory = {
            'position': token_position,
            'layers': {},
            'attention_evolution': {},
            'hidden_state_evolution': []
        }
        
        # 收集该位置在所有层的状态
        for key, states in self.token_states.items():
            if f"_pos_{token_position}" in key:
                # 更安全的层索引解析
                try:
                    if key.startswith('layer_'):
                        layer_idx = int(key.split('_')[1])
                    else:
                        # 尝试从状态中获取层索引
                        if states and states[0].coordinate.layer_idx is not None:
                            layer_idx = states[0].coordinate.layer_idx
                        else:
                            continue
                except (ValueError, IndexError):
                    continue
                
                if layer_range is None or (layer_range[0] <= layer_idx <= layer_range[1]):
                    if states:
                        state = states[0]  # 取第一个状态
                        trajectory['layers'][layer_idx] = {
                            'hidden_state': state.hidden_state,
                            'coordinate': state.coordinate
                        }
                        trajectory['hidden_state_evolution'].append({
                            'layer': layer_idx,
                            'hidden_state': state.hidden_state
                        })
        
        # 收集注意力进化
        for key, attn_info in self.attention_patterns.items():
            if f"_pos_{token_position}" in key:
                try:
                    if key.startswith('layer_'):
                        layer_idx = int(key.split('_')[1])
                    else:
                        # 尝试从 key 中提取层索引
                        parts = key.split('_')
                        layer_idx = None
                        for i, part in enumerate(parts):
                            if part == 'layer' and i + 1 < len(parts):
                                layer_idx = int(parts[i + 1])
                                break
                        if layer_idx is None:
                            continue
                except (ValueError, IndexError):
                    continue
                    
                if layer_range is None or (layer_range[0] <= layer_idx <= layer_range[1]):
                    trajectory['attention_evolution'][layer_idx] = attn_info
        
        return trajectory
    
    def compare_tokens(self, 
                      position1: int, 
                      position2: int,
                      metric: str = 'cosine') -> Dict[str, Any]:
        """
        比较两个 token 在各层的相似性
        
        Args:
            position1, position2: 要比较的两个 token 位置
            metric: 相似性度量，支持 'cosine', 'euclidean', 'manhattan'
        
        Returns:
            相似性分析结果
        """
        traj1 = self.get_token_trajectory(position1)
        traj2 = self.get_token_trajectory(position2)
        
        similarities = {}
        
        for layer_idx in set(traj1['layers'].keys()) & set(traj2['layers'].keys()):
            hidden1 = traj1['layers'][layer_idx]['hidden_state']
            hidden2 = traj2['layers'][layer_idx]['hidden_state']
            
            if metric == 'cosine':
                sim = torch.cosine_similarity(hidden1, hidden2, dim=0).item()
            elif metric == 'euclidean':
                sim = torch.dist(hidden1, hidden2, p=2).item()
            elif metric == 'manhattan':
                sim = torch.dist(hidden1, hidden2, p=1).item()
            else:
                raise ValueError(f"不支持的相似性度量: {metric}")
            
            similarities[layer_idx] = sim
        
        return {
            'similarities': similarities,
            'metric': metric,
            'positions': [position1, position2],
            'trajectory1': traj1,
            'trajectory2': traj2
        }
    
    def get_attention_flow(self, token_position: int) -> Dict[str, Any]:
        """
        获取特定 token 的注意力流动模式
        
        Args:
            token_position: token 位置
        
        Returns:
            注意力流动分析
        """
        attention_flow = {
            'position': token_position,
            'incoming_attention': {},  # 其他 token 对该 token 的注意力
            'outgoing_attention': {},  # 该 token 对其他 token 的注意力
            'self_attention': {}       # 自注意力权重
        }
        
        for key, attn_info in self.attention_patterns.items():
            if f"_pos_{token_position}" in key:
                layer_idx = int(key.split('_')[1])
                
                # 作为 query 时的注意力分布（出向）
                outgoing = attn_info['as_query']  # [num_heads, seq_length]
                attention_flow['outgoing_attention'][layer_idx] = {
                    'weights': outgoing,
                    'top_attended': torch.topk(outgoing.mean(dim=0), k=5).indices.tolist()
                }
                
                # 作为 key 时被注意的权重（入向）  
                incoming = attn_info['as_key']    # [num_heads, seq_length]
                attention_flow['incoming_attention'][layer_idx] = {
                    'weights': incoming,
                    'top_attending': torch.topk(incoming.mean(dim=0), k=5).indices.tolist()
                }
                
                # 自注意力权重
                self_attn = outgoing[:, token_position]  # [num_heads]
                attention_flow['self_attention'][layer_idx] = self_attn.tolist()
        
        return attention_flow


# 辅助函数
def visualize_token_trajectory(trajectory: Dict[str, Any], 
                              save_path: Optional[str] = None) -> None:
    """可视化 token 轨迹"""
    import matplotlib.pyplot as plt
    
    layers = sorted(trajectory['layers'].keys())
    hidden_states = [trajectory['layers'][layer]['hidden_state'] for layer in layers]
    
    # 计算每层的向量范数
    norms = [state.norm().item() for state in hidden_states]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(layers, norms, 'b-o')
    plt.title(f"Token {trajectory['position']} 隐藏状态范数变化")
    plt.xlabel("层数")
    plt.ylabel("向量范数")
    
    # 如果有注意力信息，绘制自注意力权重
    if trajectory['attention_evolution']:
        plt.subplot(2, 1, 2)
        self_attentions = []
        attn_layers = []
        
        for layer in sorted(trajectory['attention_evolution'].keys()):
            if 'self_attention' in trajectory['attention_evolution'][layer]:
                self_attn = trajectory['attention_evolution'][layer]['self_attention']
                self_attentions.append(np.mean(self_attn))
                attn_layers.append(layer)
        
        if self_attentions:
            plt.plot(attn_layers, self_attentions, 'r-s')
            plt.title("自注意力权重变化")
            plt.xlabel("层数")
            plt.ylabel("平均自注意力权重")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # 测试代码
    from model_loader import ModelLoader
    
    # 加载模型
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    # 创建追踪器
    tracker = TokenTracker(loader.model, loader.tokenizer, target_tokens=["hello", "world"])
    
    # 测试追踪
    text = "Hello world! How are you today?"
    inputs = loader.tokenize_text(text)
    
    print(f"输入文本: {text}")
    print(f"Token 信息: {loader.get_token_info(inputs['input_ids'])}")
    
    # 执行追踪
    result = tracker.trace_tokens(inputs, trace_all=False)
    
    print(f"\n追踪到 {len(result['token_states'])} 个 token 状态")
    print(f"注意力模式: {len(result['attention_patterns'])} 个")
    
    # 获取特定 token 的轨迹
    if result['target_positions']:
        pos = result['target_positions'][0]
        trajectory = tracker.get_token_trajectory(pos)
        print(f"\nToken 位置 {pos} 的轨迹包含 {len(trajectory['layers'])} 层")