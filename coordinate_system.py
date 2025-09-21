"""
坐标系统和数据导出模块
提供精确的坐标定位和多种格式的数据导出功能
"""

import torch
import numpy as np
import pandas as pd
import json
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CoordinateSystem:
    """
    坐标系统，用于精确定位中间变量
    """
    # 基本维度
    batch_idx: int = 0          # Batch 维度
    sequence_idx: int = 0       # 序列位置
    layer_idx: Optional[int] = None      # 层索引
    
    # 注意力相关
    head_idx: Optional[int] = None       # 注意力头索引
    key_pos: Optional[int] = None        # Key 位置（用于注意力权重）
    query_pos: Optional[int] = None      # Query 位置
    
    # 向量维度
    feature_idx: Optional[int] = None    # 特征维度索引
    hidden_dim_start: Optional[int] = None  # 隐藏维度起始
    hidden_dim_end: Optional[int] = None    # 隐藏维度结束
    
    # 模块类型
    module_type: str = "unknown"         # 模块类型 (attention, mlp, layernorm, etc.)
    module_name: str = ""               # 具体模块名称
    
    # 元数据
    timestamp: Optional[str] = None      # 时间戳
    model_name: str = ""                # 模型名称
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_tuple(self) -> Tuple:
        """转换为元组（用于快速比较和哈希）"""
        return (
            self.batch_idx, self.sequence_idx, self.layer_idx,
            self.head_idx, self.key_pos, self.query_pos,
            self.feature_idx, self.module_type
        )
    
    def __str__(self) -> str:
        parts = [f"B{self.batch_idx}", f"S{self.sequence_idx}"]
        
        if self.layer_idx is not None:
            parts.append(f"L{self.layer_idx}")
        
        if self.head_idx is not None:
            parts.append(f"H{self.head_idx}")
        
        if self.key_pos is not None and self.query_pos is not None:
            parts.append(f"Q{self.query_pos}K{self.key_pos}")
        
        if self.feature_idx is not None:
            parts.append(f"F{self.feature_idx}")
        elif self.hidden_dim_start is not None and self.hidden_dim_end is not None:
            parts.append(f"D{self.hidden_dim_start}:{self.hidden_dim_end}")
        
        if self.module_type != "unknown":
            parts.append(f"({self.module_type})")
        
        return "_".join(parts)


@dataclass
class IntermediateVariable:
    """
    中间变量数据结构
    """
    coordinate: CoordinateSystem
    value: Union[torch.Tensor, np.ndarray, float, List]
    shape: Tuple[int, ...]
    dtype: str
    description: str = ""
    
    def to_dict(self, include_value: bool = True) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            'coordinate': self.coordinate.to_dict(),
            'shape': self.shape,
            'dtype': self.dtype,
            'description': self.description
        }
        
        if include_value:
            if isinstance(self.value, torch.Tensor):
                data['value'] = self.value.detach().cpu().numpy().tolist()
            elif isinstance(self.value, np.ndarray):
                data['value'] = self.value.tolist()
            else:
                data['value'] = self.value
        
        return data


class DataExporter:
    """
    数据导出器
    支持多种格式的数据导出
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_intermediate_variables(self,
                                    variables: List[IntermediateVariable],
                                    filename: str,
                                    formats: List[str] = ["json", "npz", "csv"],
                                    compress: bool = True) -> Dict[str, str]:
        """
        导出中间变量到多种格式
        
        Args:
            variables: 中间变量列表
            filename: 文件名（不含扩展名）
            formats: 导出格式列表
            compress: 是否压缩
        
        Returns:
            导出文件路径字典
        """
        exported_files = {}
        
        for fmt in formats:
            if fmt == "json":
                exported_files["json"] = self._export_json(variables, filename, compress)
            elif fmt == "npz":
                exported_files["npz"] = self._export_npz(variables, filename, compress)
            elif fmt == "csv":
                exported_files["csv"] = self._export_csv(variables, filename)
            elif fmt == "hdf5":
                exported_files["hdf5"] = self._export_hdf5(variables, filename)
            elif fmt == "pickle":
                exported_files["pickle"] = self._export_pickle(variables, filename)
            else:
                logger.warning(f"不支持的导出格式: {fmt}")
        
        return exported_files
    
    def _export_json(self, variables: List[IntermediateVariable], 
                    filename: str, compress: bool = True) -> str:
        """导出为 JSON 格式"""
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'num_variables': len(variables),
                'format_version': '1.0'
            },
            'variables': [var.to_dict() for var in variables]
        }
        
        if compress:
            import gzip
            output_path = self.output_dir / f"{filename}.json.gz"
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"JSON 数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_npz(self, variables: List[IntermediateVariable], 
                   filename: str, compress: bool = True) -> str:
        """导出为 NumPy .npz 格式"""
        arrays = {}
        metadata = {}
        
        for i, var in enumerate(variables):
            key = f"var_{i}_{str(var.coordinate)}"
            
            # 转换为 numpy 数组
            if isinstance(var.value, torch.Tensor):
                arrays[key] = var.value.detach().cpu().numpy()
            elif isinstance(var.value, np.ndarray):
                arrays[key] = var.value
            else:
                arrays[key] = np.array(var.value)
            
            # 存储元数据
            metadata[key] = {
                'coordinate': var.coordinate.to_dict(),
                'shape': var.shape,
                'dtype': var.dtype,
                'description': var.description
            }
        
        # 将元数据转换为字符串存储
        arrays['metadata'] = np.array(json.dumps(metadata))
        
        output_path = self.output_dir / f"{filename}.npz"
        
        if compress:
            np.savez_compressed(output_path, **arrays)
        else:
            np.savez(output_path, **arrays)
        
        logger.info(f"NPZ 数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_csv(self, variables: List[IntermediateVariable], filename: str) -> str:
        """导出为 CSV 格式（扁平化结构）"""
        rows = []
        
        for var in variables:
            # 展平张量值
            if isinstance(var.value, (torch.Tensor, np.ndarray)):
                flat_value = var.value.flatten() if isinstance(var.value, np.ndarray) else var.value.detach().cpu().numpy().flatten()
                
                for idx, val in enumerate(flat_value):
                    row = {
                        'coordinate_str': str(var.coordinate),
                        'batch_idx': var.coordinate.batch_idx,
                        'sequence_idx': var.coordinate.sequence_idx,
                        'layer_idx': var.coordinate.layer_idx,
                        'head_idx': var.coordinate.head_idx,
                        'feature_idx': idx,
                        'module_type': var.coordinate.module_type,
                        'module_name': var.coordinate.module_name,
                        'value': float(val),
                        'original_shape': str(var.shape),
                        'dtype': var.dtype,
                        'description': var.description
                    }
                    rows.append(row)
            else:
                # 标量值
                row = {
                    'coordinate_str': str(var.coordinate),
                    'batch_idx': var.coordinate.batch_idx,
                    'sequence_idx': var.coordinate.sequence_idx,
                    'layer_idx': var.coordinate.layer_idx,
                    'head_idx': var.coordinate.head_idx,
                    'feature_idx': 0,
                    'module_type': var.coordinate.module_type,
                    'module_name': var.coordinate.module_name,
                    'value': float(var.value),
                    'original_shape': str(var.shape),
                    'dtype': var.dtype,
                    'description': var.description
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path = self.output_dir / f"{filename}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"CSV 数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_hdf5(self, variables: List[IntermediateVariable], filename: str) -> str:
        """导出为 HDF5 格式"""
        output_path = self.output_dir / f"{filename}.h5"
        
        with h5py.File(output_path, 'w') as f:
            # 创建元数据组
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['export_time'] = datetime.now().isoformat()
            metadata_group.attrs['num_variables'] = len(variables)
            metadata_group.attrs['format_version'] = '1.0'
            
            # 创建数据组
            data_group = f.create_group('data')
            
            for i, var in enumerate(variables):
                var_group = data_group.create_group(f'variable_{i}')
                
                # 存储坐标信息
                coord_group = var_group.create_group('coordinate')
                for key, value in var.coordinate.to_dict().items():
                    if value is not None:
                        coord_group.attrs[key] = value
                
                # 存储数值数据
                if isinstance(var.value, torch.Tensor):
                    var_group.create_dataset('value', data=var.value.detach().cpu().numpy())
                elif isinstance(var.value, np.ndarray):
                    var_group.create_dataset('value', data=var.value)
                else:
                    var_group.create_dataset('value', data=np.array(var.value))
                
                # 存储其他属性
                var_group.attrs['shape'] = var.shape
                var_group.attrs['dtype'] = var.dtype
                var_group.attrs['description'] = var.description
        
        logger.info(f"HDF5 数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_pickle(self, variables: List[IntermediateVariable], filename: str) -> str:
        """导出为 Pickle 格式"""
        import pickle
        
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'num_variables': len(variables),
                'format_version': '1.0'
            },
            'variables': variables
        }
        
        output_path = self.output_dir / f"{filename}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Pickle 数据已导出到: {output_path}")
        return str(output_path)


class CoordinateManager:
    """
    坐标管理器
    提供坐标系统的高级操作
    """
    
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.variables = []
        
    def create_coordinate(self, **kwargs) -> CoordinateSystem:
        """创建新坐标"""
        kwargs['model_name'] = self.model_name
        return CoordinateSystem(**kwargs)
    
    def add_variable(self, 
                    coordinate: CoordinateSystem,
                    value: Union[torch.Tensor, np.ndarray, float],
                    description: str = "") -> IntermediateVariable:
        """添加中间变量"""
        # 确定形状和数据类型
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            dtype = str(value.dtype)
        elif isinstance(value, np.ndarray):
            shape = value.shape
            dtype = str(value.dtype)
        else:
            shape = ()
            dtype = type(value).__name__
        
        var = IntermediateVariable(
            coordinate=coordinate,
            value=value,
            shape=shape,
            dtype=dtype,
            description=description
        )
        
        self.variables.append(var)
        return var
    
    def filter_by_coordinate(self, **filters) -> List[IntermediateVariable]:
        """根据坐标条件过滤变量"""
        filtered = []
        
        for var in self.variables:
            match = True
            
            for key, value in filters.items():
                if hasattr(var.coordinate, key):
                    if getattr(var.coordinate, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered.append(var)
        
        return filtered
    
    def get_layer_variables(self, layer_idx: int) -> List[IntermediateVariable]:
        """获取指定层的所有变量"""
        return self.filter_by_coordinate(layer_idx=layer_idx)
    
    def get_position_variables(self, sequence_idx: int) -> List[IntermediateVariable]:
        """获取指定位置的所有变量"""
        return self.filter_by_coordinate(sequence_idx=sequence_idx)
    
    def get_attention_variables(self, layer_idx: Optional[int] = None) -> List[IntermediateVariable]:
        """获取注意力相关的变量"""
        if layer_idx is not None:
            return self.filter_by_coordinate(layer_idx=layer_idx, module_type="attention")
        else:
            return self.filter_by_coordinate(module_type="attention")
    
    def export_all(self, filename: str, **kwargs) -> Dict[str, str]:
        """导出所有变量"""
        exporter = DataExporter()
        return exporter.export_intermediate_variables(self.variables, filename, **kwargs)
    
    def clear(self):
        """清空所有变量"""
        self.variables.clear()
    
    def summary(self) -> Dict[str, Any]:
        """获取变量统计摘要"""
        if not self.variables:
            return {"total_variables": 0}
        
        layer_counts = {}
        module_type_counts = {}
        
        for var in self.variables:
            # 统计每层的变量数
            layer_idx = var.coordinate.layer_idx
            if layer_idx is not None:
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
            
            # 统计模块类型
            module_type = var.coordinate.module_type
            module_type_counts[module_type] = module_type_counts.get(module_type, 0) + 1
        
        return {
            "total_variables": len(self.variables),
            "layer_distribution": layer_counts,
            "module_type_distribution": module_type_counts,
            "layer_range": (min(layer_counts.keys()), max(layer_counts.keys())) if layer_counts else None
        }


# 辅助函数
def load_exported_data(filepath: str) -> Dict[str, Any]:
    """加载导出的数据"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.suffix == '.gz' and filepath.stem.endswith('.json'):
        import gzip
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.suffix == '.npz':
        return dict(np.load(filepath, allow_pickle=True))
    elif filepath.suffix == '.h5':
        data = {}
        with h5py.File(filepath, 'r') as f:
            # 递归读取 HDF5 数据
            def read_group(group, data_dict):
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):
                        data_dict[key] = {}
                        read_group(group[key], data_dict[key])
                    else:
                        data_dict[key] = group[key][()]
            
            read_group(f, data)
        return data
    elif filepath.suffix == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {filepath.suffix}")


if __name__ == "__main__":
    # 测试代码
    print("测试坐标系统和数据导出模块")
    
    # 创建坐标管理器
    manager = CoordinateManager("gpt2")
    
    # 创建一些测试变量
    for layer in range(3):
        for seq_pos in range(5):
            # 隐藏状态
            coord = manager.create_coordinate(
                batch_idx=0,
                sequence_idx=seq_pos,
                layer_idx=layer,
                module_type="hidden_state"
            )
            value = torch.randn(768)  # GPT-2 hidden size
            manager.add_variable(coord, value, f"Layer {layer} position {seq_pos} hidden state")
            
            # 注意力权重
            coord = manager.create_coordinate(
                batch_idx=0,
                sequence_idx=seq_pos,
                layer_idx=layer,
                head_idx=0,
                module_type="attention"
            )
            value = torch.randn(5)  # attention to all positions
            manager.add_variable(coord, value, f"Layer {layer} position {seq_pos} attention weights")
    
    # 显示摘要
    print(f"变量摘要: {manager.summary()}")
    
    # 导出数据
    exported = manager.export_all("test_export", formats=["json", "npz", "csv"])
    print(f"导出文件: {exported}")
    
    # 测试加载
    if "json" in exported:
        loaded_data = load_exported_data(exported["json"])
        print(f"加载的数据包含 {loaded_data['metadata']['num_variables']} 个变量")