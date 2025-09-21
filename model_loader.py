"""
模型加载和初始化模块
支持不同的预训练模型，如 GPT-2, BERT, LLaMA 等
"""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoConfig
)
from typing import Optional, Union, Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    统一的模型加载器，支持多种预训练模型
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = "config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件未找到: {config_path}，使用默认配置")
            return {
                "model_name": "gpt2",
                "device": "auto",
                "max_length": 128
            }
    
    def _get_device(self) -> torch.device:
        """自动选择设备"""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"使用 CUDA 设备: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("使用 MPS (Apple Silicon) 设备")
            else:
                device = torch.device("cpu")
                logger.info("使用 CPU 设备")
        else:
            device = torch.device(device_config)
            logger.info(f"使用指定设备: {device}")
        
        return device
    
    def load_model(self, 
                   model_name: Optional[str] = None,
                   model_type: str = "causal_lm") -> None:
        """
        加载指定的模型和分词器
        
        Args:
            model_name: 模型名称，如 'gpt2', 'bert-base-uncased' 等
            model_type: 模型类型，支持 'causal_lm', 'base', 'sequence_classification'
        """
        if model_name is None:
            model_name = self.config.get("model_name", "gpt2")
        
        logger.info(f"正在加载模型: {model_name}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 为分词器添加 pad_token（如果没有的话）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 根据模型类型加载模型
            if model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
            elif model_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
            else:  # base model
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
            
            # 移动模型到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"模型加载成功，设备: {self.device}")
            logger.info(f"模型参数数量: {self.get_model_parameters()}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_model_parameters(self) -> int:
        """获取模型参数数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        if self.model is None:
            return {}
        
        config = self.model.config
        
        info = {
            "model_type": config.model_type,
            "num_parameters": self.get_model_parameters(),
            "vocab_size": config.vocab_size,
            "device": str(self.device),
        }
        
        # 添加特定于模型的信息
        if hasattr(config, 'n_layer'):
            info['num_layers'] = config.n_layer
        elif hasattr(config, 'num_hidden_layers'):
            info['num_layers'] = config.num_hidden_layers
        
        if hasattr(config, 'n_head'):
            info['num_attention_heads'] = config.n_head
        elif hasattr(config, 'num_attention_heads'):
            info['num_attention_heads'] = config.num_attention_heads
        
        if hasattr(config, 'n_embd'):
            info['hidden_size'] = config.n_embd
        elif hasattr(config, 'hidden_size'):
            info['hidden_size'] = config.hidden_size
        
        return info
    
    def tokenize_text(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            **kwargs: 传递给 tokenizer 的额外参数
        
        Returns:
            包含 input_ids, attention_mask 等的字典
        """
        if self.tokenizer is None:
            raise RuntimeError("请先加载模型和分词器")
        
        # 设置默认参数
        default_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.config.get("max_length", 128)
        }
        default_kwargs.update(kwargs)
        
        tokens = self.tokenizer(text, **default_kwargs)
        
        # 移动到指定设备
        for key in tokens:
            if isinstance(tokens[key], torch.Tensor):
                tokens[key] = tokens[key].to(self.device)
        
        return tokens
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """解码 token ID 为文本"""
        if self.tokenizer is None:
            raise RuntimeError("请先加载分词器")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_token_info(self, token_ids: torch.Tensor) -> Dict[str, Any]:
        """获取 token 的详细信息"""
        if self.tokenizer is None:
            raise RuntimeError("请先加载分词器")
        
        # 确保是 1D tensor
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        
        token_info = {
            "token_ids": token_ids.cpu().tolist(),
            "tokens": [self.tokenizer.decode([tid]) for tid in token_ids],
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
            "length": len(token_ids)
        }
        
        return token_info


# 辅助函数
def list_available_models():
    """列出一些常用的预训练模型"""
    models = {
        "GPT-2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        "BERT": ["bert-base-uncased", "bert-large-uncased", "bert-base-cased"],
        "RoBERTa": ["roberta-base", "roberta-large"],
        "DistilBERT": ["distilbert-base-uncased", "distilbert-base-cased"],
        "T5": ["t5-small", "t5-base", "t5-large"],
        "BART": ["facebook/bart-base", "facebook/bart-large"],
    }
    
    print("可用的预训练模型:")
    for model_family, model_list in models.items():
        print(f"\n{model_family}:")
        for model in model_list:
            print(f"  - {model}")


if __name__ == "__main__":
    # 测试代码
    list_available_models()
    
    # 加载默认模型
    loader = ModelLoader()
    loader.load_model("gpt2")
    
    # 测试分词
    text = "Hello, how are you today?"
    tokens = loader.tokenize_text(text)
    print(f"\n输入文本: {text}")
    print(f"Token 信息: {loader.get_token_info(tokens['input_ids'])}")
    print(f"模型信息: {loader.get_model_info()}")