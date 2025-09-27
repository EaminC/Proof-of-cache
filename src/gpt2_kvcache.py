"""
GPT-2 KV Cache Extractor
Extract Key-Value cache from GPT-2 model for each layer, head, and token position.
"""

import json
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class GPT2KVCacheExtractor:
    """GPT-2 KV Cache Extractor with next token prediction"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize GPT-2 model and tokenizer
        
        Args:
            model_name: Model name, default "gpt2"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use GPT2LMHeadModel for language modeling head
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.config = self.model.config
        self.n_layers = self.config.n_layer  # 12 layers
        self.n_heads = self.config.n_head    # 12 attention heads
        self.d_head = self.config.n_embd // self.config.n_head  # dimension per head
        
        print(f"Model config: {self.n_layers} layers, {self.n_heads} heads, {self.d_head} dims per head")
        
        # Store KV cache data
        self.kv_cache_data = {}
        self.attention_weights = {}
        
    def _hook_kv_cache(self, layer_idx: int):
        """
        Create hook function to capture KV cache and attention weights
        
        Args:
            layer_idx: Layer index
        """
        def hook(module, input, output):
            # For GPT2 attention module, output contains (attn_output, present_key_value, attn_weights)
            if hasattr(output, '__len__') and len(output) > 1:
                # Extract KV cache
                if output[1] is not None:
                    key, value = output[1]
                    
                    # key and value shape: [batch_size, n_heads, seq_len, d_head]
                    batch_size, n_heads, seq_len, d_head = key.shape
                    
                    # Store KV cache for each head
                    layer_cache = {}
                    for head_idx in range(n_heads):
                        # Store KV for each token
                        tokens_kv = []
                        for token_idx in range(seq_len):
                            token_kv = {
                                "token_idx": token_idx,
                                "key": key[0, head_idx, token_idx].cpu().numpy().tolist(),  # [d_head]
                                "value": value[0, head_idx, token_idx].cpu().numpy().tolist()  # [d_head]
                            }
                            tokens_kv.append(token_kv)
                        
                        layer_cache[f"head_{head_idx}"] = {
                            "tokens": tokens_kv,
                            "shape": {
                                "seq_len": seq_len,
                                "d_head": d_head
                            }
                        }
                    
                    self.kv_cache_data[f"layer_{layer_idx}"] = layer_cache
                
                # Extract attention weights (if available)
                if len(output) > 2 and output[2] is not None:
                    attn_weights = output[2]  # [batch_size, n_heads, seq_len, seq_len]
                    
                    # Store attention weights
                    layer_attention = {}
                    for head_idx in range(self.n_heads):
                        layer_attention[f"head_{head_idx}"] = attn_weights[0, head_idx].cpu().numpy().tolist()
                    
                    self.attention_weights[f"layer_{layer_idx}"] = layer_attention
                
        return hook
    
    def predict_next_token(self, input_ids: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> Tuple[int, str, List[Tuple[str, float]]]:
        """
        Predict next token
        
        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature
            top_k: Top-K sampling K value
            
        Returns:
            (predicted token ID, predicted token text, top-k candidate tokens with probabilities)
        """
        with torch.no_grad():
            # Get model output
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]  # Last position logits
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-K sampling
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, len(probs)))
            
            # Get most likely token
            next_token_id = top_k_indices[0].item()
            next_token = self.tokenizer.decode([next_token_id])
            
            # Get top-k candidate tokens
            top_k_candidates = []
            for i in range(len(top_k_indices)):
                token_id = top_k_indices[i].item()
                token_text = self.tokenizer.decode([token_id])
                token_prob = top_k_probs[i].item()
                top_k_candidates.append((token_text, token_prob))
            
            return next_token_id, next_token, top_k_candidates
    
    def extract_kv_cache(self, input_text: str, temperature: float = 1.0, top_k: int = 10) -> Dict[str, Any]:
        """
        Extract KV cache for input text and generate next token
        
        Args:
            input_text: Input text
            temperature: Generation temperature
            top_k: Number of top-k candidate tokens to return
            
        Returns:
            Dictionary containing KV cache and generation information
        """
        # Clear previous cache
        self.kv_cache_data = {}
        self.attention_weights = {}
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get input tokens
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Register hooks
        hooks = []
        for layer_idx in range(self.n_layers):
            layer = self.model.transformer.h[layer_idx]  # GPT2 transformer layer
            hook = layer.attn.register_forward_hook(self._hook_kv_cache(layer_idx))
            hooks.append(hook)
        
        # Predict next token
        next_token_id, next_token, top_k_candidates = self.predict_next_token(
            input_ids, temperature=temperature, top_k=top_k
        )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Build output JSON structure
        result = {
            "model": "gpt2",
            "input_text": input_text,
            "input_tokens": input_tokens,
            "input_token_ids": input_ids[0].cpu().numpy().tolist(),
            "predicted_next_token": {
                "token": next_token,
                "token_id": next_token_id,
                "top_k_candidates": [
                    {"token": token, "probability": prob} 
                    for token, prob in top_k_candidates
                ]
            },
            "kv_cache": self.kv_cache_data,
            "attention_weights": self.attention_weights if self.attention_weights else "Not captured",
            "metadata": {
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_head": self.d_head,
                "sequence_length": len(input_tokens),
                "temperature": temperature,
                "top_k": top_k
            }
        }
        
        return result
    
    def get_coordinate_kv(self, data: Dict[str, Any], layer: int, head: int, token: int) -> Optional[Dict[str, Any]]:
        """
        Extract KV cache for specific coordinate
        
        Args:
            data: Complete KV cache data
            layer: Layer index (0-11)
            head: Attention head index (0-11)
            token: Token index
            
        Returns:
            KV cache for specific coordinate, or None if not found
        """
        try:
            layer_key = f"layer_{layer}"
            head_key = f"head_{head}"
            
            if layer_key in data['kv_cache'] and head_key in data['kv_cache'][layer_key]:
                tokens = data['kv_cache'][layer_key][head_key]['tokens']
                if token < len(tokens):
                    return {
                        "coordinate": f"[Layer {layer}, Head {head}, Token {token}]",
                        "token_text": data['input_tokens'][token] if token < len(data['input_tokens']) else "N/A",
                        "kv_data": tokens[token]
                    }
        except Exception as e:
            print(f"Error extracting coordinate [{layer}, {head}, {token}]: {e}")
        
        return None
    
    def save_to_json(self, data: Dict[str, Any], output_path: str = "kv_cache_output.json"):
        """
        Save KV cache data to JSON file
        
        Args:
            data: KV cache data
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"KV cache saved to: {output_path}")
    
    def analyze_kv_cache(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze KV cache statistics
        
        Args:
            data: KV cache data
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "input_length": len(data["input_tokens"]),
            "layers": data["metadata"]["n_layers"],
            "heads_per_layer": data["metadata"]["n_heads"],
            "head_dimension": data["metadata"]["d_head"],
            "total_parameters": 0
        }
        
        # Calculate total parameters
        for layer_name, layer_data in data["kv_cache"].items():
            for head_name, head_data in layer_data.items():
                # Each head has key and value, each is [seq_len, d_head]
                seq_len = len(head_data["tokens"])
                d_head = len(head_data["tokens"][0]["key"]) if seq_len > 0 else 0
                stats["total_parameters"] += 2 * seq_len * d_head  # key + value
        
        return stats


def main():
    """Main function"""
    # Create extractor
    print("Initializing GPT-2 KV Cache Extractor...")
    extractor = GPT2KVCacheExtractor(model_name="gpt2")
    
    # Test input
    test_input = "The quick brown fox jumps over the lazy dog"
    print(f"\nInput: '{test_input}'")
    
    # Extract KV cache
    print("Extracting KV cache and generating next token...")
    kv_cache_data = extractor.extract_kv_cache(
        test_input, 
        temperature=0.8,
        top_k=10
    )
    
    # Save to JSON
    output_file = "kv_cache_output.json"
    extractor.save_to_json(kv_cache_data, output_file)
    
    # Print summary
    print(f"\nPredicted next token: '{kv_cache_data['predicted_next_token']['token']}'")
    print("\nTop-5 candidate tokens:")
    for i, candidate in enumerate(kv_cache_data['predicted_next_token']['top_k_candidates'][:5]):
        print(f"  {i+1}. '{candidate['token']}' (probability: {candidate['probability']:.4f})")
    
    # Show specific coordinate example
    print("\nSpecific coordinate KV Cache example:")
    specific_kv = extractor.get_coordinate_kv(kv_cache_data, 0, 0, 0)
    if specific_kv:
        print(f"\n{specific_kv['coordinate']}:")
        print(f"  Token: '{specific_kv['token_text']}'")
        print(f"  Key first 5 dims: {specific_kv['kv_data']['key'][:5]}")
        print(f"  Value first 5 dims: {specific_kv['kv_data']['value'][:5]}")
    
    # Analyze statistics
    stats = extractor.analyze_kv_cache(kv_cache_data)
    print(f"\nKV Cache Statistics:")
    print(f"  - Input length: {stats['input_length']} tokens")
    print(f"  - Layers: {stats['layers']}")
    print(f"  - Heads per layer: {stats['heads_per_layer']}")
    print(f"  - Head dimension: {stats['head_dimension']}")
    print(f"  - Total KV cache parameters: {stats['total_parameters']:,}")


if __name__ == "__main__":
    main()
