#!/usr/bin/env python3
"""
KV Cache Visualization Tool
Visualize and analyze GPT-2 KV cache data
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional
import argparse


class KVCacheVisualizer:
    """KV Cache Visualizer"""
    
    def __init__(self, json_path: str):
        """
        Initialize visualizer
        
        Args:
            json_path: KV cache JSON file path
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.n_layers = self.data['metadata']['n_layers']
        self.n_heads = self.data['metadata']['n_heads']
        self.seq_len = self.data['metadata']['sequence_length']
        self.d_head = self.data['metadata']['d_head']
    
    def print_summary(self):
        """Print KV cache summary"""
        print("="*70)
        print("KV Cache Data Summary")
        print("="*70)
        print(f"Model: {self.data['model']}")
        print(f"Input text: '{self.data['input_text']}'")
        print(f"Input tokens: {self.data['input_tokens']}")
        print(f"Predicted next token: '{self.data['predicted_next_token']['token']}'")
        print()
        print("Model configuration:")
        print(f"  - Layers: {self.n_layers}")
        print(f"  - Attention heads: {self.n_heads}")
        print(f"  - Head dimension: {self.d_head}")
        print(f"  - Sequence length: {self.seq_len}")
        print()
        print("Top candidate tokens:")
        for i, candidate in enumerate(self.data['predicted_next_token']['top_k_candidates'][:5]):
            print(f"  {i+1}. '{candidate['token']}' (probability: {candidate['probability']:.4f})")
    
    def visualize_layer_head_matrix(self):
        """Visualize layer-head matrix KV cache statistics"""
        print("\n" + "="*70)
        print("Layer-Head Matrix KV Cache Statistics")
        print("="*70)
        
        # Create matrix to store statistics for each (layer, head)
        print("\nEach position shows average L2 norm of Key and Value:")
        print("\n     ", end="")
        for head in range(min(self.n_heads, 12)):
            print(f"Head{head:2d}", end="  ")
        print()
        
        for layer in range(self.n_layers):
            print(f"L{layer:2d}: ", end="")
            for head in range(min(self.n_heads, 12)):
                # Calculate KV norm for this layer-head
                layer_key = f"layer_{layer}"
                head_key = f"head_{head}"
                
                if layer_key in self.data['kv_cache'] and head_key in self.data['kv_cache'][layer_key]:
                    tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                    
                    # Calculate average L2 norm for all tokens
                    key_norms = []
                    value_norms = []
                    for token in tokens:
                        key_array = np.array(token['key'])
                        value_array = np.array(token['value'])
                        key_norms.append(np.linalg.norm(key_array))
                        value_norms.append(np.linalg.norm(value_array))
                    
                    avg_norm = (np.mean(key_norms) + np.mean(value_norms)) / 2
                    print(f"{avg_norm:6.2f}", end="  ")
                else:
                    print("   N/A", end="  ")
            print()
    
    def analyze_token_kvcache(self, token_idx: int):
        """Analyze KV cache for specific token"""
        if token_idx >= self.seq_len:
            print(f"Error: Token index {token_idx} out of range (max: {self.seq_len-1})")
            return
        
        token_text = self.data['input_tokens'][token_idx]
        print(f"\n" + "="*70)
        print(f"Token {token_idx}: '{token_text}' KV Cache Analysis")
        print("="*70)
        
        # Collect KV cache for this token across all layers and heads
        key_stats = []
        value_stats = []
        
        for layer in range(self.n_layers):
            layer_key = f"layer_{layer}"
            for head in range(self.n_heads):
                head_key = f"head_{head}"
                
                if layer_key in self.data['kv_cache'] and head_key in self.data['kv_cache'][layer_key]:
                    tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                    if token_idx < len(tokens):
                        token_data = tokens[token_idx]
                        key_array = np.array(token_data['key'])
                        value_array = np.array(token_data['value'])
                        
                        key_stats.append({
                            'layer': layer,
                            'head': head,
                            'mean': float(np.mean(key_array)),
                            'std': float(np.std(key_array)),
                            'min': float(np.min(key_array)),
                            'max': float(np.max(key_array)),
                            'norm': float(np.linalg.norm(key_array))
                        })
                        
                        value_stats.append({
                            'layer': layer,
                            'head': head,
                            'mean': float(np.mean(value_array)),
                            'std': float(np.std(value_array)),
                            'min': float(np.min(value_array)),
                            'max': float(np.max(value_array)),
                            'norm': float(np.linalg.norm(value_array))
                        })
        
        # Print statistics
        print("\nKey statistics (first 5 layer-head combinations):")
        print("Layer Head    Mean      Std      Min      Max     L2Norm")
        for stat in key_stats[:5]:
            print(f"  {stat['layer']:2d}   {stat['head']:2d}  {stat['mean']:8.4f} {stat['std']:8.4f} "
                  f"{stat['min']:8.4f} {stat['max']:8.4f} {stat['norm']:8.4f}")
        
        print("\nValue statistics (first 5 layer-head combinations):")
        print("Layer Head    Mean      Std      Min      Max     L2Norm")
        for stat in value_stats[:5]:
            print(f"  {stat['layer']:2d}   {stat['head']:2d}  {stat['mean']:8.4f} {stat['std']:8.4f} "
                  f"{stat['min']:8.4f} {stat['max']:8.4f} {stat['norm']:8.4f}")
        
        # Calculate overall statistics
        all_key_norms = [s['norm'] for s in key_stats]
        all_value_norms = [s['norm'] for s in value_stats]
        
        print(f"\nOverall statistics:")
        print(f"  Key L2 norm - mean: {np.mean(all_key_norms):.4f}, std: {np.std(all_key_norms):.4f}")
        print(f"  Value L2 norm - mean: {np.mean(all_value_norms):.4f}, std: {np.std(all_value_norms):.4f}")
    
    def compare_tokens(self):
        """Compare KV cache features across all tokens"""
        print("\n" + "="*70)
        print("Token-wise KV Cache Comparison")
        print("="*70)
        
        token_features = []
        
        for token_idx in range(self.seq_len):
            token_text = self.data['input_tokens'][token_idx]
            
            # Calculate average features for this token
            key_norms = []
            value_norms = []
            
            for layer in range(self.n_layers):
                layer_key = f"layer_{layer}"
                for head in range(self.n_heads):
                    head_key = f"head_{head}"
                    
                    if layer_key in self.data['kv_cache'] and head_key in self.data['kv_cache'][layer_key]:
                        tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                        if token_idx < len(tokens):
                            token_data = tokens[token_idx]
                            key_array = np.array(token_data['key'])
                            value_array = np.array(token_data['value'])
                            key_norms.append(np.linalg.norm(key_array))
                            value_norms.append(np.linalg.norm(value_array))
            
            avg_key_norm = np.mean(key_norms) if key_norms else 0
            avg_value_norm = np.mean(value_norms) if value_norms else 0
            
            token_features.append({
                'idx': token_idx,
                'text': token_text,
                'avg_key_norm': avg_key_norm,
                'avg_value_norm': avg_value_norm,
                'total_norm': avg_key_norm + avg_value_norm
            })
        
        # Print comparison results
        print("\nToken  Text              Avg Key Norm  Avg Value Norm  Total Norm")
        print("-" * 70)
        for feat in token_features:
            # Truncate or pad text to fixed width
            text_display = feat['text'][:15].ljust(15)
            print(f"{feat['idx']:3d}    {text_display}  {feat['avg_key_norm']:12.4f}  "
                  f"{feat['avg_value_norm']:14.4f}  {feat['total_norm']:11.4f}")
        
        # Find special tokens
        max_norm_token = max(token_features, key=lambda x: x['total_norm'])
        min_norm_token = min(token_features, key=lambda x: x['total_norm'])
        
        print(f"\nHighest norm token: '{max_norm_token['text']}' (index: {max_norm_token['idx']}, "
              f"total norm: {max_norm_token['total_norm']:.4f})")
        print(f"Lowest norm token: '{min_norm_token['text']}' (index: {min_norm_token['idx']}, "
              f"total norm: {min_norm_token['total_norm']:.4f})")
    
    def export_layer_head_csv(self, output_path: str = "kvcache_matrix.csv"):
        """Export layer-head matrix to CSV file"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Layer/Head'] + [f'Head_{h}' for h in range(self.n_heads)]
            writer.writerow(header)
            
            # Write data
            for layer in range(self.n_layers):
                row = [f'Layer_{layer}']
                for head in range(self.n_heads):
                    layer_key = f"layer_{layer}"
                    head_key = f"head_{head}"
                    
                    if layer_key in self.data['kv_cache'] and head_key in self.data['kv_cache'][layer_key]:
                        tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                        
                        # Calculate average norm
                        norms = []
                        for token in tokens:
                            key_array = np.array(token['key'])
                            value_array = np.array(token['value'])
                            norms.append((np.linalg.norm(key_array) + np.linalg.norm(value_array)) / 2)
                        
                        avg_norm = np.mean(norms)
                        row.append(f'{avg_norm:.4f}')
                    else:
                        row.append('N/A')
                
                writer.writerow(row)
        
        print(f"\nLayer-head matrix exported to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='KV Cache Visualization Tool')
    parser.add_argument('json_file', nargs='?', default='kv_cache_output.json',
                       help='KV cache JSON file path (default: kv_cache_output.json)')
    parser.add_argument('--token', type=int, help='Analyze specific token KV cache')
    parser.add_argument('--export-csv', action='store_true', 
                       help='Export layer-head matrix to CSV file')
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = KVCacheVisualizer(args.json_file)
        
        # Print summary
        visualizer.print_summary()
        
        # Visualize layer-head matrix
        visualizer.visualize_layer_head_matrix()
        
        # Compare tokens
        visualizer.compare_tokens()
        
        # Analyze specific token
        if args.token is not None:
            visualizer.analyze_token_kvcache(args.token)
        
        # Export CSV
        if args.export_csv:
            visualizer.export_layer_head_csv()
        
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found")
        print("Please run gpt2_kvcache.py or examples.py first to generate KV cache data")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
