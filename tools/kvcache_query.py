#!/usr/bin/env python3
"""
KV Cache Query Tool
Command-line tool for querying and analyzing KV cache data from JSON files.
"""

import json
import argparse
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
import os


class KVCacheQueryTool:
    """KV Cache Query Tool"""
    
    def __init__(self, json_file: str):
        """
        Initialize with JSON file
        
        Args:
            json_file: Path to KV cache JSON file
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.n_layers = self.data['metadata']['n_layers']
        self.n_heads = self.data['metadata']['n_heads']
        self.seq_len = self.data['metadata']['sequence_length']
        self.d_head = self.data['metadata']['d_head']
    
    def get_coordinate_kv(self, layer: int, head: int, token: int) -> Optional[Dict[str, Any]]:
        """
        Get KV cache for specific coordinate
        
        Args:
            layer: Layer index (0-11)
            head: Head index (0-11)
            token: Token index
            
        Returns:
            KV cache data for coordinate or None
        """
        try:
            layer_key = f"layer_{layer}"
            head_key = f"head_{head}"
            
            if layer_key in self.data['kv_cache'] and head_key in self.data['kv_cache'][layer_key]:
                tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                if token < len(tokens):
                    token_data = tokens[token]
                    return {
                        "coordinate": f"[Layer {layer}, Head {head}, Token {token}]",
                        "token_text": self.data['input_tokens'][token] if token < len(self.data['input_tokens']) else "N/A",
                        "key": token_data['key'],
                        "value": token_data['value'],
                        "key_norm": float(np.linalg.norm(np.array(token_data['key']))),
                        "value_norm": float(np.linalg.norm(np.array(token_data['value'])))
                    }
        except Exception as e:
            print(f"Error getting coordinate [{layer}, {head}, {token}]: {e}")
        
        return None
    
    def find_coordinates_by_token(self, token_text: str) -> List[Dict[str, Any]]:
        """
        Find all coordinates for a specific token text
        
        Args:
            token_text: Token text to search for
            
        Returns:
            List of coordinates containing this token
        """
        coordinates = []
        
        for token_idx, token in enumerate(self.data['input_tokens']):
            if token_text in token or token in token_text:
                for layer in range(self.n_layers):
                    for head in range(self.n_heads):
                        kv_data = self.get_coordinate_kv(layer, head, token_idx)
                        if kv_data:
                            coordinates.append({
                                "layer": layer,
                                "head": head,
                                "token": token_idx,
                                "token_text": token,
                                "key_norm": kv_data['key_norm'],
                                "value_norm": kv_data['value_norm']
                            })
        
        return coordinates
    
    def get_layer_summary(self, layer: int) -> Dict[str, Any]:
        """
        Get summary statistics for a specific layer
        
        Args:
            layer: Layer index
            
        Returns:
            Layer summary statistics
        """
        layer_key = f"layer_{layer}"
        if layer_key not in self.data['kv_cache']:
            return None
        
        key_norms = []
        value_norms = []
        
        for head in range(self.n_heads):
            head_key = f"head_{head}"
            if head_key in self.data['kv_cache'][layer_key]:
                tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
                for token in tokens:
                    key_norms.append(np.linalg.norm(np.array(token['key'])))
                    value_norms.append(np.linalg.norm(np.array(token['value'])))
        
        return {
            "layer": layer,
            "avg_key_norm": float(np.mean(key_norms)),
            "avg_value_norm": float(np.mean(value_norms)),
            "std_key_norm": float(np.std(key_norms)),
            "std_value_norm": float(np.std(value_norms)),
            "max_key_norm": float(np.max(key_norms)),
            "max_value_norm": float(np.max(value_norms))
        }
    
    def get_head_summary(self, layer: int, head: int) -> Dict[str, Any]:
        """
        Get summary statistics for a specific head
        
        Args:
            layer: Layer index
            head: Head index
            
        Returns:
            Head summary statistics
        """
        layer_key = f"layer_{layer}"
        head_key = f"head_{head}"
        
        if layer_key not in self.data['kv_cache'] or head_key not in self.data['kv_cache'][layer_key]:
            return None
        
        tokens = self.data['kv_cache'][layer_key][head_key]['tokens']
        key_norms = []
        value_norms = []
        
        for token in tokens:
            key_norms.append(np.linalg.norm(np.array(token['key'])))
            value_norms.append(np.linalg.norm(np.array(token['value'])))
        
        return {
            "layer": layer,
            "head": head,
            "avg_key_norm": float(np.mean(key_norms)),
            "avg_value_norm": float(np.mean(value_norms)),
            "std_key_norm": float(np.std(key_norms)),
            "std_value_norm": float(np.std(value_norms)),
            "token_count": len(tokens)
        }
    
    def get_token_summary(self, token_idx: int) -> Dict[str, Any]:
        """
        Get summary statistics for a specific token across all layers and heads
        
        Args:
            token_idx: Token index
            
        Returns:
            Token summary statistics
        """
        if token_idx >= self.seq_len:
            return None
        
        key_norms = []
        value_norms = []
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                kv_data = self.get_coordinate_kv(layer, head, token_idx)
                if kv_data:
                    key_norms.append(kv_data['key_norm'])
                    value_norms.append(kv_data['value_norm'])
        
        return {
            "token_idx": token_idx,
            "token_text": self.data['input_tokens'][token_idx],
            "avg_key_norm": float(np.mean(key_norms)),
            "avg_value_norm": float(np.mean(value_norms)),
            "std_key_norm": float(np.std(key_norms)),
            "std_value_norm": float(np.std(value_norms)),
            "total_coordinates": len(key_norms)
        }
    
    def search_by_norm_range(self, min_norm: float, max_norm: float, norm_type: str = "key") -> List[Dict[str, Any]]:
        """
        Search coordinates by norm range
        
        Args:
            min_norm: Minimum norm value
            max_norm: Maximum norm value
            norm_type: "key" or "value"
            
        Returns:
            List of coordinates within norm range
        """
        results = []
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                for token in range(self.seq_len):
                    kv_data = self.get_coordinate_kv(layer, head, token)
                    if kv_data:
                        norm = kv_data[f'{norm_type}_norm']
                        if min_norm <= norm <= max_norm:
                            results.append({
                                "layer": layer,
                                "head": head,
                                "token": token,
                                "token_text": kv_data['token_text'],
                                "key_norm": kv_data['key_norm'],
                                "value_norm": kv_data['value_norm']
                            })
        
        return results
    
    def export_coordinates_csv(self, output_file: str, coordinates: List[Dict[str, Any]]):
        """
        Export coordinates to CSV file
        
        Args:
            output_file: Output CSV file path
            coordinates: List of coordinate data
        """
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if coordinates:
                fieldnames = coordinates[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(coordinates)
        
        print(f"Coordinates exported to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='KV Cache Query Tool')
    parser.add_argument('json_file', help='KV cache JSON file path')
    
    # Query options
    parser.add_argument('--coordinate', nargs=3, type=int, metavar=('LAYER', 'HEAD', 'TOKEN'),
                       help='Get KV cache for specific coordinate')
    parser.add_argument('--token-search', type=str, metavar='TOKEN_TEXT',
                       help='Find coordinates containing specific token text')
    parser.add_argument('--layer-summary', type=int, metavar='LAYER',
                       help='Get summary for specific layer')
    parser.add_argument('--head-summary', nargs=2, type=int, metavar=('LAYER', 'HEAD'),
                       help='Get summary for specific head')
    parser.add_argument('--token-summary', type=int, metavar='TOKEN_IDX',
                       help='Get summary for specific token')
    
    # Search options
    parser.add_argument('--search-norm', nargs=3, metavar=('MIN', 'MAX', 'TYPE'),
                       help='Search by norm range (MIN MAX TYPE where TYPE is key or value)')
    
    # Export options
    parser.add_argument('--export-csv', type=str, metavar='OUTPUT_FILE',
                       help='Export search results to CSV file')
    
    # Info options
    parser.add_argument('--info', action='store_true',
                       help='Show basic info about the dataset')
    parser.add_argument('--list-tokens', action='store_true',
                       help='List all tokens in the dataset')
    
    args = parser.parse_args()
    
    try:
        # Initialize query tool
        tool = KVCacheQueryTool(args.json_file)
        
        # Show basic info
        if args.info:
            print("="*60)
            print("Dataset Information")
            print("="*60)
            print(f"Model: {tool.data['model']}")
            print(f"Input text: '{tool.data['input_text']}'")
            print(f"Input tokens: {tool.data['input_tokens']}")
            print(f"Predicted next token: '{tool.data['predicted_next_token']['token']}'")
            print(f"Layers: {tool.n_layers}")
            print(f"Heads per layer: {tool.n_heads}")
            print(f"Head dimension: {tool.d_head}")
            print(f"Sequence length: {tool.seq_len}")
            print(f"Total coordinates: {tool.n_layers * tool.n_heads * tool.seq_len}")
            return
        
        # List tokens
        if args.list_tokens:
            print("="*60)
            print("Tokens in Dataset")
            print("="*60)
            for i, token in enumerate(tool.data['input_tokens']):
                print(f"Token {i}: '{token}'")
            return
        
        # Query specific coordinate
        if args.coordinate:
            layer, head, token = args.coordinate
            kv_data = tool.get_coordinate_kv(layer, head, token)
            if kv_data:
                print("="*60)
                print(f"Coordinate {kv_data['coordinate']}")
                print("="*60)
                print(f"Token text: '{kv_data['token_text']}'")
                print(f"Key norm: {kv_data['key_norm']:.6f}")
                print(f"Value norm: {kv_data['value_norm']:.6f}")
                print(f"Key vector (first 10): {kv_data['key'][:10]}")
                print(f"Value vector (first 10): {kv_data['value'][:10]}")
            else:
                print(f"Coordinate [{layer}, {head}, {token}] not found")
            return
        
        # Search by token text
        if args.token_search:
            coordinates = tool.find_coordinates_by_token(args.token_search)
            print("="*60)
            print(f"Coordinates containing '{args.token_search}'")
            print("="*60)
            for coord in coordinates[:20]:  # Show first 20
                print(f"Layer {coord['layer']}, Head {coord['head']}, Token {coord['token']}: "
                      f"'{coord['token_text']}' (key_norm: {coord['key_norm']:.4f}, "
                      f"value_norm: {coord['value_norm']:.4f})")
            if len(coordinates) > 20:
                print(f"... and {len(coordinates) - 20} more")
            
            if args.export_csv:
                tool.export_coordinates_csv(args.export_csv, coordinates)
            return
        
        # Layer summary
        if args.layer_summary is not None:
            summary = tool.get_layer_summary(args.layer_summary)
            if summary:
                print("="*60)
                print(f"Layer {args.layer_summary} Summary")
                print("="*60)
                print(f"Average key norm: {summary['avg_key_norm']:.6f}")
                print(f"Average value norm: {summary['avg_value_norm']:.6f}")
                print(f"Key norm std: {summary['std_key_norm']:.6f}")
                print(f"Value norm std: {summary['std_value_norm']:.6f}")
                print(f"Max key norm: {summary['max_key_norm']:.6f}")
                print(f"Max value norm: {summary['max_value_norm']:.6f}")
            else:
                print(f"Layer {args.layer_summary} not found")
            return
        
        # Head summary
        if args.head_summary:
            layer, head = args.head_summary
            summary = tool.get_head_summary(layer, head)
            if summary:
                print("="*60)
                print(f"Head [{layer}, {head}] Summary")
                print("="*60)
                print(f"Average key norm: {summary['avg_key_norm']:.6f}")
                print(f"Average value norm: {summary['avg_value_norm']:.6f}")
                print(f"Key norm std: {summary['std_key_norm']:.6f}")
                print(f"Value norm std: {summary['std_value_norm']:.6f}")
                print(f"Token count: {summary['token_count']}")
            else:
                print(f"Head [{layer}, {head}] not found")
            return
        
        # Token summary
        if args.token_summary is not None:
            summary = tool.get_token_summary(args.token_summary)
            if summary:
                print("="*60)
                print(f"Token {args.token_summary} Summary")
                print("="*60)
                print(f"Token text: '{summary['token_text']}'")
                print(f"Average key norm: {summary['avg_key_norm']:.6f}")
                print(f"Average value norm: {summary['avg_value_norm']:.6f}")
                print(f"Key norm std: {summary['std_key_norm']:.6f}")
                print(f"Value norm std: {summary['std_value_norm']:.6f}")
                print(f"Total coordinates: {summary['total_coordinates']}")
            else:
                print(f"Token {args.token_summary} not found")
            return
        
        # Search by norm range
        if args.search_norm:
            min_norm, max_norm, norm_type = args.search_norm
            min_norm, max_norm = float(min_norm), float(max_norm)
            results = tool.search_by_norm_range(min_norm, max_norm, norm_type)
            print("="*60)
            print(f"Search Results: {norm_type} norm between {min_norm} and {max_norm}")
            print("="*60)
            for result in results[:20]:  # Show first 20
                print(f"Layer {result['layer']}, Head {result['head']}, Token {result['token']}: "
                      f"'{result['token_text']}' ({norm_type}_norm: {result[f'{norm_type}_norm']:.6f})")
            if len(results) > 20:
                print(f"... and {len(results) - 20} more")
            
            if args.export_csv:
                tool.export_coordinates_csv(args.export_csv, results)
            return
        
        # If no specific query, show help
        print("No query specified. Use --help for available options.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
