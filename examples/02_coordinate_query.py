#!/usr/bin/env python3
"""
Example 2: Coordinate Query
Show how to query specific coordinates [layer, head, token].
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpt2_kvcache import GPT2KVCacheExtractor


def main():
    print("Example 2: Coordinate Query")
    print("="*50)
    
    # Create extractor
    extractor = GPT2KVCacheExtractor(model_name="gpt2")
    
    # Extract KV cache
    input_text = "Machine learning is fascinating"
    print(f"1. Input: '{input_text}'")
    result = extractor.extract_kv_cache(input_text)
    
    # Query specific coordinates
    coordinates = [
        (0, 0, 0),   # First layer, first head, first token
        (5, 5, 1),   # Middle layer, middle head, second token
        (11, 11, 2), # Last layer, last head, third token
    ]
    
    print("2. Querying specific coordinates:")
    for layer, head, token in coordinates:
        kv_data = extractor.get_coordinate_kv(result, layer, head, token)
        if kv_data:
            print(f"\n   Coordinate [Layer {layer}, Head {head}, Token {token}]:")
            print(f"   - Token: '{kv_data['token_text']}'")
            print(f"   - Key norm: {kv_data['key_norm']:.6f}")
            print(f"   - Value norm: {kv_data['value_norm']:.6f}")
            print(f"   - Key (first 5): {kv_data['key'][:5]}")
            print(f"   - Value (first 5): {kv_data['value'][:5]}")
        else:
            print(f"   Coordinate [Layer {layer}, Head {head}, Token {token}]: Not found")
    
    print("\n✅ Example 2 completed!")


if __name__ == "__main__":
    main()
