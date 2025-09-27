#!/usr/bin/env python3
"""
Example 1: Basic KV Cache Extraction
Simple example showing how to extract KV cache from a single input.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpt2_kvcache import GPT2KVCacheExtractor


def main():
    print("Example 1: Basic KV Cache Extraction")
    print("="*50)
    
    # Create extractor
    print("1. Creating GPT-2 KV Cache Extractor...")
    extractor = GPT2KVCacheExtractor(model_name="gpt2")
    
    # Input text
    input_text = "The quick brown fox jumps over the lazy dog"
    print(f"2. Input text: '{input_text}'")
    
    # Extract KV cache
    print("3. Extracting KV cache...")
    result = extractor.extract_kv_cache(input_text)
    
    # Save result
    output_file = "../data/example_1_output.json"
    extractor.save_to_json(result, output_file)
    print(f"4. Saved to: {output_file}")
    
    # Print results
    print("5. Results:")
    print(f"   - Input tokens: {result['input_tokens']}")
    print(f"   - Predicted next token: '{result['predicted_next_token']['token']}'")
    print(f"   - Top-3 candidates:")
    for i, candidate in enumerate(result['predicted_next_token']['top_k_candidates'][:3]):
        print(f"     {i+1}. '{candidate['token']}' (probability: {candidate['probability']:.4f})")
    
    print("\n✅ Example 1 completed!")


if __name__ == "__main__":
    main()
