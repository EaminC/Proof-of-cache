#!/usr/bin/env python3
"""
Example 3: Batch Processing
Process multiple inputs and save each result.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpt2_kvcache import GPT2KVCacheExtractor


def main():
    print("Example 3: Batch Processing")
    print("="*50)
    
    # Create extractor
    extractor = GPT2KVCacheExtractor(model_name="gpt2")
    
    # Multiple inputs
    inputs = [
        "Artificial intelligence will",
        "Machine learning algorithms",
        "Deep neural networks",
        "Computer vision systems"
    ]
    
    print(f"1. Processing {len(inputs)} inputs:")
    results = []
    
    for i, input_text in enumerate(inputs):
        print(f"\n   Input {i+1}: '{input_text}'")
        result = extractor.extract_kv_cache(input_text)
        output_file = f"../data/batch_example_{i+1}.json"
        extractor.save_to_json(result, output_file)
        
        results.append({
            "input": input_text,
            "predicted": result['predicted_next_token']['token'],
            "file": output_file
        })
        
        print(f"   - Predicted: '{result['predicted_next_token']['token']}'")
        print(f"   - Saved to: {output_file}")
    
    # Summary
    print(f"\n2. Batch processing summary:")
    for result in results:
        print(f"   '{result['input']}' → '{result['predicted']}'")
    
    print("\n✅ Example 3 completed!")


if __name__ == "__main__":
    main()
