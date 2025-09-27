#!/usr/bin/env python3
"""
Example 6: Analysis Workflow
Complete workflow from generation to analysis.
"""

import sys
import os
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpt2_kvcache import GPT2KVCacheExtractor


def main():
    print("Example 6: Analysis Workflow")
    print("="*50)
    
    # Step 1: Generate dataset
    print("1. Generating analysis dataset...")
    extractor = GPT2KVCacheExtractor(model_name="gpt2")
    
    analysis_inputs = [
        "The quick brown fox",
        "Machine learning is",
        "Deep learning models"
    ]
    
    analysis_files = []
    for i, input_text in enumerate(analysis_inputs):
        print(f"   Processing: '{input_text}'")
        result = extractor.extract_kv_cache(input_text)
        output_file = f"../data/analysis_{i+1}.json"
        extractor.save_to_json(result, output_file)
        analysis_files.append(output_file)
        print(f"   Generated: {output_file}")
    
    # Step 2: Analyze each file
    print("\n2. Analyzing generated files:")
    for i, file_path in enumerate(analysis_files):
        print(f"\n   Analysis {i+1}: {file_path}")
        
        # Basic info
        info_cmd = f"python3 ../tools/kvcache_query.py {file_path} --info"
        try:
            result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[2:8]:  # Show key info lines
                    print(f"     {line}")
        except Exception as e:
            print(f"     Error: {e}")
        
        # Coordinate analysis
        coord_cmd = f"python3 ../tools/kvcache_query.py {file_path} --coordinate 0 0 0"
        try:
            result = subprocess.run(coord_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"     Coordinate [0,0,0]: {lines[-1] if lines else 'No output'}")
        except Exception as e:
            print(f"     Error: {e}")
    
    # Step 3: Summary
    print("\n3. Analysis summary:")
    print("   - All files generated successfully")
    print("   - Each file contains complete KV cache data")
    print("   - Ready for detailed analysis using query tool")
    
    print("\n✅ Example 6 completed!")


if __name__ == "__main__":
    main()
