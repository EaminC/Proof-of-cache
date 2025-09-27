#!/usr/bin/env python3
"""
Dataset Generator Tool
Generate multiple KV cache datasets for analysis and training.
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any
import random
import time
from pathlib import Path

# Add src to path to import the main module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gpt2_kvcache import GPT2KVCacheExtractor


class DatasetGenerator:
    """Dataset Generator for KV Cache data"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize dataset generator
        
        Args:
            model_name: GPT-2 model name
        """
        self.extractor = GPT2KVCacheExtractor(model_name=model_name)
        self.model_name = model_name
    
    def generate_single_dataset(self, input_text: str, output_dir: str, 
                              temperature: float = 0.8, top_k: int = 10) -> str:
        """
        Generate single dataset
        
        Args:
            input_text: Input text
            output_dir: Output directory
            temperature: Generation temperature
            top_k: Top-k candidates
            
        Returns:
            Output file path
        """
        print(f"Processing: '{input_text}'")
        
        # Extract KV cache
        result = self.extractor.extract_kv_cache(
            input_text,
            temperature=temperature,
            top_k=top_k
        )
        
        # Create output filename
        safe_text = "".join(c for c in input_text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')[:50]  # Limit length
        output_file = os.path.join(output_dir, f"{safe_text}.json")
        
        # Save result
        self.extractor.save_to_json(result, output_file)
        
        print(f"  → Saved to: {output_file}")
        return output_file
    
    def generate_from_text_file(self, text_file: str, output_dir: str, 
                               temperature: float = 0.8, top_k: int = 10) -> List[str]:
        """
        Generate datasets from text file (one line per input)
        
        Args:
            text_file: Input text file path
            output_dir: Output directory
            temperature: Generation temperature
            top_k: Top-k candidates
            
        Returns:
            List of output file paths
        """
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Processing {len(lines)} inputs from {text_file}")
        
        for i, line in enumerate(lines):
            input_text = line.strip()
            if not input_text:
                continue
            
            try:
                output_file = self.generate_single_dataset(
                    input_text, output_dir, temperature, top_k
                )
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing line {i+1}: {e}")
                continue
        
        return output_files
    
    def generate_from_prompts(self, prompts: List[str], output_dir: str,
                             temperature: float = 0.8, top_k: int = 10) -> List[str]:
        """
        Generate datasets from prompt list
        
        Args:
            prompts: List of input prompts
            output_dir: Output directory
            temperature: Generation temperature
            top_k: Top-k candidates
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        print(f"Processing {len(prompts)} prompts")
        
        for i, prompt in enumerate(prompts):
            try:
                output_file = self.generate_single_dataset(
                    prompt, output_dir, temperature, top_k
                )
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing prompt {i+1}: {e}")
                continue
        
        return output_files
    
    def generate_random_dataset(self, num_samples: int, output_dir: str,
                              min_length: int = 3, max_length: int = 10,
                              temperature: float = 0.8, top_k: int = 10) -> List[str]:
        """
        Generate random dataset with random prompts
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            min_length: Minimum prompt length
            max_length: Maximum prompt length
            temperature: Generation temperature
            top_k: Top-k candidates
            
        Returns:
            List of output file paths
        """
        # Common words for random prompts
        words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their", "me", "him", "us", "them",
            "cat", "dog", "house", "car", "book", "computer", "music", "art", "science", "technology",
            "love", "hate", "good", "bad", "big", "small", "fast", "slow", "hot", "cold",
            "happy", "sad", "angry", "excited", "tired", "hungry", "thirsty", "sleepy"
        ]
        
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        print(f"Generating {num_samples} random samples")
        
        for i in range(num_samples):
            # Generate random prompt
            prompt_length = random.randint(min_length, max_length)
            prompt_words = random.sample(words, prompt_length)
            prompt = " ".join(prompt_words).capitalize()
            
            try:
                output_file = self.generate_single_dataset(
                    prompt, output_dir, temperature, top_k
                )
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing sample {i+1}: {e}")
                continue
        
        return output_files
    
    def create_dataset_index(self, output_files: List[str], output_dir: str) -> str:
        """
        Create dataset index file
        
        Args:
            output_files: List of output file paths
            output_dir: Output directory
            
        Returns:
            Index file path
        """
        index_data = {
            "model": self.model_name,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(output_files),
            "files": []
        }
        
        for file_path in output_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    index_data["files"].append({
                        "file": os.path.basename(file_path),
                        "input_text": data["input_text"],
                        "input_tokens": data["input_tokens"],
                        "predicted_token": data["predicted_next_token"]["token"],
                        "sequence_length": data["metadata"]["sequence_length"]
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        index_file = os.path.join(output_dir, "dataset_index.json")
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset index created: {index_file}")
        return index_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='KV Cache Dataset Generator')
    parser.add_argument('--model', default='gpt2', help='GPT-2 model name (default: gpt2)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--temperature', type=float, default=0.8, help='Generation temperature')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k candidates')
    
    # Input options
    parser.add_argument('--text-file', help='Input text file (one line per prompt)')
    parser.add_argument('--prompts', nargs='+', help='List of input prompts')
    parser.add_argument('--random', type=int, metavar='NUM_SAMPLES',
                       help='Generate random dataset with NUM_SAMPLES samples')
    parser.add_argument('--random-min', type=int, default=3, help='Minimum prompt length for random')
    parser.add_argument('--random-max', type=int, default=10, help='Maximum prompt length for random')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = DatasetGenerator(model_name=args.model)
        
        output_files = []
        
        # Generate from text file
        if args.text_file:
            output_files = generator.generate_from_text_file(
                args.text_file, args.output_dir, args.temperature, args.top_k
            )
        
        # Generate from prompts
        elif args.prompts:
            output_files = generator.generate_from_prompts(
                args.prompts, args.output_dir, args.temperature, args.top_k
            )
        
        # Generate random dataset
        elif args.random:
            output_files = generator.generate_random_dataset(
                args.random, args.output_dir, args.random_min, args.random_max,
                args.temperature, args.top_k
            )
        
        else:
            print("Error: Must specify one of --text-file, --prompts, or --random")
            sys.exit(1)
        
        # Create dataset index
        if output_files:
            generator.create_dataset_index(output_files, args.output_dir)
            print(f"\nDataset generation completed!")
            print(f"Generated {len(output_files)} samples in {args.output_dir}")
        else:
            print("No samples generated")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
