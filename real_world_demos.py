#!/usr/bin/env python3
"""
实际应用示例集合
展示 Transformer 中间状态分析器在真实场景中的应用
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from main import TransformerInspector
from model_loader import ModelLoader  
from token_tracker import TokenTracker
from coordinate_system import CoordinateManager

def demo_sentiment_analysis():
    """Demo 1: 情感分析中的中间表示"""
    print("🎭 Demo 1: 情感分析 - 追踪情感词汇在各层的表示变化")
    print("="*70)
    
    inspector = TransformerInspector()
    inspector.load_model("gpt2")
    
    # 对比正面和负面句子
    positive_text = "I absolutely love this amazing product!"
    negative_text = "I completely hate this terrible product!"
    
    # 分析正面情感
    print("😊 分析正面句子...")
    pos_result = inspector.analyze_text(
        text=positive_text,
        target_tokens=["love", "amazing"],
        export_formats=["json"]
    )
    
    # 分析负面情感  
    print("😠 分析负面句子...")
    neg_result = inspector.analyze_text(
        text=negative_text,
        target_tokens=["hate", "terrible"],
        export_formats=["json"]
    )
    
    # 比较结果
    pos_report = pos_result['analysis_report']
    neg_report = neg_result['analysis_report']
    
    print("\n📊 对比分析:")
    print(f"正面句子 tokens: {pos_report['summary']['tokens']}")
    print(f"负面句子 tokens: {neg_report['summary']['tokens']}")
    
    # 分析情感词的隐藏状态范数变化
    print("\n💡 情感词汇的表示强度变化:")
    for pos_key, analysis in pos_report['token_analysis'].items():
        if 'hidden_state_norms' in analysis:
            norms = analysis['hidden_state_norms']
            print(f"  '{analysis['token_text']}' (正面): 均值范数 = {np.mean(norms):.3f}")
    
    for pos_key, analysis in neg_report['token_analysis'].items():
        if 'hidden_state_norms' in analysis:
            norms = analysis['hidden_state_norms']
            print(f"  '{analysis['token_text']}' (负面): 均值范数 = {np.mean(norms):.3f}")
    
    return pos_result, neg_result


def demo_word_similarity():
    """Demo 2: 词汇语义相似性分析"""
    print("\n🔍 Demo 2: 词汇语义相似性 - 同义词在模型内部的表示")
    print("="*70)
    
    loader = ModelLoader()
    loader.load_model("gpt2")
    tracker = TokenTracker(loader.model, loader.tokenizer)
    
    # 测试同义词组
    synonym_pairs = [
        ("The car is fast", "car", "auto"),
        ("The house is big", "big", "large"),
        ("I am happy today", "happy", "joyful")
    ]
    
    for text_template, word1, word2 in synonym_pairs:
        print(f"\n🔤 分析同义词对: '{word1}' vs '{word2}'")
        
        # 创建包含两个词的句子
        text1 = text_template.replace(word1, word1)
        text2 = text_template.replace(word1, word2)
        
        # 分析第一个词
        inputs1 = loader.tokenize_text(text1)
        result1 = tracker.trace_tokens(inputs1, trace_all=True)
        token_info1 = loader.get_token_info(inputs1['input_ids'])
        
        # 分析第二个词
        inputs2 = loader.tokenize_text(text2)
        result2 = tracker.trace_tokens(inputs2, trace_all=True)
        token_info2 = loader.get_token_info(inputs2['input_ids'])
        
        # 找到目标词的位置
        pos1 = None
        pos2 = None
        
        for i, token in enumerate(token_info1['tokens']):
            if word1.lower() in token.lower():
                pos1 = i
                break
                
        for i, token in enumerate(token_info2['tokens']):
            if word2.lower() in token.lower():
                pos2 = i
                break
        
        if pos1 is not None and pos2 is not None:
            # 比较相同位置的表示
            print(f"  '{word1}' 位置: {pos1}, '{word2}' 位置: {pos2}")
            
            # 简单的层间相似度分析
            word1_states = result1['token_states']
            word2_states = result2['token_states']
            
            similarities = []
            for layer_idx in range(12):  # GPT-2 has 12 layers
                key1 = f'position_{pos1}_layer_{layer_idx}'
                key2 = f'position_{pos2}_layer_{layer_idx}'
                
                if key1 in word1_states and key2 in word2_states:
                    h1 = word1_states[key1]['hidden_state']
                    h2 = word2_states[key2]['hidden_state']
                    
                    if h1 is not None and h2 is not None:
                        # 计算余弦相似度
                        sim = torch.cosine_similarity(h1.flatten(), h2.flatten(), dim=0)
                        similarities.append(sim.item())
            
            if similarities:
                print(f"  各层平均相似度: {np.mean(similarities):.4f}")
                print(f"  最高相似度: {max(similarities):.4f} (层 {similarities.index(max(similarities))})")
                print(f"  最低相似度: {min(similarities):.4f} (层 {similarities.index(min(similarities))})")


def demo_attention_patterns():
    """Demo 3: 注意力模式分析"""
    print("\n👁️ Demo 3: 注意力模式 - 分析语法和语义关系")
    print("="*70)
    
    loader = ModelLoader()
    loader.load_model("gpt2")
    tracker = TokenTracker(loader.model, loader.tokenizer)
    
    # 包含不同语法关系的句子
    sentences = [
        "The red car drives quickly down the street",
        "She carefully reads the interesting book",
        "The scientist who discovered DNA won the prize"
    ]
    
    for i, text in enumerate(sentences):
        print(f"\n📖 句子 {i+1}: {text}")
        
        inputs = loader.tokenize_text(text)
        result = tracker.trace_tokens(inputs, trace_all=True)
        token_info = loader.get_token_info(inputs['input_ids'])
        
        print(f"  Tokens: {token_info['tokens']}")
        
        # 分析名词和形容词的注意力关系
        nouns = []
        adjectives = []
        
        # 简单的词性检测（基于常见模式）
        noun_indicators = ["car", "book", "scientist", "DNA", "prize", "street"]
        adj_indicators = ["red", "interesting", "careful", "quick"]
        
        for j, token in enumerate(token_info['tokens']):
            if any(indicator in token.lower() for indicator in noun_indicators):
                nouns.append((j, token))
            elif any(indicator in token.lower() for indicator in adj_indicators):
                adjectives.append((j, token))
        
        print(f"  识别的名词: {nouns}")
        print(f"  识别的形容词: {adjectives}")
        
        # 分析形容词对名词的注意力
        for adj_pos, adj_token in adjectives:
            if adj_pos < len(token_info['tokens']):
                attention_flow = tracker.get_attention_flow(adj_pos)
                
                print(f"\n  🎯 '{adj_token}' (位置 {adj_pos}) 的注意力:")
                
                # 查看前几层的注意力模式
                for layer in list(attention_flow['outgoing_attention'].keys())[:3]:
                    attn_info = attention_flow['outgoing_attention'][layer]
                    top_positions = attn_info['top_attended'][:3]
                    
                    attended_tokens = []
                    for pos in top_positions:
                        if pos < len(token_info['tokens']):
                            attended_tokens.append(token_info['tokens'][pos])
                    
                    print(f"    层 {layer}: 关注 → {attended_tokens}")


def demo_linguistic_phenomena():
    """Demo 4: 语言现象分析"""
    print("\n🗣️ Demo 4: 语言现象 - 分析一词多义和上下文理解")
    print("="*70)
    
    inspector = TransformerInspector()
    inspector.load_model("gpt2")
    
    # 一词多义的例子
    polysemy_examples = [
        ("The bank is closed today", "bank"),           # 银行
        ("The river bank is muddy", "bank"),            # 河岸
        ("I can see the light", "light"),               # 光线
        ("The box is very light", "light"),             # 轻的
        ("Time flies like an arrow", "flies"),          # 飞行
        ("Fruit flies like a banana", "flies")          # 果蝇
    ]
    
    polysemy_results = {}
    
    for text, target_word in polysemy_examples:
        print(f"\n📝 分析: {text}")
        
        result = inspector.analyze_text(
            text=text,
            target_tokens=[target_word],
            export_formats=["json"]
        )
        
        polysemy_results[text] = result
        
        # 分析目标词的表示
        report = result['analysis_report']
        for pos_key, analysis in report['token_analysis'].items():
            if target_word in analysis['token_text']:
                norms = analysis.get('hidden_state_norms', [])
                if norms:
                    print(f"  '{analysis['token_text']}' 表示强度: {np.mean(norms):.3f}")
    
    # 比较相同词在不同上下文中的表示
    print(f"\n🔄 一词多义比较分析:")
    word_contexts = {}
    
    for text, target_word in polysemy_examples:
        if target_word not in word_contexts:
            word_contexts[target_word] = []
        word_contexts[target_word].append((text, polysemy_results[text]))
    
    for word, contexts in word_contexts.items():
        if len(contexts) > 1:
            print(f"\n  '{word}' 在不同上下文:")
            for text, result in contexts:
                report = result['analysis_report']
                for pos_key, analysis in report['token_analysis'].items():
                    if word in analysis['token_text']:
                        norms = analysis.get('hidden_state_norms', [])
                        if norms:
                            context_desc = text.replace(word, f"[{word}]")
                            print(f"    {context_desc}: 均值强度 = {np.mean(norms):.3f}")


def demo_multilingual_analysis():
    """Demo 5: 多语言分析（如果模型支持）"""
    print("\n🌍 Demo 5: 多语言文本分析")
    print("="*70)
    
    inspector = TransformerInspector()
    inspector.load_model("gpt2")  # Note: GPT-2 主要是英文，但我们可以测试其他字符
    
    # 包含不同字符和概念的文本
    multilingual_texts = [
        "Hello world from Earth",
        "The number 42 is special",
        "Email: user@example.com",
        "Python code: print('hello')",
        "Math: x = 2 + 2 = 4"
    ]
    
    for text in multilingual_texts:
        print(f"\n🔤 分析文本: {text}")
        
        result = inspector.analyze_text(
            text=text,
            target_tokens=None,  # 分析所有 tokens
            export_formats=["json"]
        )
        
        report = result['analysis_report']
        print(f"  总 tokens: {report['summary']['total_tokens']}")
        print(f"  Token 列表: {report['summary']['tokens']}")
        
        # 分析特殊字符和符号的处理
        for pos_key, analysis in report['token_analysis'].items():
            token_text = analysis['token_text']
            norms = analysis.get('hidden_state_norms', [])
            
            if norms:
                # 检测特殊字符
                has_numbers = any(c.isdigit() for c in token_text)
                has_symbols = any(c in '@.=+()' for c in token_text)
                
                characteristics = []
                if has_numbers:
                    characteristics.append("数字")
                if has_symbols:
                    characteristics.append("符号")
                
                char_str = f" ({','.join(characteristics)})" if characteristics else ""
                print(f"  '{token_text}'{char_str}: 均值强度 = {np.mean(norms):.3f}")


def demo_performance_analysis():
    """Demo 6: 性能和资源使用分析"""
    print("\n⚡ Demo 6: 性能分析 - 监控计算资源使用")
    print("="*70)
    
    import psutil
    import gc
    
    inspector = TransformerInspector()
    inspector.load_model("gpt2")
    
    # 测试不同长度文本的处理性能
    test_texts = [
        ("Short", "Hello world"),
        ("Medium", "The quick brown fox jumps over the lazy dog in the forest"),
        ("Long", "Natural language processing is a fascinating field that combines computer science and linguistics to help computers understand and process human language in a meaningful way, enabling applications like translation, summarization, and conversational AI")
    ]
    
    performance_results = []
    
    for name, text in test_texts:
        print(f"\n🧪 测试 {name} 文本 ({len(text.split())} 词):")
        print(f"  文本: {text[:50]}...")
        
        # 记录初始状态
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 执行分析
        result = inspector.analyze_text(
            text=text,
            target_tokens=None,  # 分析所有 tokens
            export_formats=["json"]
        )
        
        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 计算性能指标
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        tokens_processed = result['analysis_report']['summary']['total_tokens']
        
        performance_data = {
            'name': name,
            'duration': duration,
            'memory_usage': memory_usage,
            'tokens_processed': tokens_processed,
            'tokens_per_second': tokens_processed / duration if duration > 0 else 0
        }
        
        if torch.cuda.is_available():
            gpu_memory_usage = end_gpu_memory - start_gpu_memory
            performance_data['gpu_memory_usage'] = gpu_memory_usage
            print(f"  GPU 内存使用: {gpu_memory_usage:.1f} MB")
        
        performance_results.append(performance_data)
        
        print(f"  处理时间: {duration:.2f} 秒")
        print(f"  内存使用: {memory_usage:.1f} MB")
        print(f"  Token 数量: {tokens_processed}")
        print(f"  处理速度: {performance_data['tokens_per_second']:.1f} tokens/秒")
    
    # 性能总结
    print(f"\n📈 性能总结:")
    for perf in performance_results:
        print(f"  {perf['name']}: {perf['tokens_per_second']:.1f} tokens/秒, {perf['memory_usage']:.1f} MB 内存")
    
    return performance_results


def run_all_demos():
    """运行所有演示"""
    print("🚀 Transformer 中间状态分析器 - 实际应用演示")
    print("="*80)
    
    try:
        # 确保输出目录存在
        Path("./outputs").mkdir(exist_ok=True)
        
        # 运行所有演示
        demo_sentiment_analysis()
        demo_word_similarity()
        demo_attention_patterns()
        demo_linguistic_phenomena()
        demo_multilingual_analysis()
        demo_performance_analysis()
        
        print("\n" + "="*80)
        print("🎉 所有演示完成！")
        print("📁 查看 ./outputs 目录获取详细分析结果")
        print("💡 这些演示展示了工具在实际 NLP 研究中的应用潜力")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    demos = {
        1: demo_sentiment_analysis,
        2: demo_word_similarity, 
        3: demo_attention_patterns,
        4: demo_linguistic_phenomena,
        5: demo_multilingual_analysis,
        6: demo_performance_analysis
    }
    
    if len(sys.argv) > 1:
        demo_num = int(sys.argv[1])
        if demo_num in demos:
            demos[demo_num]()
        else:
            print(f"演示 {demo_num} 不存在")
            print(f"可用演示: {list(demos.keys())}")
    else:
        run_all_demos()