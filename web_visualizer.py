#!/usr/bin/env python3
"""
Web可视化服务器
提供交互式的Transformer中间状态可视化界面
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
import torch
import traceback
from pathlib import Path
import gzip
import time

# 导入我们的分析器
from main import TransformerInspector
from model_loader import ModelLoader
from token_tracker import TokenTracker
from coordinate_system import CoordinateManager

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储当前分析状态
current_inspector = None
current_analysis_result = None
current_token_info = None

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载模型API"""
    global current_inspector
    
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'gpt2')
        
        # 创建分析器并加载模型
        current_inspector = TransformerInspector()
        current_inspector.load_model(model_name)
        
        # 获取模型信息
        model_info = {
            'model_type': model_name,
            'num_parameters': current_inspector.loader.model.num_parameters(),
            'num_layers': current_inspector.loader.model.config.n_layer,
            'num_heads': current_inspector.loader.model.config.n_head,
            'hidden_size': current_inspector.loader.model.config.n_embd,
            'vocab_size': current_inspector.loader.model.config.vocab_size
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'message': f'模型 {model_name} 加载成功'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/analyze_text', methods=['POST'])
def analyze_text():
    """分析文本API"""
    global current_inspector, current_analysis_result, current_token_info
    
    try:
        if current_inspector is None:
            return jsonify({
                'success': False,
                'error': '请先加载模型'
            })
        
        data = request.get_json()
        text = data.get('text', '')
        target_tokens = data.get('target_tokens', None)
        
        if not text.strip():
            return jsonify({
                'success': False,
                'error': '请输入文本'
            })
        
        # 执行分析
        result = current_inspector.analyze_text(
            text=text,
            target_tokens=target_tokens,
            export_formats=["json"]
        )
        
        current_analysis_result = result
        current_token_info = result['token_info']
        
        # 构建响应数据
        response_data = {
            'success': True,
            'tokens': current_token_info['tokens'],
            'token_ids': current_token_info['token_ids'].tolist(),
            'analysis_summary': result['analysis_report']['summary'],
            'token_analysis': result['analysis_report']['token_analysis'],
            'coordinate_summary': result['coordinate_summary']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/get_token_details/<int:token_position>')
def get_token_details(token_position):
    """获取特定token的详细信息"""
    global current_analysis_result, current_token_info
    
    try:
        if current_analysis_result is None:
            return jsonify({
                'success': False,
                'error': '请先进行文本分析'
            })
        
        if token_position >= len(current_token_info['tokens']):
            return jsonify({
                'success': False,
                'error': 'Token位置超出范围'
            })
        
        # 获取该token的详细分析
        position_key = f'position_{token_position}'
        token_analysis = current_analysis_result['analysis_report']['token_analysis'].get(position_key, {})
        
        # 获取coordinate manager中的相关变量
        coord_manager = current_inspector.coordinate_manager
        position_vars = coord_manager.get_position_variables(token_position)
        
        # 构建层级数据
        layer_data = {}
        for layer_idx in range(current_inspector.loader.model.config.n_layer):
            layer_vars = coord_manager.get_layer_variables(layer_idx)
            position_layer_vars = [v for v in layer_vars if v['coordinate'].sequence_idx == token_position]
            
            layer_info = {
                'layer_index': layer_idx,
                'variables': [],
                'hidden_state_norm': None,
                'attention_weights': None
            }
            
            for var_info in position_layer_vars:
                coord = var_info['coordinate']
                value = var_info['value']
                
                var_data = {
                    'module_type': coord.module_type,
                    'module_name': coord.module_name,
                    'shape': list(value.shape) if hasattr(value, 'shape') else [],
                    'norm': float(torch.norm(value)) if hasattr(value, 'norm') else None,
                    'mean': float(torch.mean(value)) if hasattr(value, 'mean') else None,
                    'std': float(torch.std(value)) if hasattr(value, 'std') else None
                }
                
                if coord.module_type == 'hidden_state':
                    layer_info['hidden_state_norm'] = var_data['norm']
                elif coord.module_type == 'attention' and coord.head_idx is not None:
                    if layer_info['attention_weights'] is None:
                        layer_info['attention_weights'] = {}
                    layer_info['attention_weights'][coord.head_idx] = var_data
                
                layer_info['variables'].append(var_data)
            
            layer_data[layer_idx] = layer_info
        
        # 获取注意力流信息
        attention_flow = {}
        if hasattr(current_inspector.tracker, 'get_attention_flow'):
            try:
                attention_flow = current_inspector.tracker.get_attention_flow(token_position)
            except:
                pass
        
        response_data = {
            'success': True,
            'token_position': token_position,
            'token_text': current_token_info['tokens'][token_position],
            'token_id': int(current_token_info['token_ids'][token_position]),
            'layer_data': layer_data,
            'attention_flow': attention_flow,
            'analysis': token_analysis
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/get_attention_matrix')
def get_attention_matrix():
    """获取完整的注意力矩阵"""
    global current_analysis_result
    
    try:
        if current_analysis_result is None:
            return jsonify({
                'success': False,
                'error': '请先进行文本分析'
            })
        
        # 从tracking结果中获取注意力模式
        tracking_result = current_analysis_result['tracking_result']
        attention_patterns = tracking_result.get('attention_patterns', {})
        
        # 构建注意力矩阵数据
        attention_matrices = {}
        
        for layer_key, layer_attention in attention_patterns.items():
            if 'attention_weights' in layer_attention:
                # 转换注意力权重为可序列化的格式
                attention_weights = layer_attention['attention_weights']
                if torch.is_tensor(attention_weights):
                    # 假设形状为 [num_heads, seq_len, seq_len]
                    attention_matrices[layer_key] = {
                        'weights': attention_weights.detach().cpu().numpy().tolist(),
                        'shape': list(attention_weights.shape)
                    }
        
        return jsonify({
            'success': True,
            'attention_matrices': attention_matrices,
            'sequence_length': len(current_token_info['tokens'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/get_model_structure')
def get_model_structure():
    """获取模型结构信息"""
    global current_inspector
    
    try:
        if current_inspector is None:
            return jsonify({
                'success': False,
                'error': '请先加载模型'
            })
        
        model = current_inspector.loader.model
        
        # 构建模型结构树
        structure = {
            'model_type': model.config.model_type,
            'num_layers': model.config.n_layer,
            'num_heads': model.config.n_head,
            'hidden_size': model.config.n_embd,
            'layers': []
        }
        
        for i in range(model.config.n_layer):
            layer_info = {
                'index': i,
                'name': f'transformer.h.{i}',
                'components': [
                    {
                        'name': 'attention',
                        'type': 'multi_head_attention',
                        'num_heads': model.config.n_head,
                        'head_dim': model.config.n_embd // model.config.n_head
                    },
                    {
                        'name': 'mlp',
                        'type': 'feedforward',
                        'hidden_size': model.config.n_embd * 4
                    },
                    {
                        'name': 'ln_1',
                        'type': 'layer_norm'
                    },
                    {
                        'name': 'ln_2', 
                        'type': 'layer_norm'
                    }
                ]
            }
            structure['layers'].append(layer_info)
        
        return jsonify({
            'success': True,
            'structure': structure
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    import os
    import socket
    
    # 创建必要的目录
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    Path('static/css').mkdir(exist_ok=True)
    Path('static/js').mkdir(exist_ok=True)
    
    # 获取本机IP地址
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # 端口配置
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 启动Transformer可视化服务器...")
    print("📍 本地访问: http://localhost:{}".format(port))
    print("🌐 内网访问: http://{}:{}".format(local_ip, port))
    print("🌍 外部访问: http://192.5.86.157:{}".format(port))
    print("💡 如果外部无法访问，请检查防火墙设置")
    print("")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 端口 {port} 已被占用，尝试使用端口 {port+1}")
            app.run(host='0.0.0.0', port=port+1, debug=True, threaded=True)
        else:
            raise e