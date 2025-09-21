"""
VLLM Intermediate State Inspector - Main Program
For analyzing intermediate states and token trajectories in Transformer models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Any

# 导入自定义模块
from model_loader import ModelLoader
from hook_system import IntermediateInspector
from token_tracker import TokenTracker
from coordinate_system import CoordinateManager, DataExporter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLLMInspector:
    """
    VLLM Intermediate State Inspector main class
    Integrates all functional modules and provides a unified interface
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize inspector
        
        Args:
            config_path: Configuration file path
        """
        self.loader = ModelLoader(config_path)
        self.inspector = None
        self.tracker = None
        self.coordinate_manager = None
        self.data_exporter = DataExporter()
        
    def load_model(self, model_name: str, model_type: str = "causal_lm") -> None:
        """Load model"""
        logger.info(f"Loading model: {model_name}")
        self.loader.load_model(model_name, model_type)
        
        # Initialize other components
        self.inspector = IntermediateInspector(self.loader.model)
        self.tracker = TokenTracker(self.loader.model, self.loader.tokenizer)
        self.coordinate_manager = CoordinateManager(model_name)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {self.loader.get_model_info()}")
    
    def analyze_text(self, 
                    text: str,
                    target_tokens: Optional[List[str]] = None,
                    output_dir: str = "./outputs",
                    export_formats: List[str] = ["json", "npz"]) -> Dict[str, Any]:
        """
        Complete text analysis workflow
        
        Args:
            text: Input text
            target_tokens: Target token list
            output_dir: Output directory
            export_formats: Export formats
        
        Returns:
            Analysis results
        """
        if self.loader.model is None:
            raise RuntimeError("Please load model first")
        
        logger.info(f"Starting text analysis: '{text}'")
        
        # 1. Tokenization
        inputs = self.loader.tokenize_text(text)
        token_info = self.loader.get_token_info(inputs['input_ids'])
        logger.info(f"Token info: {len(token_info['tokens'])} tokens")
        
        # 2. Set target tokens
        if target_tokens:
            self.tracker.set_target_tokens(target_tokens)
        
        # 3. Execute tracking
        logger.info("Executing token tracking...")
        tracking_result = self.tracker.trace_tokens(inputs, trace_all=(target_tokens is None))
        
        # 4. Collect intermediate variables
        logger.info("Collecting intermediate variables...")
        self._collect_intermediate_variables(tracking_result, token_info)
        
        # 5. Generate analysis report
        logger.info("Generating analysis report...")
        analysis_report = self._generate_analysis_report(tracking_result, token_info)
        
        # 6. 导出数据
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = tracking_result.get('timestamp', 'unknown')
        filename = f"analysis_{timestamp}"
        
        exported_files = self.coordinate_manager.export_all(
            filename, 
            formats=export_formats
        )
        
        logger.info(f"Data exported: {exported_files}")
        
        return {
            'tracking_result': tracking_result,
            'analysis_report': analysis_report,
            'token_info': token_info,
            'exported_files': exported_files,
            'coordinate_summary': self.coordinate_manager.summary()
        }
    
    def _collect_intermediate_variables(self, 
                                      tracking_result: Dict[str, Any],
                                      token_info: Dict[str, Any]) -> None:
        """Collect intermediate variables to coordinate manager"""
        self.coordinate_manager.clear()
        
        # 收集 token 状态
        for state_key, states in tracking_result['token_states'].items():
            for state in states:
                coord = self.coordinate_manager.create_coordinate(
                    batch_idx=state.coordinate.batch_idx,
                    sequence_idx=state.coordinate.sequence_idx,
                    layer_idx=state.coordinate.layer_idx,
                    module_type="hidden_state",
                    module_name=state_key
                )
                
                self.coordinate_manager.add_variable(
                    coord,
                    state.hidden_state,
                    f"Hidden state at layer {state.coordinate.layer_idx}, position {state.coordinate.sequence_idx}"
                )
        
        # 收集注意力权重
        for attn_key, attn_info in tracking_result['attention_patterns'].items():
            parts = attn_key.split('_')
            layer_idx = int(parts[1])
            pos = int(parts[3])
            
            # 出向注意力
            coord = self.coordinate_manager.create_coordinate(
                batch_idx=0,
                sequence_idx=pos,
                layer_idx=layer_idx,
                module_type="attention",
                module_name=f"{attn_key}_outgoing"
            )
            
            self.coordinate_manager.add_variable(
                coord,
                attn_info['as_query'],
                f"Outgoing attention weights from layer {layer_idx}, position {pos}"
            )
            
            # 入向注意力
            coord = self.coordinate_manager.create_coordinate(
                batch_idx=0,
                sequence_idx=pos,
                layer_idx=layer_idx,
                module_type="attention",
                module_name=f"{attn_key}_incoming"
            )
            
            self.coordinate_manager.add_variable(
                coord,
                attn_info['as_key'],
                f"Incoming attention weights to layer {layer_idx}, position {pos}"
            )
    
    def _generate_analysis_report(self, 
                                tracking_result: Dict[str, Any],
                                token_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            'summary': {
                'input_text': token_info['text'],
                'num_tokens': token_info['length'],
                'tokens': token_info['tokens'],
                'tracked_positions': tracking_result.get('target_positions', []),
                'num_layers_analyzed': len(set(
                    state.coordinate.layer_idx 
                    for states in tracking_result['token_states'].values() 
                    for state in states
                    if state.coordinate.layer_idx is not None
                ))
            },
            'token_analysis': {},
            'attention_analysis': {},
            'layer_analysis': {}
        }
        
        # 分析每个被追踪的 token
        for pos in tracking_result.get('target_positions', []):
            token_text = token_info['tokens'][pos] if pos < len(token_info['tokens']) else f"pos_{pos}"
            
            # 获取轨迹
            trajectory = self.tracker.get_token_trajectory(pos)
            attention_flow = self.tracker.get_attention_flow(pos)
            
            report['token_analysis'][f"position_{pos}"] = {
                'token_text': token_text,
                'trajectory_layers': list(trajectory['layers'].keys()),
                'hidden_state_norms': [
                    state['hidden_state'].norm().item() 
                    for state in trajectory['hidden_state_evolution']
                ],
                'attention_summary': {
                    layer: {
                        'self_attention': np.mean(flow['self_attention']) if 'self_attention' in flow else 0,
                        'top_attended': flow.get('top_attended', [])[:3],
                        'top_attending': flow.get('top_attending', [])[:3]
                    }
                    for layer, flow in attention_flow['outgoing_attention'].items()
                }
            }
        
        return report
    
    def visualize_results(self, 
                         analysis_result: Dict[str, Any],
                         save_plots: bool = True,
                         output_dir: str = "./outputs") -> None:
        """可视化分析结果"""
        tracking_result = analysis_result['tracking_result']
        analysis_report = analysis_result['analysis_report']
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Token 轨迹可视化
        self._plot_token_trajectories(tracking_result, analysis_report, output_path, save_plots)
        
        # 2. 注意力热图
        self._plot_attention_heatmaps(tracking_result, output_path, save_plots)
        
        # 3. 层级分析图
        self._plot_layer_analysis(analysis_report, output_path, save_plots)
    
    def _plot_token_trajectories(self, 
                               tracking_result: Dict[str, Any],
                               analysis_report: Dict[str, Any],
                               output_path: Path,
                               save_plots: bool) -> None:
        """绘制 token 轨迹图"""
        target_positions = tracking_result.get('target_positions', [])
        
        if not target_positions:
            return
        
        fig, axes = plt.subplots(len(target_positions), 1, figsize=(12, 4 * len(target_positions)))
        if len(target_positions) == 1:
            axes = [axes]
        
        for i, pos in enumerate(target_positions):
            trajectory = self.tracker.get_token_trajectory(pos)
            
            if trajectory['hidden_state_evolution']:
                layers = [state['layer'] for state in trajectory['hidden_state_evolution']]
                norms = [state['hidden_state'].norm().item() for state in trajectory['hidden_state_evolution']]
                
                axes[i].plot(layers, norms, 'b-o', linewidth=2, markersize=6)
                axes[i].set_title(f"Token {pos} 隐藏状态范数变化")
                axes[i].set_xlabel("层数")
                axes[i].set_ylabel("向量范数")
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_path / "token_trajectories.png", dpi=300, bbox_inches='tight')
            logger.info(f"Token 轨迹图已保存到: {output_path / 'token_trajectories.png'}")
        
        plt.show()
    
    def _plot_attention_heatmaps(self, 
                               tracking_result: Dict[str, Any],
                               output_path: Path,
                               save_plots: bool) -> None:
        """绘制注意力热图"""
        attention_patterns = tracking_result.get('attention_patterns', {})
        
        if not attention_patterns:
            return
        
        # 选择第一个注意力模式作为示例
        first_key = next(iter(attention_patterns.keys()))
        attn_info = attention_patterns[first_key]
        
        if 'as_query' in attn_info:
            attn_weights = attn_info['as_query'].mean(dim=0).numpy()  # 平均所有头
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attn_weights.reshape(1, -1),
                annot=True,
                fmt='.3f',
                cmap='Blues',
                cbar=True
            )
            plt.title(f"注意力权重热图 - {first_key}")
            plt.xlabel("目标位置")
            plt.ylabel("查询位置")
            
            if save_plots:
                plt.savefig(output_path / "attention_heatmap.png", dpi=300, bbox_inches='tight')
                logger.info(f"注意力热图已保存到: {output_path / 'attention_heatmap.png'}")
            
            plt.show()
    
    def _plot_layer_analysis(self, 
                           analysis_report: Dict[str, Any],
                           output_path: Path,
                           save_plots: bool) -> None:
        """绘制层级分析图"""
        summary = self.coordinate_manager.summary()
        
        if 'layer_distribution' in summary:
            layers = list(summary['layer_distribution'].keys())
            counts = list(summary['layer_distribution'].values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(layers, counts, color='skyblue', alpha=0.7)
            plt.title("各层捕获的中间变量数量")
            plt.xlabel("层数")
            plt.ylabel("变量数量")
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(output_path / "layer_analysis.png", dpi=300, bbox_inches='tight')
                logger.info(f"层级分析图已保存到: {output_path / 'layer_analysis.png'}")
            
            plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLLM 中间状态检查器")
    parser.add_argument("--model", type=str, default="gpt2", help="模型名称")
    parser.add_argument("--text", type=str, default="Hello world! How are you today?", help="输入文本")
    parser.add_argument("--target-tokens", type=str, nargs='+', help="目标 token 列表")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--export-formats", type=str, nargs='+', default=["json", "npz"], help="导出格式")
    parser.add_argument("--no-visualize", action="store_true", help="不生成可视化图表")
    
    args = parser.parse_args()
    
    # 创建检查器
    inspector = VLLMInspector()
    
    try:
        # 加载模型
        inspector.load_model(args.model)
        
        # 分析文本
        result = inspector.analyze_text(
            text=args.text,
            target_tokens=args.target_tokens,
            output_dir=args.output_dir,
            export_formats=args.export_formats
        )
        
        # Print analysis report
        print("\n" + "="*50)
        print("Analysis Report")
        print("="*50)
        
        report = result['analysis_report']
        print(f"Input text: {report['summary']['input_text']}")
        print(f"Token count: {report['summary']['num_tokens']}")
        print(f"Tracked positions: {report['summary']['tracked_positions']}")
        print(f"Analyzed layers: {report['summary']['num_layers_analyzed']}")
        
        # Coordinate system summary
        coord_summary = result['coordinate_summary']
        print(f"\nIntermediate variables statistics:")
        print(f"  Total variables: {coord_summary['total_variables']}")
        if coord_summary['layer_range']:
            print(f"  Layer range: {coord_summary['layer_range'][0]} - {coord_summary['layer_range'][1]}")
        
        # Export file information
        print(f"\nExported files:")
        for fmt, path in result['exported_files'].items():
            print(f"  {fmt.upper()}: {path}")
        
        # Visualization
        if not args.no_visualize:
            print("\nGenerating visualization charts...")
            inspector.visualize_results(result, save_plots=True, output_dir=args.output_dir)
        
        print(f"\nAnalysis completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()