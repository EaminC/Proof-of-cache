#!/bin/bash

# 🚀 Transformer 可视化器启动脚本

echo "🧠 启动 Transformer 中间状态可视化器..."
echo "======================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    echo "请确保已安装 Python 3.7 或更高版本"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
if [ -f "requirements.txt" ]; then
    python3 -c "
import pkg_resources
import sys

def check_requirements():
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().splitlines()
        
        missing = []
        for requirement in requirements:
            if requirement.strip() and not requirement.startswith('#'):
                package_name = requirement.split('>=')[0].split('==')[0].strip()
                try:
                    pkg_resources.get_distribution(package_name)
                except pkg_resources.DistributionNotFound:
                    missing.append(package_name)
        
        if missing:
            print(f'❌ 缺少依赖: {missing}')
            print('请运行: pip install -r requirements.txt')
            sys.exit(1)
        else:
            print('✅ 所有依赖已安装')
    except Exception as e:
        print(f'⚠️ 依赖检查警告: {e}')

check_requirements()
"
    if [ $? -ne 0 ]; then
        echo "正在安装依赖..."
        pip install -r requirements.txt
    fi
else
    echo "⚠️ 未找到 requirements.txt"
fi

# 检查必要目录
echo "📁 检查目录结构..."
for dir in "templates" "static" "static/css" "static/js" "outputs"; do
    if [ ! -d "$dir" ]; then
        echo "创建目录: $dir"
        mkdir -p "$dir"
    fi
done

# 检查必要文件
echo "📄 检查必要文件..."
required_files=(
    "web_visualizer.py"
    "main.py"
    "model_loader.py"
    "hook_system.py"
    "token_tracker.py"
    "coordinate_system.py"
    "templates/index.html"
    "static/css/style.css"
    "static/js/utils.js"
    "static/js/api.js"
    "static/js/visualizer.js"
    "static/js/main.js"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ 缺少必要文件:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo "请确保所有文件都已正确创建"
    exit 1
fi

echo "✅ 所有文件检查完成"

# 检查端口是否被占用
echo "🔍 检查端口 5000..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️ 端口 5000 已被占用"
    echo "正在查找可用端口..."
    
    # 寻找可用端口
    for port in {5001..5010}; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "✅ 使用端口: $port"
            export FLASK_PORT=$port
            break
        fi
    done
else
    echo "✅ 端口 5000 可用"
    export FLASK_PORT=5000
fi

# 设置环境变量
export FLASK_APP=web_visualizer.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# 显示启动信息
echo ""
echo "🎉 启动信息:"
echo "  - 应用文件: web_visualizer.py"
echo "  - 访问地址: http://localhost:${FLASK_PORT}"
echo "  - 调试模式: 开启"
echo ""
echo "💡 使用说明:"
echo "  1. 在浏览器中打开 http://localhost:${FLASK_PORT}"
echo "  2. 选择模型并点击'加载模型'"
echo "  3. 输入文本并点击'开始分析'"
echo "  4. 点击任意token查看详细信息和中间状态"
echo ""
echo "⌨️ 快捷键:"
echo "  - Ctrl+C: 停止服务器"
echo "  - Ctrl+Enter: 在页面中快速分析文本"
echo "  - Esc: 关闭侧边面板"
echo ""

# 启动Flask应用
echo "🚀 正在启动服务器..."
echo "======================================"

# 使用python3启动Flask应用
python3 web_visualizer.py

echo ""
echo "👋 服务器已停止"