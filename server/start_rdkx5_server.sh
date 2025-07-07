#!/bin/bash

echo "=== SpatialLM RDK X5服务器启动脚本 ==="
echo

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请先安装Python"
    exit 1
fi

# 检查必要文件
if [ ! -f "inference_cpu.py" ]; then
    echo "❌ inference_cpu.py 未找到"
    exit 1
fi

if [ ! -f "visualize.py" ]; then
    echo "❌ visualize.py 未找到"
    exit 1
fi

if [ ! -f "rdkx5_server.py" ]; then
    echo "❌ rdkx5_server.py 未找到"
    exit 1
fi

# 激活conda环境
echo "🔧 激活spatiallm环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatiallm

if [ $? -ne 0 ]; then
    echo "❌ 无法激活spatiallm环境，请检查conda安装"
    exit 1
fi

# 安装依赖
echo "📦 检查并安装依赖..."
pip install -r requirements_web.txt

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

# 创建必要目录
mkdir -p received_files output_files

echo "✅ 环境检查完成"
echo
echo "🚀 启动RDK X5服务器..."
echo "   访问地址: http://localhost:5001"
echo "   按 Ctrl+C 停止服务"
echo

python rdkx5_server.py 