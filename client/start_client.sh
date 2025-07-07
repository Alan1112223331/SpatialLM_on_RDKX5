#!/bin/bash

echo "=== SpatialLM CUDA服务器启动脚本 ==="
echo

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请先安装Python"
    exit 1
fi

# 检查必要文件
if [ ! -f "encode_pointcloud.py" ]; then
    echo "❌ encode_pointcloud.py 未找到"
    exit 1
fi

if [ ! -f "cuda_server.py" ]; then
    echo "❌ cuda_server.py 未找到"
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
mkdir -p uploads encoded

echo "✅ 环境检查完成"
echo
echo "🚀 启动CUDA服务器..."
echo "   访问地址: http://localhost:5000"
echo "   按 Ctrl+C 停止服务"
echo

python cuda_server.py 