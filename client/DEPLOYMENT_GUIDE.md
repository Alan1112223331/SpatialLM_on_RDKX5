# SpatialLM 分离部署指南

本指南帮助您在CUDA环境和RDK X5等无CUDA环境之间分离部署SpatialLM项目。

## 架构概述

```
[CUDA环境]                    [RDK X5环境]
点云文件 → 点云编码器 → 特征文件 → 传输 → LLM推理 → 可视化
```

## 环境准备

### CUDA环境（用于点云编码）

**系统要求：**
- 支持CUDA 12.4的GPU
- Python 3.11
- 足够的GPU内存（推荐8GB+）

**安装依赖：**
```bash
# 创建conda环境
conda create -n spatiallm-cuda python=3.11
conda activate spatiallm-cuda
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

# 安装poetry和基础依赖
pip install poetry && poetry config virtualenvs.create false --local
poetry install

# 安装点云编码器依赖
# SpatialLM1.0 依赖
poe install-torchsparse

# SpatialLM1.1 依赖（推荐）
poe install-sonata
```

### RDK X5环境（用于推理和可视化）

**系统要求：**
- RDK X5开发板
- 4GB虚拟内存（已配置）
- Python 3.11

**安装依赖：**
```bash
# 创建conda环境
conda create -n spatiallm-cpu python=3.11
conda activate spatiallm-cpu

# 仅安装CPU版本依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors pandas einops numpy scipy scikit-learn
pip install tokenizers huggingface_hub rerun-sdk shapely bbox terminaltables open3d
pip install addict

# 安装spatiallm包（不含CUDA依赖）
pip install -e . --no-deps
```

## 使用流程

### 第一步：在CUDA环境中编码点云

**基本用法：**
```bash
# 编码单个点云文件
python encode_pointcloud.py -p scene0000_00.ply -o scene0000_00_encoded.pkl -m manycore-research/SpatialLM1.1-Qwen-0.5B

# # 批量编码点云文件夹
# python encode_pointcloud.py -p pcd_folder/ -o encoded_features/ -m manycore-research/SpatialLM1.1-Qwen-0.5B
```

**完整示例：**
```bash
# 下载测试数据
huggingface-cli download manycore-research/SpatialLM-Testset pcd/scene0000_00.ply --repo-type dataset --local-dir .

# 编码点云
python encode_pointcloud.py \
    --point_cloud pcd/scene0000_00.ply \
    --output scene0000_00_encoded.pkl \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B
```

**输出文件结构：**
编码后的`.pkl`文件包含：
- `features`: 编码后的点云特征向量
- `min_extent`: 点云的最小坐标范围
- `grid_size`: 网格大小
- `num_bins`: 离散化分箱数
- `model_config`: 模型配置信息
- `file_info`: 原始文件信息

### 第二步：传输文件到RDK X5

将编码后的特征文件和必要的模型文件传输到RDK X5：

```bash
# 通过scp传输（示例）
scp scene0000_00_encoded.pkl user@rdkx5:/path/to/spatiallm/
scp code_template.txt user@rdkx5:/path/to/spatiallm/

# 或者通过USB/网络共享等方式传输
```

### 第三步：在RDK X5中推理

**基本用法：**
```bash
# 推理单个特征文件
python inference_cpu.py -f scene0000_00_encoded.pkl -o scene0000_00.txt -m manycore-research/SpatialLM1.1-Qwen-0.5B

# # 批量推理
# python inference_cpu.py -f encoded_features/ -o results/ -m manycore-research/SpatialLM1.1-Qwen-0.5B
```

**完整示例：**
```bash
python inference_cpu.py \
    --encoded_features scene0000_00_encoded.pkl \
    --output scene0000_00.txt \
    --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
    --detect_type all \
    --temperature 0.6 \
    --top_p 0.95
```

### 第四步：可视化结果

<!-- **从编码数据可视化：**
```bash
python visualize_cpu.py encoded \
    --encoded_file scene0000_00_encoded.pkl \
    --layout scene0000_00.txt \
    --save scene0000_00.rrd
```

**直接可视化（如果有原始点云文件）：**
```bash
python visualize_cpu.py direct \
    --point_cloud scene0000_00.ply \
    --layout scene0000_00.txt \
    --save scene0000_00.rrd
``` -->

```bash
python visualize.py --point_cloud pcd/scene0000_00.ply --layout scene0000_00.txt --save scene0000_00.rrd
```

<!-- 
## 性能优化建议

### CUDA环境优化
- 使用SpatialLM1.1模型以获得更好的性能
- 适当调整batch size以平衡速度和内存使用
- 考虑使用混合精度训练

### RDK X5环境优化
- 使用float32精度以保证兼容性
- 适当调整温度参数以平衡质量和速度
- 监控内存使用，必要时降低max_new_tokens

```bash
# 内存优化示例
python inference_cpu.py \
    --encoded_features scene.pkl \
    --output result.txt \
    --inference_dtype float32 \
    --temperature 0.7 \
    --max_new_tokens 2048
```

## 故障排除

### 常见问题

**1. CUDA环境中点云编码器安装失败**
```bash
# 检查CUDA版本
nvidia-smi
nvcc --version

# 重新安装对应版本的torch
pip uninstall torch torchvision torchaudio
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 -f https://download.pytorch.org/whl/torch_stable.html
```

**2. RDK X5中内存不足**
```bash
# 检查内存使用
free -h
# 检查虚拟内存
swapon --show

# 降低模型精度
python inference_cpu.py --inference_dtype float32 ...
```

**3. 编码文件不兼容**
确保CUDA环境和RDK X5环境使用相同版本的：
- PyTorch版本
- transformers版本  
- spatiallm包版本

**4. 可视化无法显示**
```bash
# 检查rerun安装
pip list | grep rerun

# 保存为文件而不是实时显示
python visualize_cpu.py encoded -f xxx.pkl -l xxx.txt -s output.rrd
```

## 文件管理建议

### 目录结构
```
spatiallm/
├── cuda_env/              # CUDA环境
│   ├── pcd/              # 原始点云文件
│   ├── encoded/          # 编码后的特征文件
│   └── encode_pointcloud.py
├── rdkx5_env/            # RDK X5环境
│   ├── features/         # 传输来的特征文件
│   ├── results/          # 推理结果
│   ├── visualizations/   # 可视化文件
│   ├── inference_cpu.py
│   └── visualize_cpu.py
└── shared/
    └── code_template.txt  # 共享的代码模板
```

### 批处理脚本示例

**CUDA环境批处理脚本：**
```bash
#!/bin/bash
# encode_batch.sh
for ply_file in pcd/*.ply; do
    base_name=$(basename "$ply_file" .ply)
    echo "编码 $ply_file..."
    python encode_pointcloud.py \
        -p "$ply_file" \
        -o "encoded/${base_name}_encoded.pkl" \
        -m manycore-research/SpatialLM1.1-Qwen-0.5B
done
```

**RDK X5批处理脚本：**
```bash
#!/bin/bash
# inference_batch.sh
for feature_file in features/*_encoded.pkl; do
    base_name=$(basename "$feature_file" _encoded.pkl)
    echo "推理 $feature_file..."
    python inference_cpu.py \
        -f "$feature_file" \
        -o "results/${base_name}.txt" \
        -m manycore-research/SpatialLM1.1-Qwen-0.5B
done
```

## 检查清单

### 部署前检查
- [ ] CUDA环境GPU可用且内存充足
- [ ] RDK X5虚拟内存已配置
- [ ] 两个环境的Python包版本兼容
- [ ] 网络连接正常（用于模型下载）

### 运行前检查
- [ ] 点云文件格式正确（.ply）
- [ ] code_template.txt文件存在
- [ ] 编码特征文件完整传输
- [ ] 磁盘空间充足

### 结果验证
- [ ] 编码特征文件大小合理
- [ ] 推理结果包含预期元素
- [ ] 可视化显示正常
- [ ] 性能指标满足要求

通过这种分离部署方式，您可以充分利用CUDA环境的计算能力进行点云编码，同时在RDK X5等边缘设备上实现高效的推理和可视化。  -->