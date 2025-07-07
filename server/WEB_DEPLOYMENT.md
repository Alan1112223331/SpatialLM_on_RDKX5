# SpatialLM 网页服务部署指南

## 系统架构

本系统包含两个服务器：
- **CUDA服务器**: 处理ply文件上传、编码和文件传输
- **RDK X5服务器**: 处理推理、可视化和文件下载

## 部署步骤

### 1. CUDA服务器端 (编码服务器)

1. **环境准备**
```bash
# 激活CUDA环境
conda activate your_cuda_env

# 安装网页服务依赖
pip install -r requirements_web.txt
```

2. **修改配置**
编辑 `cuda_server.py` 第26行，设置RDK X5服务器地址：
```python
RDK_X5_URL = 'http://YOUR_RDK_X5_IP:5001'  # 替换为实际IP
```

3. **启动服务**
```bash
python cuda_server.py
```

服务将在 `http://0.0.0.0:5000` 启动

### 2. RDK X5服务器端 (推理服务器)

1. **环境准备**
```bash
# 激活spatiallm环境
conda activate spatiallm

# 安装网页服务依赖
pip install -r requirements_web.txt
```

2. **启动服务**
```bash
python rdkx5_server.py
```

服务将在 `http://0.0.0.0:5001` 启动

## 使用流程

1. 用户访问CUDA服务器的网页界面 (`http://CUDA_SERVER_IP:5000`)
2. 上传 `.ply` 文件
3. 系统自动完成以下步骤：
   - CUDA环境编码点云 → 生成 `.pkl` 文件
   - 发送 `.ply` 和 `.pkl` 到RDK X5
   - RDK X5推理 → 生成 `.txt` 文件
   - 可视化 → 生成 `.rrd` 文件
   - 提供下载链接

## 文件结构

```
SpatialLM/
├── cuda_server.py          # CUDA服务器
├── rdkx5_server.py         # RDK X5服务器
├── templates/
│   └── index.html          # 前端页面
├── requirements_web.txt    # 网页服务依赖
├── encode_pointcloud.py    # 编码脚本
├── inference_cpu.py        # 推理脚本
└── visualize.py           # 可视化脚本
```

## 测试

1. **检查服务状态**
```bash
# CUDA服务器
curl http://localhost:5000/status

# RDK X5服务器
curl http://localhost:5001/status
```

2. **测试完整流程**
- 访问 `http://CUDA_SERVER_IP:5000`
- 上传测试的 `.ply` 文件
- 观察处理过程和结果

## 注意事项

1. **网络连接**: 确保CUDA服务器能够访问RDK X5服务器
2. **防火墙**: 开放端口5000(CUDA)和5001(RDK X5)
3. **存储空间**: 确保有足够空间存储临时文件
4. **超时设置**: 大文件处理可能需要更长时间，可调整超时参数
5. **日志监控**: 服务器会输出详细日志，便于调试

## 故障排除

1. **编码失败**: 检查CUDA环境和模型是否正确加载
2. **推理失败**: 检查RDK X5环境和依赖是否完整
3. **文件传输失败**: 检查网络连接和服务器状态
4. **下载失败**: 检查生成的文件是否存在

## 扩展

- 可以添加用户认证
- 可以添加文件管理功能
- 可以添加处理队列
- 可以添加实时状态更新 