#!/usr/bin/env python3
"""
CUDA服务器 - 处理ply文件上传、编码和传输到RDK X5
"""

import os
import subprocess
import requests
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB最大文件大小

# 配置
UPLOAD_FOLDER = 'uploads'
ENCODED_FOLDER = 'encoded'
MODEL_NAME = 'manycore-research/SpatialLM1.1-Qwen-0.5B'
RDK_X5_URL = 'http://47.239.111.130:5001'  # RDK X5服务器地址

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENCODED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'ply'

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有找到文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '只支持.ply文件'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_id = Path(filename).stem  # 不带扩展名的文件名
        ply_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(ply_path)
        
        logger.info(f"文件已保存: {ply_path}")
        
        # 开始编码处理
        return process_encoding(file_id, ply_path)
        
    except Exception as e:
        logger.error(f"上传文件时出错: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

def process_encoding(file_id, ply_path):
    """编码ply文件"""
    try:
        pkl_path = os.path.join(ENCODED_FOLDER, f"{file_id}_encoded.pkl")
        
        # 调用编码脚本
        logger.info(f"开始编码: {ply_path}")
        cmd = [
            'python', 'encode_pointcloud.py',
            '-p', ply_path,
            '-o', pkl_path,
            '-m', MODEL_NAME
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"编码失败: {result.stderr}")
            return jsonify({'error': f'编码失败: {result.stderr}'}), 500
        
        logger.info(f"编码完成: {pkl_path}")
        
        # 发送文件到RDK X5
        return send_to_rdkx5(file_id, ply_path, pkl_path)
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': '编码超时（5分钟）'}), 500
    except Exception as e:
        logger.error(f"编码时出错: {str(e)}")
        return jsonify({'error': f'编码失败: {str(e)}'}), 500

def send_to_rdkx5(file_id, ply_path, pkl_path):
    """发送文件到RDK X5"""
    try:
        logger.info(f"发送文件到RDK X5: {file_id}")
        
        # 准备文件
        files = {
            'ply_file': (f"{file_id}.ply", open(ply_path, 'rb'), 'application/octet-stream'),
            'pkl_file': (f"{file_id}_encoded.pkl", open(pkl_path, 'rb'), 'application/octet-stream')
        }
        
        data = {'file_id': file_id}
        
        # 发送到RDK X5
        response = requests.post(
            f"{RDK_X5_URL}/process",
            files=files,
            data=data,
            timeout=1800  # 30分钟超时
        )
        
        # 关闭文件
        for file_obj in files.values():
            file_obj[1].close()
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"RDK X5处理成功: {result}")
            return jsonify({
                'success': True,
                'file_id': file_id,
                'download_url': f"{RDK_X5_URL}/download/{file_id}.rrd",
                'message': '处理完成，可以下载rrd文件'
            })
        else:
            logger.error(f"RDK X5处理失败: {response.text}")
            return jsonify({'error': f'RDK X5处理失败: {response.text}'}), 500
            
    except requests.RequestException as e:
        logger.error(f"发送到RDK X5时出错: {str(e)}")
        return jsonify({'error': f'发送到RDK X5失败: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"发送文件时出错: {str(e)}")
        return jsonify({'error': f'发送失败: {str(e)}'}), 500

@app.route('/status')
def status():
    """检查服务状态"""
    return jsonify({
        'status': 'running',
        'message': 'CUDA服务器运行中'
    })

if __name__ == '__main__':
    # 检查必要的脚本是否存在
    if not os.path.exists('encode_pointcloud.py'):
        logger.error("encode_pointcloud.py 脚本不存在")
        exit(1)
    
    logger.info("启动CUDA服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True) 