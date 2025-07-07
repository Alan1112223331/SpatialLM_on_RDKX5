#!/usr/bin/env python3
"""
RDK X5服务器 - 处理推理、可视化和文件下载
"""

import os
import subprocess
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB最大文件大小

# 配置
RECEIVED_FOLDER = 'received_files'
OUTPUT_FOLDER = 'output_files'
MODEL_NAME = 'manycore-research/SpatialLM1.1-Qwen-0.5B'

# 创建必要的文件夹
os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_files():
    """处理从CUDA服务器接收的文件"""
    try:
        # 检查文件
        if 'ply_file' not in request.files or 'pkl_file' not in request.files:
            return jsonify({'error': '缺少必要的文件'}), 400
        
        file_id = request.form.get('file_id')
        if not file_id:
            return jsonify({'error': '缺少file_id'}), 400
        
        ply_file = request.files['ply_file']
        pkl_file = request.files['pkl_file']
        
        # 保存接收的文件
        ply_filename = f"{file_id}.ply"
        pkl_filename = f"{file_id}_encoded.pkl"
        
        ply_path = os.path.join(RECEIVED_FOLDER, ply_filename)
        pkl_path = os.path.join(RECEIVED_FOLDER, pkl_filename)
        
        ply_file.save(ply_path)
        pkl_file.save(pkl_path)
        
        logger.info(f"文件已保存: {ply_path}, {pkl_path}")
        
        # 开始推理处理
        return run_inference(file_id, ply_path, pkl_path)
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

def run_inference(file_id, ply_path, pkl_path):
    """运行推理"""
    try:
        txt_path = os.path.join(OUTPUT_FOLDER, f"{file_id}.txt")
        
        # 调用推理脚本
        logger.info(f"开始推理: {pkl_path}")
        cmd = [
            'python', 'inference_cpu.py',
            '-f', pkl_path,
            '-o', txt_path,
            '-m', MODEL_NAME
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            logger.error(f"推理失败: {result.stderr}")
            return jsonify({'error': f'推理失败: {result.stderr}'}), 500
        
        logger.info(f"推理完成: {txt_path}")
        
        # 生成可视化
        return generate_visualization(file_id, ply_path, txt_path)
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': '推理超时（30分钟）'}), 500
    except Exception as e:
        logger.error(f"推理时出错: {str(e)}")
        return jsonify({'error': f'推理失败: {str(e)}'}), 500

def generate_visualization(file_id, ply_path, txt_path):
    """生成可视化文件"""
    try:
        rrd_path = os.path.join(OUTPUT_FOLDER, f"{file_id}.rrd")
        
        # 调用可视化脚本
        logger.info(f"开始生成可视化: {file_id}")
        cmd = [
            'python', 'visualize.py',
            '--point_cloud', ply_path,
            '--layout', txt_path,
            '--save', rrd_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"可视化失败: {result.stderr}")
            return jsonify({'error': f'可视化失败: {result.stderr}'}), 500
        
        logger.info(f"可视化完成: {rrd_path}")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'message': '处理完成',
            'rrd_path': rrd_path
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': '可视化超时（5分钟）'}), 500
    except Exception as e:
        logger.error(f"可视化时出错: {str(e)}")
        return jsonify({'error': f'可视化失败: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载生成的文件"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"下载文件时出错: {str(e)}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/status')
def status():
    """检查服务状态"""
    return jsonify({
        'status': 'running',
        'message': 'RDK X5服务器运行中'
    })

@app.route('/files')
def list_files():
    """列出可用的文件"""
    try:
        files = []
        if os.path.exists(OUTPUT_FOLDER):
            for filename in os.listdir(OUTPUT_FOLDER):
                if filename.endswith('.rrd'):
                    file_path = os.path.join(OUTPUT_FOLDER, filename)
                    files.append({
                        'filename': filename,
                        'size': os.path.getsize(file_path),
                        'created': os.path.getctime(file_path)
                    })
        
        return jsonify({
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        logger.error(f"列出文件时出错: {str(e)}")
        return jsonify({'error': f'列出文件失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 检查必要的脚本是否存在
    required_scripts = ['inference_cpu.py', 'visualize.py']
    for script in required_scripts:
        if not os.path.exists(script):
            logger.error(f"{script} 脚本不存在")
            exit(1)
    
    logger.info("启动RDK X5服务器...")
    app.run(host='0.0.0.0', port=5001, debug=True) 