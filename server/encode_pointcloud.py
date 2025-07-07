#!/usr/bin/env python3
"""
点云编码器脚本 - 在CUDA环境中运行
将点云编码为特征向量并保存，供后续在CPU环境中推理使用
"""

import os
import glob
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    """预处理点云数据"""
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def encode_point_cloud(model, point_cloud):
    """使用模型的点云编码器对点云进行编码"""
    model.eval()
    with torch.no_grad():
        # 调用模型的点云编码函数
        point_cloud = point_cloud.to(model.device)
        point_feature = model.forward_point_cloud(
            point_cloud[0], model.device, torch.float32
        )
        return point_feature.cpu()


def main():
    parser = argparse.ArgumentParser("SpatialLM 点云编码器")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="输入点云文件路径或包含多个点云文件的文件夹",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出编码特征的文件路径或文件夹",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
        help="模型路径",
    )
    parser.add_argument(
        "--no_cleanup",
        default=False,
        action="store_true",
        help="是否不清理点云",
    )
    
    args = parser.parse_args()

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU运行（速度较慢）")
        device = "cpu"
    else:
        device = "cuda"
        print(f"使用设备: {device}")

    # 加载模型（仅需要点云编码器部分）
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float32,
        device_map=device
    )
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # 获取配置
    num_bins = model.config.point_config["num_bins"]
    grid_size = Layout.get_grid_size(num_bins)

    # 确定输入文件
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))

    print(f"找到 {len(point_cloud_files)} 个点云文件")

    # 处理每个点云文件
    for point_cloud_file in tqdm(point_cloud_files, desc="编码点云"):
        # 加载点云
        point_cloud = load_o3d_pcd(point_cloud_file)
        
        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # 预处理点云
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # 编码点云
        print(f"正在编码: {os.path.basename(point_cloud_file)}")
        encoded_features = encode_point_cloud(model, input_pcd)

        # 准备保存的数据
        encoded_data = {
            "features": encoded_features,
            "min_extent": min_extent,
            "grid_size": grid_size,
            "num_bins": num_bins,
            "model_config": {
                "point_config": model.config.point_config,
                "model_path": args.model_path,
            },
            "file_info": {
                "original_file": point_cloud_file,
                "points_shape": points.shape,
                "colors_shape": colors.shape,
            }
        }

        # 保存编码结果
        if os.path.splitext(args.output)[-1] == '.pkl':
            # 单文件输出
            output_file = args.output
        else:
            # 文件夹输出
            output_filename = os.path.basename(point_cloud_file).replace(".ply", "_encoded.pkl")
            os.makedirs(args.output, exist_ok=True)
            output_file = os.path.join(args.output, output_filename)

        with open(output_file, 'wb') as f:
            pickle.dump(encoded_data, f)
        
        print(f"已保存编码特征到: {output_file}")

    print("点云编码完成！")
    print(f"编码特征已保存，可以传输到RDK X5进行推理")


if __name__ == "__main__":
    main() 