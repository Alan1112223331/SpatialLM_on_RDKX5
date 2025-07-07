#!/usr/bin/env python3
"""
CPU推理脚本 - 适用于RDK X5等无CUDA环境
加载预编码的点云特征进行推理和可视化
"""

import os
import glob
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer, set_seed

from spatiallm import Layout


DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}


def create_cpu_model(model_path, inference_dtype="float32"):
    """创建CPU版本的模型，不加载点云编码器"""
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载模型，强制使用CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=getattr(torch, inference_dtype),
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # 禁用点云编码器以节省内存
    if hasattr(model, 'point_backbone'):
        model.point_backbone = None
    
    model.eval()
    return model, tokenizer


def generate_layout_from_features(
    model,
    encoded_features,
    tokenizer,
    code_template_file,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    seed=-1,
    max_new_tokens=4096,
    detect_type="all",
    categories=[],
):
    """使用预编码的特征生成布局"""
    if seed >= 0:
        set_seed(seed)

    # 加载代码模板
    with open(code_template_file, "r") as f:
        code_template = f.read()

    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    print("任务提示: ", task_prompt)

    prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt} The reference code is as followed: {code_template}"

    # 准备对话数据
    conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )

    print(f"编码特征形状: {encoded_features.shape}")
    
    # 创建attention_mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print("正在生成布局...\n")
    
    # 直接生成，不使用streamer避免超时问题
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_clouds=encoded_features,  # 直接传递预编码特征
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # 解码生成的token
        prompt_length = input_ids.shape[1]
        generated_tokens = generated_ids[0, prompt_length:]
        layout_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(layout_str)
        
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return "生成失败"
    
    print("\n完成！")
    return layout_str


def main():
    parser = argparse.ArgumentParser("SpatialLM CPU推理脚本")
    parser.add_argument(
        "-f",
        "--encoded_features",
        type=str,
        required=True,
        help="预编码的点云特征文件路径或包含多个特征文件的文件夹",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出布局txt文件路径或保存多个布局txt文件的文件夹",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
        help="模型路径",
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        type=str,
        default="all",
        choices=["all", "arch", "object"],
        help="要检测的室内元素类型",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        help="要检测的对象类别列表",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default="code_template.txt",
        help="代码模板文件路径",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="top-k过滤的词汇token数量",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="top-p过滤阈值",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="温度参数",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="束搜索的束数量",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="推理使用的数据类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="随机种子",
    )
    
    args = parser.parse_args()

    print("正在加载模型...")
    model, tokenizer = create_cpu_model(args.model_path, args.inference_dtype)
    print("模型加载完成")

    # 确定输入文件
    if os.path.isfile(args.encoded_features):
        feature_files = [args.encoded_features]
    else:
        feature_files = glob.glob(os.path.join(args.encoded_features, "*_encoded.pkl"))

    print(f"找到 {len(feature_files)} 个特征文件")

    for feature_file in tqdm(feature_files, desc="推理中"):
        # 加载预编码特征
        print(f"正在加载特征: {os.path.basename(feature_file)}")
        with open(feature_file, 'rb') as f:
            encoded_data = pickle.load(f)

        encoded_features = encoded_data["features"]
        min_extent = encoded_data["min_extent"]
        num_bins = encoded_data["num_bins"]

        # 生成布局
        layout_str = generate_layout_from_features(
            model,
            encoded_features,
            tokenizer,
            args.code_template_file,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=args.num_beams,
            seed=args.seed,
            detect_type=args.detect_type,
            categories=args.category,
        )

        # 解析布局
        layout = Layout(layout_str)
        layout.undiscretize_and_unnormalize(num_bins=num_bins)
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # 保存结果
        if os.path.splitext(args.output)[-1]:
            output_file = args.output
        else:
            output_filename = os.path.basename(feature_file).replace("_encoded.pkl", ".txt")
            os.makedirs(args.output, exist_ok=True)
            output_file = os.path.join(args.output, output_filename)

        with open(output_file, "w") as f:
            f.write(pred_language_string)

        print(f"结果已保存到: {output_file}")

    print("推理完成！")


if __name__ == "__main__":
    main() 