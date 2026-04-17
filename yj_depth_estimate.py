#!/usr/bin/env python3
# video_depth_analyzer.py

import argparse
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
from scipy.io import savemat

# Depth-Anything-V2 导入
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("请确保脚本位于 Depth-Anything-V2 项目根目录下，或已正确安装依赖。")

# --- 全局配置 ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
ENCODER = 'vitl'
CHECKPOINT_PATH = f'depth_anything_v2_{ENCODER}.pth'

def load_model():
    """加载深度估计模型"""
    print(f"Loading model: {ENCODER} to {DEVICE}...")
    model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
    
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"找不到权重文件: {CHECKPOINT_PATH}")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    print("Model loaded successfully.")
    return model

def calculate_window_variance(data_array, window_size):
    """计算滑动窗口方差 (非重叠窗口)"""
    n_windows = int(np.ceil(len(data_array) / window_size))
    variances = []
    for i in range(n_windows):
        seg = data_array[i*window_size : min((i+1)*window_size, len(data_array))]
        # 如果片段非空则计算方差，否则填0.0
        variances.append(float(np.nanvar(seg)) if seg.size > 0 else 0.0)
    return np.array(variances, dtype=float)

def process_single_video(video_path, output_dir, model):
    """处理单个视频：推理 -> 内存计算 -> 保存MAT"""
    video_path = Path(video_path)
    # 定义输出文件名
    mat_path = output_dir / f"{video_path.stem}.mat"
    
    # 如果已存在，可选择跳过（根据需求注释掉下面两行）
    if mat_path.exists():
        print(f"[Skip] {mat_path.name} already exists.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(round(fps))
    
    if fps_int == 0:
        warnings.warn(f"视频 FPS 异常 (0): {video_path}")
        cap.release()
        return

    frame_id = 0
    avg_depths_list = [] # 存储每秒的平均深度值

    print(f"Processing: {video_path.name} (FPS: {fps_int})")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 逻辑：只处理每秒的第 0 帧
        if frame_id % fps_int == 0:
            with torch.no_grad():
                # 1. 深度推理
                depth_map = model.infer_image(frame_bgr) # HxW numpy array
                
                # 2. 立即计算平均值 (Analysis)
                # 这里的 nanmean 对应之前的 compute_depth_metrics 逻辑
                current_avg = float(np.nanmean(depth_map))
                avg_depths_list.append(current_avg)
                
                # 注意：这里不保存 depth_map，直接循环进入下一帧，内存不累积

        frame_id += 1

    cap.release()

    # --- 后处理与保存 ---
    if not avg_depths_list:
        warnings.warn(f"未提取到数据: {video_path.name}")
        return

    # 转换为 numpy 数组
    avg_depths_arr = np.array(avg_depths_list, dtype=float)

    # 计算方差指标 (15s 和 60s 窗口)
    var15 = calculate_window_variance(avg_depths_arr, 15)
    var60 = calculate_window_variance(avg_depths_arr, 60)

    # 准备输出数据
    out_dict = {
        'avg_depth_1s':  avg_depths_arr,
        'var_depth_15s': var15,
        'var_depth_60s': var60
    }

    # 保存 .mat
    output_dir.mkdir(parents=True, exist_ok=True)
    savemat(str(mat_path), out_dict)
    
    print(f"[Done] Saved {mat_path.name}: {len(avg_depths_arr)}s data")

def main():
    parser = argparse.ArgumentParser(description="Directly extract depth metrics from video without saving intermediate files.")
    parser.add_argument('--input_dir', required=True, type=str, help="输入视频目录")
    parser.add_argument('--output_dir', required=True, type=str, help="输出 .mat 文件目录")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 1. 加载模型 (只加载一次)
    model = load_model()

    # 2. 遍历视频
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [p for p in input_dir.iterdir() if p.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting processing...")

    for i, video_file in enumerate(sorted(video_files)):
        print(f"[{i+1}/{len(video_files)}] ", end="")
        try:
            process_single_video(video_file, output_dir, model)
        except Exception as e:
            print(f"\n[Error] Failed to process {video_file.name}: {e}")

if __name__ == "__main__":
    main()