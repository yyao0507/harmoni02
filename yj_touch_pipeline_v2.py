import numpy as np
#!/usr/bin/env python3
# clean_touch_detection_pipeline.py

""""
Frame t:

1. Detect + track persons
2. Get 2D keypoints
3. Get pose-3D
4. Get depth map

For each pair (A,B):
    ↓
    Hand-region overlap (2D gating)
    ↓
    Compute:
        d_pose
        d_depth
    ↓
    Normalize + fuse
    ↓
    Temporal smoothing (5-frame window)
    ↓
    Final touch decision

export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Depth-Anything-V2
python yj_touch.py \
  --input-dir /root/autodl-tmp/clipped_videos_yj \
  --output-root /root/autodl-tmp/human_behavior_yj \
  --det-config ./yj_configs/rtmdet_m_640-8xb32_coco-person.py \
  --det-checkpoint ./yj_configs/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  --pose3d-config ./yj_configs/rtmw3d-x_8xb32_cocktail14-384x288.py \
  --pose3d-checkpoint ./yj_configs/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth \

"""
#!/usr/bin/env python3
# touch_detection_fused_pipeline.py

import os
import cv2
import json
import numpy as np
import torch
from collections import defaultdict, deque
from argparse import ArgumentParser

from yj_rtmpose3d_v2_func import (
    init_detector, init_pose3d_estimator,
    run_2d_detection_and_tracking,
    run_pose2d_inference,
    refine_pose2d_results
)

# from yj_depth_estimate import DepthAnythingV2, MODEL_CONFIGS, CHECKPOINT_PATH
#!/usr/bin/env python3
# video_depth_analyzer.py

import argparse
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
from scipy.io import savemat
import mmdet.datasets.transforms  # registers PackDetInputs


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
CHECKPOINT_PATH = f'/root/autodl-tmp/Depth-Anything-V2/depth_anything_v2_{ENCODER}.pth'

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

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

COCO_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12), (5, 6), (11, 12)
]


def point_to_segment_distance(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    ap = p - a

    if np.all(ab == 0):
        return np.linalg.norm(ap)

    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)

    proj = a + t * ab
    return np.linalg.norm(p - proj)


def min_hand_to_body_distance(hand_kpts, body_kpts):
    min_dist = float("inf")

    for h_idx in [9, 10]:  # wrists
        hand = hand_kpts[h_idx]
        if np.all(hand == 0):
            continue

        for i, j in COCO_BONES:
            a, b = body_kpts[i], body_kpts[j]
            if np.all(a == 0) or np.all(b == 0):
                continue

            d = point_to_segment_distance(hand, a, b)
            min_dist = min(min_dist, d)

    return min_dist


# -----------------------------------------------------------------------------
# Depth lifting
# -----------------------------------------------------------------------------

def lift_2d_to_3d(kpts2d, depth_map, fx=1000, fy=1000, cx=0, cy=0):
    points = []
    h, w = depth_map.shape

    for x, y in kpts2d[:, :2]:
        x_i = int(np.clip(x, 0, w - 1))
        y_i = int(np.clip(y, 0, h - 1))

        z = depth_map[y_i, x_i]

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        points.append([X, Y, z])

    return np.array(points)


# -----------------------------------------------------------------------------
# 2D gating (hand-specific)
# -----------------------------------------------------------------------------

def get_hand_boxes(kpts2d):
    left = kpts2d[91:112]
    right = kpts2d[112:133]

    def box(pts):
        valid = pts[pts[:, 0] > 0]
        if len(valid) == 0:
            return None
        x1, y1 = np.min(valid[:, :2], axis=0)
        x2, y2 = np.max(valid[:, :2], axis=0)
        return [x1, y1, x2, y2]

    return [box(left), box(right)]


def check_overlap(a, b):
    if a is None or b is None:
        return False
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def hand_body_overlap(person_a, person_b):
    hand_boxes = get_hand_boxes(person_a["kpts2d"])

    for hb in hand_boxes:
        if hb and check_overlap(hb, person_b["bbox_xyxy"]):
            return True
    return False


# -----------------------------------------------------------------------------
# Fusion-based touch detection
# -----------------------------------------------------------------------------

def compute_scale(kpts3d):
    if np.all(kpts3d[5] == 0) or np.all(kpts3d[11] == 0):
        return 1.0
    return np.linalg.norm(kpts3d[5] - kpts3d[11])


def detect_touch_fused(person_a, person_b, depth_map, alpha=0.6):
    kpts3d_a = person_a["kpts3d"]
    kpts3d_b = person_b["kpts3d"]

    kpts2d_a = person_a["kpts2d"]
    kpts2d_b = person_b["kpts2d"]

    # Pose distance
    d_pose = min_hand_to_body_distance(kpts3d_a, kpts3d_b)

    # Depth distance
    k3d_a = lift_2d_to_3d(kpts2d_a, depth_map)
    k3d_b = lift_2d_to_3d(kpts2d_b, depth_map)
    d_depth = min_hand_to_body_distance(k3d_a, k3d_b)

    # Normalize
    scale = compute_scale(kpts3d_b)
    d_pose /= (scale + 1e-6)
    d_depth /= (scale + 1e-6)

    # Fusion
    d_final = alpha * d_pose + (1 - alpha) * d_depth

    return d_final < 0.25, d_final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose3d-config', required=True)
    parser.add_argument('--pose3d-checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bbox-thr', type=float, default=0.5, help='Threshold for detection bbox confidence')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from mmpose.utils import adapt_mmdet_pipeline
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    
    pose_estimator, visualizer = init_pose3d_estimator(
        args.pose3d_config, args.pose3d_checkpoint, device=device
    )
    

    depth_model = DepthAnythingV2(**MODEL_CONFIGS['vitl']).to(device).eval()
    depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))

    touch_memory = defaultdict(lambda: deque(maxlen=5))

    print("Initialization done. Starting video processing...")

    for vid in os.listdir(args.input_dir):
        if not vid.endswith(".mp4"):
            continue
        print(f"Processing video: {vid}")

        cap = cv2.VideoCapture(os.path.join(args.input_dir, vid))
        output = {"frames": []}
        if not cap.isOpened():
            warnings.warn(f"无法打开视频: {vid}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_int = int(round(fps))
        
        if fps_int == 0:
            warnings.warn(f"视频 FPS 异常 (0): {vid}")
            cap.release()
            return

        frame_idx = 0
        sec_stride = 15


        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if frame_idx % int(round(fps)) != 0:
                frame_idx += 1
                continue
           
            if frame_idx % 1000 == 0:
                print(f"Processing frame {frame_idx} of video {vid} …")
            
            # --------


            depth_map = depth_model.infer_image(frame)

            bboxes, tids, *_ = run_2d_detection_and_tracking(
                detector, frame, frame_idx, args,
                {}, {}, {}, {}, {}, 75, 0, []
            )
            
            if len(bboxes) == 0:
                output["frames"].append({
                    "frame": frame_idx,
                    "second": frame_idx / fps,
                    "touch_occurred": False
                })
                frame_idx += 1
                continue
            xywh, pose2d = run_pose2d_inference(pose_estimator, frame, bboxes)
            pose2d = refine_pose2d_results(pose2d)

            persons = []
            print(f"Frame {frame_idx}: Detected {len(pose2d)} persons.")

            for i, res in enumerate(pose2d):
                kpts2d = res.pred_instances.keypoints.reshape(-1, 3)

                kpts3d = kpts2d.copy()
                kpts3d = -kpts3d[..., [0, 2, 1]]

                cx, cy, w, h = xywh[i]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2

                persons.append({
                    "tid": tids[i],
                    "kpts2d": kpts2d,
                    "kpts3d": kpts3d,
                    "bbox_xyxy": [x1, y1, x2, y2]
                })

            frame_touches = []

            for i in range(len(persons)):
                for j in range(len(persons)):
                    if i == j:
                        continue

                    A, B = persons[i], persons[j]
                    pid = (A["tid"], B["tid"])

                    if hand_body_overlap(A, B):
                        is_touch, dist = detect_touch_fused(A, B, depth_map)
                        touch_memory[pid].append(is_touch)

                        history = touch_memory[pid]

                        # hysteresis
                        if sum(history) >= 3:
                            frame_touches.append({
                                "A": A["tid"],
                                "B": B["tid"],
                                "dist": float(dist)
                            })
                    else:
                        touch_memory[pid].append(False)
                    
                    #print(f"Pair {pid}: Overlap={hand_body_overlap(A, B)}, Touch={is_touch}, Dist={dist:.4f}, History={list(history)}")

            # If the frame_touches list has at least one item, a touch happened.
            frame_has_touch = len(frame_touches) > 0

            output["frames"].append({
                "frame": frame_idx,
                "second": frame_idx / fps,
                "touch_occurred": frame_has_touch
            })

            frame_idx += 1

        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_root, vid + ".json"), "w") as f:
            json.dump(output, f, indent=2)
            print(f"saved to {vid}")

    print("Done.")


if __name__ == "__main__":
    main()