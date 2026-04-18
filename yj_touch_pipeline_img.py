#!/usr/bin/env python3
# image_gif_touch_pipeline.py

import os
import cv2
import json
import numpy as np
import torch
import imageio
from collections import defaultdict, deque
from argparse import ArgumentParser
from pathlib import Path

# Assuming these are available in your environment as before
from yj_rtmpose3d_v2_func import (
    init_detector, init_pose3d_estimator,
    run_2d_detection_and_tracking,
    run_pose2d_inference,
    refine_pose2d_results
)
from depth_anything_v2.dpt import DepthAnythingV2

# --- Globals & Helpers from your original script ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_CONFIGS = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
CHECKPOINT_PATH = f'/root/autodl-tmp/Depth-Anything-V2/depth_anything_v2_vitl.pth'

COCO_BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]

def point_to_segment_distance(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab, ap = b - a, p - a
    if np.all(ab == 0): return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))

def min_hand_to_body_distance(hand_kpts, body_kpts):
    min_dist = float("inf")
    for h_idx in [9, 10]:
        if np.all(hand_kpts[h_idx] == 0): continue
        for i, j in COCO_BONES:
            if np.all(body_kpts[i] == 0) or np.all(body_kpts[j] == 0): continue
            min_dist = min(min_dist, point_to_segment_distance(hand_kpts[h_idx], body_kpts[i], body_kpts[j]))
    return min_dist

def lift_2d_to_3d(kpts2d, depth_map, fx=1000, fy=1000, cx=0, cy=0):
    h, w = depth_map.shape
    points = []
    for x, y in kpts2d[:, :2]:
        z = depth_map[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]
        points.append([(x - cx) * z / fx, (y - cy) * z / fy, z])
    return np.array(points)

def get_hand_boxes(kpts2d):
    def box(pts):
        valid = pts[pts[:, 0] > 0]
        if len(valid) == 0: return None
        return [*np.min(valid[:, :2], axis=0), *np.max(valid[:, :2], axis=0)]
    return [box(kpts2d[91:112]), box(kpts2d[112:133])]

def hand_body_overlap(person_a, person_b):
    for hb in get_hand_boxes(person_a["kpts2d"]):
        if hb and not (hb[2] < person_b["bbox_xyxy"][0] or hb[0] > person_b["bbox_xyxy"][2] or hb[3] < person_b["bbox_xyxy"][1] or hb[1] > person_b["bbox_xyxy"][3]):
            return True
    return False

def compute_scale(kpts3d):
    if np.all(kpts3d[5] == 0) or np.all(kpts3d[11] == 0): return 1.0
    return np.linalg.norm(kpts3d[5] - kpts3d[11])

def detect_touch_fused(person_a, person_b, depth_map, alpha=0.6):
    d_pose = min_hand_to_body_distance(person_a["kpts3d"], person_b["kpts3d"])
    d_depth = min_hand_to_body_distance(lift_2d_to_3d(person_a["kpts2d"], depth_map), lift_2d_to_3d(person_b["kpts2d"], depth_map))
    scale = compute_scale(person_b["kpts3d"]) + 1e-6
    d_final = alpha * (d_pose / scale) + (1 - alpha) * (d_depth / scale)
    return d_final < 0.25, d_final

# --- Core Processor ---

def process_frame(frame, frame_idx, detector, pose_estimator, depth_model, args, touch_memory, is_sequence=True):
    """Processes a single BGR image array and returns boolean touch status."""
    
    depth_map = depth_model.infer_image(frame)

    # Note: tracking IDs will be temporary/unstable for static images, but structural logic remains identical.
    bboxes, tids, *_ = run_2d_detection_and_tracking(
        detector, frame, frame_idx, args, {}, {}, {}, {}, {}, 75, 0, []
    )

    xywh, pose2d = run_pose2d_inference(pose_estimator, frame, bboxes)
    pose2d = refine_pose2d_results(pose2d)

    persons = []
    for i, res in enumerate(pose2d):
        kpts2d = res.pred_instances.keypoints.reshape(-1, 3)
        kpts3d = -kpts2d.copy()[..., [0, 2, 1]] # Simple heuristic flip from your code
        cx, cy, w, h = xywh[i]
        persons.append({
            "tid": i, # Overriding tracker ID for isolated frames
            "kpts2d": kpts2d, "kpts3d": kpts3d,
            "bbox_xyxy": [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        })

    frame_touch_occurred = False

    for i in range(len(persons)):
        for j in range(len(persons)):
            if i == j: continue
            A, B = persons[i], persons[j]
            pid = (A["tid"], B["tid"])

            if hand_body_overlap(A, B):
                is_touch, dist = detect_touch_fused(A, B, depth_map)
                
                if is_sequence:
                    touch_memory[pid].append(is_touch)
                    # Use hysteresis for GIFs
                    if sum(touch_memory[pid]) >= 3:
                        frame_touch_occurred = True
                else:
                    # Instant decision for single images
                    if is_touch:
                        frame_touch_occurred = True
            else:
                if is_sequence:
                    touch_memory[pid].append(False)

    return frame_touch_occurred

def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', required=True, help="Path to a .jpg, .png, or .gif")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--pose3d-config', required=True)
    parser.add_argument('--pose3d-checkpoint', required=True)
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()

    # Model Initialization
    from mmpose.utils import adapt_mmdet_pipeline
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator, _ = init_pose3d_estimator(args.pose3d_config, args.pose3d_checkpoint, device=args.device)
    depth_model = DepthAnythingV2(**MODEL_CONFIGS['vitl']).to(args.device).eval()
    depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))

    input_path = Path(args.input_path)
    ext = input_path.suffix.lower()
    
    output_data = {"filename": input_path.name, "frames": []}
    touch_memory = defaultdict(lambda: deque(maxlen=5))

    print(f"Processing: {input_path.name}")

    if ext == '.gif':
        # Process GIF frame by frame
        gif_reader = imageio.get_reader(input_path)
        for frame_idx, frame_rgb in enumerate(gif_reader):
            # Convert RGB (imageio default) to BGR (cv2/model default)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            touch_status = process_frame(
                frame_bgr, frame_idx, detector, pose_estimator, depth_model, 
                args, touch_memory, is_sequence=True
            )
            
            output_data["frames"].append({
                "frame": frame_idx,
                "touch_occurred": touch_status
            })
            print(f"Frame {frame_idx} Touch: {touch_status}")
            
    elif ext in ['.jpg', '.jpeg', '.png']:
        # Process single static image
        frame_bgr = cv2.imread(str(input_path))
        if frame_bgr is None:
            raise ValueError(f"Could not read image: {input_path}")
            
        touch_status = process_frame(
            frame_bgr, 0, detector, pose_estimator, depth_model, 
            args, touch_memory, is_sequence=False
        )
        
        output_data["frames"].append({
            "frame": 0,
            "touch_occurred": touch_status
        })
        print(f"Image Touch Detected: {touch_status}")
        
    else:
        raise ValueError("Unsupported file type. Please provide .gif, .jpg, or .png")

    # Save Output
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output_dir) / f"{input_path.stem}.json"
    
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Done. Output saved to {out_file}")

if __name__ == "__main__":
    main()