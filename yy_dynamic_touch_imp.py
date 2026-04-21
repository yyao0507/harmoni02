#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
import torch
import warnings
from pathlib import Path
from collections import defaultdict, deque
from argparse import ArgumentParser


from yj_rtmpose3d_v2_func import (
    init_detector, init_pose3d_estimator,
    run_2d_detection_and_tracking,
    run_pose2d_inference,
    refine_pose2d_results
)


try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("请确保安装了 Depth-Anything-V2 并在正确路径下运行。")


class HybridTouchDetector:
    def __init__(self, depth_tolerance=12.0, touch_dist_thresh=0.25, memory_size=5, alpha=0.6):

        self.depth_tolerance = depth_tolerance
        self.touch_dist_thresh = touch_dist_thresh
        self.alpha = alpha
        self.touch_memory = defaultdict(lambda: deque(maxlen=memory_size))
        
        self.WRIST_IDXS = [9, 10] 
        self.BODY_BONES = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 11), (6, 12), (5, 6), (11, 12)
        ]

    def compute_robust_scale(self, kpts3d):
        shoulder_dist = np.linalg.norm(kpts3d[5, :3] - kpts3d[6, :3])
        torso_dist = np.linalg.norm(kpts3d[5, :3] - kpts3d[11, :3])
        scale = max(shoulder_dist, torso_dist)
        return scale if scale > 0.1 else 1.0

    def verify_depth_consistency(self, depth_map, hand_pt_2d, patch_size=5):
        if depth_map is None or hand_pt_2d is None:
            return False
            
        h, w = depth_map.shape
        x, y = int(hand_pt_2d[0]), int(hand_pt_2d[1])
        
 
        x1, x2 = max(0, x - patch_size), min(w, x + patch_size)
        y1, y2 = max(0, y - patch_size), min(h, y + patch_size)
        
        depth_patch = depth_map[y1:y2, x1:x2]
        if depth_patch.size == 0:
            return False
            
        depth_std = np.std(depth_patch)
      
        return depth_std < self.depth_tolerance

    def _point_to_segment_distance(self, p, a, b):
       
        ab, ap = b - a, p - a
        if np.all(ab == 0): return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    def min_hand_to_body_distance(self, hand_kpts, body_kpts, hand_confs, depth_map=None, original_kpts2d=None):
        min_dist = float("inf")
        is_depth_consistent = False
        best_hand_2d = None

        for h_idx in self.WRIST_IDXS:
    
            if hand_confs[h_idx] < 0.3 or np.all(hand_kpts[h_idx, :3] == 0):
                continue
                
            hand_pt = hand_kpts[h_idx, :3]

            for i, j in self.BODY_BONES:
                if np.all(body_kpts[i, :3] == 0) or np.all(body_kpts[j, :3] == 0):
                    continue

                a, b = body_kpts[i, :3], body_kpts[j, :3]
                d = self._point_to_segment_distance(hand_pt, a, b)
                
                if d < min_dist:
                    min_dist = d
                    if original_kpts2d is not None:
                        best_hand_2d = original_kpts2d[h_idx, :2] 

       
        if min_dist < float("inf") and best_hand_2d is not None and depth_map is not None:
            is_depth_consistent = self.verify_depth_consistency(depth_map, best_hand_2d)
        else:
            is_depth_consistent = True 

        return min_dist, is_depth_consistent

    def _lift_2d_to_3d(self, kpts2d, depth_map, fx=1000, fy=1000, cx=None, cy=None):
  
        h, w = depth_map.shape
        if cx is None: cx = w / 2
        if cy is None: cy = h / 2
        
        points_3d = np.zeros((kpts2d.shape[0], 3))
        for idx, (x, y) in enumerate(kpts2d[:, :2]):
            if x == 0 and y == 0:
                continue
            x_i = int(np.clip(x, 0, w - 1))
            y_i = int(np.clip(y, 0, h - 1))
            z = depth_map[y_i, x_i]
            
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3d[idx] = [X, Y, z]
            
        return points_3d

    def _fused_distance(self, actor, receiver, depth_map):
      
        actor_kpts2d, actor_confs = actor["kpts2d"][:, :2], actor["kpts2d"][:, 2]
        receiver_kpts3d = receiver["kpts3d"]
        
    
        d_pose, _ = self.min_hand_to_body_distance(actor["kpts3d"], receiver_kpts3d, actor_confs)
        
        actor_kpts3d_lifted = self._lift_2d_to_3d(actor_kpts2d, depth_map)
        receiver_kpts3d_lifted = self._lift_2d_to_3d(receiver["kpts2d"][:, :2], depth_map)
        
        d_depth, is_consistent = self.min_hand_to_body_distance(
            actor_kpts3d_lifted, receiver_kpts3d_lifted, actor_confs, depth_map, original_kpts2d=actor_kpts2d
        )
        
      
        scale = self.compute_robust_scale(receiver_kpts3d)
        d_pose /= (scale + 1e-6)
        d_depth /= (scale + 1e-6)
        

        fused_d = self.alpha * d_pose + (1 - self.alpha) * d_depth
        return fused_d, is_consistent

    def evaluate_pair(self, person_A, person_B, depth_map):  
        dist_A_to_B, consistent_A = self._fused_distance(person_A, person_B, depth_map)
        dist_B_to_A, consistent_B = self._fused_distance(person_B, person_A, depth_map)
        
        min_dist = min(dist_A_to_B, dist_B_to_A)
        
        is_touching_now = (min_dist < self.touch_dist_thresh) and (consistent_A or consistent_B)
        return is_touching_now, min_dist

    def update_and_get_decision(self, pid_tuple, is_touching_now):
  
        history = self.touch_memory[pid_tuple]
        history.append(is_touching_now)
      
        return sum(history) >= 3

def get_hand_boxes(kpts2d):
    if kpts2d.shape[0] > 90:
        left = kpts2d[91:112]
        right = kpts2d[112:133]
    else:
        left = np.array([[kpts2d[9, 0]-20, kpts2d[9, 1]-20], [kpts2d[9, 0]+20, kpts2d[9, 1]+20]])
        right = np.array([[kpts2d[10, 0]-20, kpts2d[10, 1]-20], [kpts2d[10, 0]+20, kpts2d[10, 1]+20]])

    def box(pts):
        valid = pts[pts[:, 0] > 0]
        if len(valid) == 0: return None
        x1, y1 = np.min(valid[:, :2], axis=0)
        x2, y2 = np.max(valid[:, :2], axis=0)
        return [x1, y1, x2, y2]

    return [box(left), box(right)]

def check_overlap(a, b):

    if a is None or b is None: return False
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

def hand_body_overlap(person_a, person_b):
    hand_boxes = get_hand_boxes(person_a["kpts2d"])
    for hb in hand_boxes:
        if hb and check_overlap(hb, person_b["bbox_xyxy"]):
            return True
    return False


def parse_args():
    parser = ArgumentParser(description="Dynamic Touch Detection Pipeline")
    parser.add_argument('--input-dir', required=True, help="Input videos directory")
    parser.add_argument('--output-root', required=True, help="Output JSON directory")
    parser.add_argument('--det-config', required=True, help="MMDet config path")
    parser.add_argument('--det-checkpoint', required=True, help="MMDet weights path")
    parser.add_argument('--pose3d-config', required=True, help="Pose3D config path")
    parser.add_argument('--pose3d-checkpoint', required=True, help="Pose3D weights path")
    parser.add_argument('--depth-checkpoint', required=True, help="DepthAnything weights path")
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=> 正在初始化人体检测与 3D Pose 模型...")
    from mmpose.utils import adapt_mmdet_pipeline
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator, _ = init_pose3d_estimator(args.pose3d_config, args.pose3d_checkpoint, device=device)

    print("=> 正在初始化 Depth-Anything-V2...")
    depth_model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    depth_model = DepthAnythingV2(**depth_model_config).to(device).eval()
    depth_model.load_state_dict(torch.load(args.depth_checkpoint, map_location='cpu'))

    touch_detector = HybridTouchDetector(depth_tolerance=12.0, touch_dist_thresh=0.25, memory_size=5, alpha=0.6)

    print("=> 初始化完成。开始处理视频...")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for vid in os.listdir(args.input_dir):
        if not vid.endswith((".mp4", ".avi", ".mov")):
            continue
            
        print(f"正在处理视频: {vid}")
        video_path = os.path.join(args.input_dir, vid)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            warnings.warn(f"无法打开视频: {vid}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_int = int(round(fps)) if fps > 0 else 30
        
        output_data = {"video": vid, "fps": fps, "frames": []}
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            if frame_idx % 100 == 0:
                print(f"  处理到第 {frame_idx} 帧...")

            with torch.no_grad():
                depth_map = depth_model.infer_image(frame)

            bboxes, tids, *_ = run_2d_detection_and_tracking(
                detector, frame, frame_idx, args,
                {}, {}, {}, {}, {}, 75, 0, []
            )

            if len(bboxes) == 0:
                output_data["frames"].append({
                    "frame": frame_idx, "second": frame_idx / fps, "touch_occurred": False, "events": []
                })
                frame_idx += 1
                continue

            xywh, pose2d = run_pose2d_inference(pose_estimator, frame, bboxes)
            pose2d = refine_pose2d_results(pose2d)

            persons = []
            for i, res in enumerate(pose2d):
                kpts2d = res.pred_instances.keypoints.reshape(-1, 3) 
                kpts3d = kpts2d.copy()
                kpts3d = -kpts3d[..., [0, 2, 1]]

                cx, cy, w, h = xywh[i]
                persons.append({
                    "tid": tids[i],
                    "kpts2d": kpts2d,
                    "kpts3d": kpts3d,
                    "bbox_xyxy": [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                })

            frame_touches = []

            for i in range(len(persons)):
                for j in range(len(persons)):
                    if i == j: continue

                    A, B = persons[i], persons[j]
                    pid_tuple = (A["tid"], B["tid"])

                    if hand_body_overlap(A, B):
                        is_touching_now, dist = touch_detector.evaluate_pair(A, B, depth_map)
                    else:
                        is_touching_now, dist = False, float('inf')
                        
                    final_decision = touch_detector.update_and_get_decision(pid_tuple, is_touching_now)

                    if final_decision:
                        frame_touches.append({
                            "actor_id": A["tid"],
                            "receiver_id": B["tid"],
                            "fused_distance": float(dist)
                        })

            output_data["frames"].append({
                "frame": frame_idx,
                "second": frame_idx / fps,
                "touch_occurred": len(frame_touches) > 0,
                "events": frame_touches
            })

            frame_idx += 1

        cap.release()
        

        out_json_path = os.path.join(args.output_root, vid + ".json")
        with open(out_json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"=> 视频 {vid} 处理完成，结果已保存至: {out_json_path}")

    print("所有视频处理完毕。")

if __name__ == "__main__":
    main()
