#!/usr/bin/env python3
# custom_rtmpose3d_analysis_clean_v2_yj.py
import os
import pandas as pd
import cv2
import mmcv
import copy
import torch
import json
import mimetypes
import collections
import numpy as np
from argparse import ArgumentParser

from yj_depth_estimate import DepthAnythingV2, MODEL_CONFIGS, CHECKPOINT_PATH
from yj_rtmpose3d_v2_func import *


def cal_child_parent_score(kpts3d, frame, x, y, w_box, h_box, classifier, cls_transform, device, skeleton_model):
    head_pt = kpts3d[0]
    shoulder_mid = (kpts3d[5] + kpts3d[6]) / 2.0
    head_height = float(np.linalg.norm(head_pt - shoulder_mid))
    hip_mid = (kpts3d[11] + kpts3d[12]) / 2.0
    body_height = float(np.linalg.norm(shoulder_mid - hip_mid)) + 1e-6
    head_body_ratio = head_height / body_height
    hb_min, hb_max = 0.15, 0.65
    norm_ratio_hb = np.clip((head_body_ratio - hb_min) / (hb_max - hb_min), 0, 1)
    
    # 2) 计算头肩宽比（耳朵间距 / 肩宽）
    # COCOWholeBody: 3=左耳, 4=右耳, 5=左肩, 6=右肩
    ear_l, ear_r = kpts3d[3], kpts3d[4]
    head_width = float(np.linalg.norm(ear_l - ear_r)) + 1e-6
    shoulder_width = float(np.linalg.norm(kpts3d[5] - kpts3d[6])) + 1e-6
    head_shoulder_ratio = head_width / shoulder_width
    hs_min, hs_max = 0.05, 0.75 # 可根据分布微调
    norm_ratio_hw = np.clip((head_shoulder_ratio - hs_min) / (hs_max - hs_min), 0, 1)
    
    # 3) 二分类网络预测
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w_box)
    y2 = min(frame.shape[0], y + h_box)
    fullbody_crop = frame[y1:y2, x1:x2]
    cls_label, p_child, p_adult = predict_child_adult(
        classifier, fullbody_crop, cls_transform, device
    )
    

    # 4) 四分量加权融合
    p_child_sk, p_adult_sk = predict_child_skeleton(
        kpts3d[:17], skeleton_model, device
    )
    print()
    # 三路融合：head/body, head/shoulder, image, skeleton
    # w_hb, w_hw, w_cls, w_sk = 0.03, 0.07, 0.00, 0.90
    # 建议先调整为这个比例进行观察
    w_hb, w_hw, w_cls, w_sk = 0.15, 0.15, 0.40, 0.30
    score_child = (w_hb * norm_ratio_hb
                 + w_hw * norm_ratio_hw
                 + w_cls * p_child
                 + w_sk * p_child_sk)
    score_child = np.clip(score_child, 0.0, 1.0)
    score_adult = 1.0 - score_child
    return score_child, score_adult



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='Input directory containing videos'
    )
    parser.add_argument(
        '--output-root', type=str, required=True,
        help='Root directory for outputs (will create b_data/ and p_video/)'
    )
    parser.add_argument(
        '--det-config', required=True,
        help='MMDetection config file for 2D person detector'
    )
    parser.add_argument(
        '--det-checkpoint', required=True,
        help='MMDetection checkpoint file for 2D detector'
    )
    parser.add_argument(
        '--pose3d-config', required=True,
        help='MMPose 3D pose estimator config file'
    )
    parser.add_argument(
        '--pose3d-checkpoint', required=True,
        help='MMPose 3D pose estimator checkpoint file'
    )
    parser.add_argument(
        '--cls-checkpoint', type=str, required=True,
        help='Checkpoint file for the child/adult classifier'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device used for inference (e.g. "cuda:0" or "cpu")'
    )
    parser.add_argument(
        '--bbox-thr', type=float, default=0.5,
        help='Threshold for detection bbox confidence'
    )
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3,
        help='Keypoint visibility threshold (unused here)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Whether to show per-frame visualization in a window'
    )
    parser.add_argument(
        '--num-instances', type=int, default=-1,
        help='Max number of 3D poses to visualize per frame (use -1 for all)'
    )
    parser.add_argument('--sharingan-ckpt', required=True, help='Sharingan gaze 模型权重')
    parser.add_argument(
        '--skeleton-checkpoint', type=str, required=True,
        help='Checkpoint file for the skeleton-based child/adult classifier')
    parser.add_argument(
        '--max-videos', type=int, default=-1,
        help='Only process first N videos for debugging; use -1 to process all'
    )
    parser.add_argument(
        '--save-kp-debug', action='store_true',
        help='Save per-person debug images when kp_points are computed'
    )
    parser.add_argument(
        '--kp-debug-max-per-video', type=int, default=40,
        help='Max number of kp debug images to save per video'
    )
    return parser.parse_args()



import warnings
warnings.filterwarnings("ignore", message=".*dist attribute.*")


def add_missing_record(missing_info_records, video_id, frame_id, track_id, missing_field, reason):
    missing_info_records.append({
        "video_id": str(video_id),
        "frame_id": int(frame_id),
        "track_id": int(track_id),
        "missing_field": str(missing_field),
        "reason": str(reason)
    })
    return missing_info_records

def organize_missing_info(records):
    """Group missing info by video_id first, then by missing_field type."""
    organized = {}
    for rec in records:
        video_id = rec.get("video_id", "unknown_video")
        missing_type = rec.get("missing_field", "unknown_type")
        if video_id not in organized:
            organized[video_id] = {}
        if missing_type not in organized[video_id]:
            organized[video_id][missing_type] = []
        organized[video_id][missing_type].append({
            "frame_id": rec.get("frame_id", -1),
            "track_id": rec.get("track_id", -1),
            "reason": rec.get("reason", "")
        })
    return organized



def main():
    args = parse_args()

    # 把一下初始化的部分单独写进一个函数

    # ------------------------------------------------
    # 1. 创建输出目录（b_data/ 和 p_video/）
    # ------------------------------------------------
    os.makedirs(args.output_root, exist_ok=True)
    out_json_dir = os.path.join(args.output_root, 'b_data')
    # out_video_dir = os.path.join(args.output_root, 'p_video')
    out_debug_dir = os.path.join(args.output_root, 'debug')
    os.makedirs(out_json_dir, exist_ok=True)
    # os.makedirs(out_video_dir, exist_ok=True)
    os.makedirs(out_debug_dir, exist_ok=True)
    
    # ------------------------------------------------
    # 2. 初始化 2D 检测器 (MMDetection)
    # ------------------------------------------------
    device = args.device
    from mmpose.utils import adapt_mmdet_pipeline
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=device
    )
    # 把 MMDetection pipeline 转为 MMPose 可识别格式
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)


    depth_model = DepthAnythingV2(**MODEL_CONFIGS['vitl'])
    # 加载模型时使用这个绝对路径
    depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    print("Depth-Anything-V2 loaded successfully.")

    # ------------------------------------------------
    # 3. 初始化 3D 姿态估计器 (MMPose)
    # ------------------------------------------------
    pose_estimator, visualizer = init_pose3d_estimator(
        args.pose3d_config,
        args.pose3d_checkpoint,
        device
    )
    #print(pose_estimator.cfg.test_dataloader.dataset)
   

    # ------------------------------------------------
    # 4. 初始化 Child/Adult 二分类模型 (ResNet50)
    # ------------------------------------------------
    classifier = build_binary_resnet50(num_classes=2).to(device)
    classifier.load_state_dict(
        torch.load(args.cls_checkpoint, map_location=device)
    )
    classifier.eval()
    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
   
    # ─── Skeleton-based 分类模型（Multi-Head GAT + LayerNorm + Residual GCN + EMA） ────
    A = get_adjacency_matrix().to(device)
    skeleton_model = SingleFrameSTGCN(in_c=3, num_class=2, A=A).to(device)
    skeleton_model.load_state_dict(
        torch.load(args.skeleton_checkpoint, map_location=device)
    )
    skeleton_model.eval()
   
    # ------------------------------
    # 初始化 Gaze 模型
    # ------------------------------
    gaze_sharingan = load_gaze_models(args.sharingan_ckpt, device)

    # 全局跟踪与分类缓存
    track_kf = {} # { tid: KalmanFilter 实例 }
    track_last_seen = {} # { tid: 最近出现的帧号 }
    track_age = {} # { tid: 已存活帧计数 }
    #max_age = 30 # 关键帧重连最大间隔
    max_age = 75 # 关键帧重连最大间隔
    score_hist = {} # { tid: deque([最近若干帧的 child_score], maxlen=10) }
    label_lock = {} # { tid: "child"/"adult" }
    stable_count = {} # { tid: 连续复查与当前标签一致的次数 }
    hard_lock = {} # { tid: 是否已经彻底锁定，不再允许修改 }
    next_tid = 0
    track_colors = {} # tid -> (B,G,R)
    
    # ------------------------------------------------
    # 5. 遍历输入目录下所有视频
    # ------------------------------------------------
    video_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])
    if args.max_videos > 0:
        video_files = video_files[:args.max_videos]
        print(f"Debug mode: only processing first {len(video_files)} videos.")
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return



    missing_info_records = []

    for video_name in video_files:
        
        video_path = os.path.join(args.input_dir, video_name)
        video_basename = os.path.splitext(video_name)[0]
        # if video_basename not in ["81", "88", "6", "89", "100", "95"]:
        #     continue

        out_json_path = os.path.join(
            out_json_dir, f'{video_basename}_labels.json'
        )
        # out_video_path = os.path.join(
        #     out_video_dir, f'{video_basename}_vis.mp4'
        # )

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sec_stride = max(1, int(round(fps)))

        max_seconds = 5 * 60
        #max_seconds = 20
        max_frame_idx = int(round(fps * max_seconds))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and total_frames < max_frame_idx:
            duration_sec = total_frames / max(float(fps), 1e-6)
            missing_info_records = add_missing_record(
                missing_info_records,
                video_basename,
                -1,
                -1,
                "video_duration",
                f"video shorter than max_seconds ({duration_sec:.2f}s < {max_seconds}s)"
            )

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # #writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width+height, height))
        frame_idx = 0
        fallback_second = None
        kp_debug_saved = 0
       
        # 每个视频重新初始化全局跟踪与分类缓存
        track_kf.clear()
        track_last_seen.clear()
        track_age.clear()
        score_hist.clear()
        label_lock.clear()
        stable_count.clear()
        hard_lock.clear()
        next_tid = 0

        # 新增：定义 track_buffer，用于存储每个 tid 每一帧的 score_child
        track_buffer = collections.defaultdict(list)
       
        # 初始化 JSON 结构，按帧追加
        json_dict = {
            "video_name": video_name,
            "fps:": fps,
            "frames": []
        }
       

        while cap.isOpened():
            # add error info if total frame less than max_frame_idx
            if frame_idx >= max_frame_idx: # part after 5 minutes will not processed
                break
            success, frame = cap.read()
            if not success:
                break

            current_second = frame_idx // sec_stride
            is_primary_frame = (frame_idx % sec_stride == 0)
            if is_primary_frame:
                # Start with the first frame in the second; if output is poor, keep trying this second.
                fallback_second = current_second
            should_process = is_primary_frame or (fallback_second == current_second)

            if not should_process:
                frame_idx += 1
                continue
           
            if frame_idx % 1000 == 0:
                print(f"Processing frame {frame_idx} of video {video_basename} …")
            
            # ----------------------------------------
            # 5.1 2D 检测 (inference_detector)

            # ----------------------------------------
            (
                raw_bboxes, # 当前帧通过人检测阈值后的原始框，形状大致是 N x 4（xyxy）
                associated_tids, # 长度为 N 的列表，表示 raw_bboxes 中每个框对应的 track ID（tid）。这个是 5.1 主要输出，用于后续关联和跟踪。
                next_tid, # 2.1.3 中更新的下一个可用 track ID（tid）。如果当前帧有新检测到的人，next_tid 会增加。
                missing_info_records,
                should_continue_frame,
                no_person_detected,
            ) = run_2d_detection_and_tracking(
                detector=detector,
                frame=frame,
                frame_idx=frame_idx,
                args=args,
                track_kf=track_kf,
                track_last_seen=track_last_seen,
                track_age=track_age,
                score_hist=score_hist,
                label_lock=label_lock,
                max_age=max_age,
                next_tid=next_tid,
                missing_info_records=missing_info_records,
            )

            #print(f"\nFrame {frame_idx}: label_lock {label_lock}")
            if should_continue_frame:
                #print(f"Frame {frame_idx}: No detections. {'No person detected in this frame.' if no_person_detected else 'Attempting keyframe reconnection...'}")
                # 仅当这一秒最后一帧仍失败，才记录 missing_info 并导出该帧图像。
                is_last_frame_of_second = ((frame_idx + 1) % sec_stride == 0)
                if no_person_detected and is_last_frame_of_second:
                    miss_img_path = os.path.join(
                        out_debug_dir,
                        f"{video_basename}_missing_person_f{frame_idx}.jpg"
                    )
                    cv2.imwrite(miss_img_path, frame)
                    missing_info_records = add_missing_record(
                        missing_info_records,
                        video_basename,
                        frame_idx,
                        -1,
                        "persons",
                        f"no person detected in all attempted frames for second {current_second}; saved frame: {miss_img_path}"
                    )
                frame_idx += 1
                continue
            
            if raw_bboxes.shape[0] != 2:
                missing_info_records = add_missing_record(
                    missing_info_records,
                    video_basename,
                    frame_idx,
                    -1,
                    "persons",
                    f"Expected 2 detections, but got {raw_bboxes.shape[0]} after tracking association."
                )
            # ----------------------------------------
            # 5.2 2D→3D 姿态推理 (inference_topdown)
            # 注：去掉不被支持的 return_heatmap 等参数
            # 主要输出是 pose2d_results，包含每个人的 2D 关键点信息，后续会用来调试头部框计算和作为 3D 推理的输入。
            # xywh_bboxes 是根据 raw_bboxes 转换得到的中心点坐标和宽高
            # ----------------------------------------
            (
                xywh_bboxes,
                pose2d_results,
            ) = run_pose2d_inference(
                pose_estimator=pose_estimator,
                frame=frame,
                raw_bboxes=raw_bboxes
            )

            # print(f'Frame {frame_idx}, raw_bboxes.shape = {raw_bboxes.shape}, associated_tids = {associated_tids}')
            # print(f'pose2d_results length = {len(pose2d_results)}')


            refined_pose2d_results = refine_pose2d_results(pose2d_results)
            

            openpose_points_from_kpts = {}
            j25_2d_from_kpts = {}
            # --- 调试开始 ---
            # print(f"\n>>> Frame {frame_idx} Analysis:")
            # print(f"Detected instances: {len(refined_pose2d_results)}")
            
            dbg_img = frame.copy() # 每一帧初始化底图
            head_bbox_wholebody = []
            
            for idx, res in enumerate(refined_pose2d_results):
                # 确保 tid 存在
                try:
                    tid = int(associated_tids[idx])
                except IndexError:
                    print(f"Warning: No TID for instance {idx}")
                    continue

                raw_kpts = np.asarray(res.pred_instances.keypoints)
                if raw_kpts.ndim == 3: raw_kpts = raw_kpts[0]
                

                # 重新计算头部
                # head_bbox = get_head_bbox_from_res(res)
                head_box_wholebody = get_head_bbox_wholebody(res, raw_bboxes[idx])
                
                if head_box_wholebody is not None and np.asarray(head_box_wholebody).shape == (4,):
                    head_bbox_wholebody.append(np.asarray(head_box_wholebody, dtype=float))
                    hx1, hy1, hx2, hy2 = np.asarray(head_box_wholebody).astype(int)
                    # # print(f"TID {tid} - Head BBox: [{hx1}, {hy1}, {hx2}, {hy2}]")
                    
                    # cv2.rectangle(dbg_img, (hx1, hy1), (hx2, hy2), (0, 255, 255), 3)
                    # cv2.putText(dbg_img, f"ID:{tid}", (hx1, hy1-10), 0, 0.7, (0, 255, 255), 2)
                    
                    # # 画出所有点，不设阈值过滤，看看到底在哪
                    # for kx, ky, kc in raw_kpts:
                    #     cv2.circle(dbg_img, (int(kx), int(ky)), 2, (0, 0, 255), -1)

                else:
                    head_bbox_wholebody.append(None)
                    #print(f"TID {tid} - Head BBox was NONE")
                    missing_info_records = add_missing_record(
                        missing_info_records,
                        video_basename,
                        frame_idx,
                        tid,
                        "head_bbox",
                        "Failed to compute head bbox from keypoints"
                    )
            # out_path = os.path.join(out_debug_dir, f"force_debug_f{frame_idx}.jpg")
            # cv2.imwrite(out_path, dbg_img)
            # print(f"Saved debug image to: {out_path}")

            # ----------------------------------------
            # 5.3 method 1  后处理 3D 关键点：坐标还原 -> reshape -> 坐标变换 -> rebase
            # ----------------------------------------
            threed_refined_pose2d_results = copy.deepcopy(refined_pose2d_results) # 深复制，避免修改原始数据
            for idx, res in enumerate(threed_refined_pose2d_results):
                # 4. 这里的 tid 处理
                tid = associated_tids[idx]
                res.track_id = tid
                kpts = res.pred_instances.keypoints
                kpts = kpts.reshape(-1, 3)
                transformed_3d = kpts.copy()
                # transformed_3d = -transformed_3d[..., [0, 2, 1]] 
                transformed_3d = -transformed_3d[..., [0, 2, 1]]
                # 把x轴取反
                transformed_3d[..., 0] = -transformed_3d[..., 0]
                
                # 6. 让最低点落地 (z 轴 rebase)
                # 在 (-x, z, y) 坐标系下，最后一维 [2] 是原本的 y，现在代表高度/深度方向
                # 如果你是想让脚部接触地面，通常是对深度轴或高度轴做偏移
                transformed_3d[..., 2] -= np.min(transformed_3d[..., 2], axis=-1, keepdims=True)

                # 7. 写回 DataSample，扩展为 (1, num_joints, 3)
                res.pred_instances.keypoints = transformed_3d[np.newaxis, ...]
            
                
            # --- 新增：保存 3D 空间预览图 ---
            # viz_3d_path = os.path.join(out_debug_dir, f"kinect_3D_space_f{frame_idx}.png")
            # save_3d_pose_kinect_style_color(threed_refined_pose2d_results, associated_tids, viz_3d_path)

            # 合并结果
            merged = merge_data_samples(threed_refined_pose2d_results)
            instances_3d = merged.get('pred_instances', None)
        

            # ----------------------------------------
            # 5.4 Child/Adult 分类 & 标签锁定
            # ----------------------------------------
            tmp_info = [] # [(tid, bbox, kpts3d, label, score_child, score_adult, plabel), ...]


            for idx, res in enumerate(threed_refined_pose2d_results):
                corresponding_2d = refined_pose2d_results[idx].pred_instances.keypoints
                head_box = head_bbox_wholebody[idx] if idx < len(head_bbox_wholebody) else None
                #print(f"\nProcessing instance {idx} with TID {associated_tids[idx]} for Child/Adult classification, get corresponding_2d:{corresponding_2d}")
                tid = associated_tids[idx] # 取卡尔曼匹配出来的 id
                res.track_id = tid
                kpts3d = instances_3d.keypoints[idx] # (J,3)


                               
                cx, cy, w_box, h_box = xywh_bboxes[idx]
                x = int(cx - w_box / 2)
                y = int(cy - h_box / 2)
                w_box = int(w_box)
                h_box = int(h_box)

                score_child, score_adult = cal_child_parent_score(kpts3d, frame, x, y, w_box, h_box, classifier, cls_transform, device, skeleton_model)
               
               
                # 在 score_child 计算之后立刻加：
                # 把该 tid 的滑动窗口插入本帧 score_child
                if tid not in score_hist:
                    score_hist[tid] = collections.deque(maxlen=100)

                # 如果这是一个新出现的 track，要把 stable_count 和 hard_lock 初始化：
                if tid not in stable_count:
                    stable_count[tid] = 0
                if tid not in hard_lock:
                    hard_lock[tid] = False
                
                # 只有当分数在有效区间时才加入滑动窗口（异常检测过滤）
                # 如果 score_child 和 score_adult 同时极低，说明模型根本没看清人，不应存入历史影响均值
                if (score_child + score_adult) > 0.1:
                    score_hist[tid].append(score_child)


                # 如果 tid 已经有锁定标签，就带上旧标签；否则先暂不决定（后续锁定逻辑再更新）
                old_label = label_lock.get(tid, None)
                print(f"tid:{tid}, old_label:{old_label}, score_child:{score_child}, score_adult:{score_adult}")
                tmp_info.append((
                    tid, 
                    (x, y, w_box, h_box), 
                    kpts3d, 
                    old_label, 
                    score_child, 
                    score_adult, 
                    corresponding_2d, 
                    head_box
                ))

           
        with open(out_json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)
        print(f"Saved JSON to {out_json_path}")
        #print(f"Saved visualization video to {out_video_path}")

    missing_info_path = os.path.join(args.output_root, 'missing_info.json')
    organized_missing_info = organize_missing_info(missing_info_records)
    with open(missing_info_path, 'w') as f:
        json.dump({"missing_info": organized_missing_info}, f, indent=2)
    print(f"Saved missing info JSON to {missing_info_path}")

    print("All videos processed.")


if __name__ == '__main__':
    main()