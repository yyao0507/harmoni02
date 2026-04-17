from argparse import ArgumentParser


from filterpy.kalman import KalmanFilter
from skimage.filters import threshold_otsu

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.visualization import Pose3dLocalVisualizer
import cv2
import numpy as np
from mmpose.structures.bbox import get_warp_matrix

import torch

from dapa import get_dapa_model
from train_stgcn import SingleFrameSTGCN, get_adjacency_matrix

# from depth_anything_v2.dpt import DepthAnythingV2
# MODEL_CONFIGS = {
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
# }
# ENCODER = 'vitl'
# CHECKPOINT_PATH = f'/root/autodl-tmp/Depth-Anything-V2/depth_anything_v2_{ENCODER}.pth' 



import torch.nn as nn
from torchvision import transforms, models
import kornia
from kornia.geometry.conversions import rotation_matrix_to_angle_axis

import os


# 2. COCO-WholeBody (133) → OpenPose25 映射
COCO2OP25 = {
    0: 0, # Nose
    1: 16, # left_eye → LEye (16)
    2: 15, # right_eye → REye (15)
    3: 18, # left_ear → LEar (18)
    4: 17, # right_ear → REar (17)
    5: 5, # left_shoulder → LShoulder (5)
    6: 2, # right_shoulder → RShoulder (2)
    7: 6, # left_elbow → LElbow (6)
    8: 3, # right_elbow → RElbow (3)
    9: 7, # left_wrist → LWrist (7)
    10: 4, # right_wrist → RWrist (4)
    11:12, # left_hip → LHip (12)
    12: 9, # right_hip → RHip (9)
    13:13, # left_knee → LKnee (13)
    14:10, # right_knee → RKnee (10)
    15:14, # left_ankle → LAnkle (14)
    16:11, # right_ankle → RAnkle (11)
    17:19, # left_big_toe → LBigToe (19)
    18:20, # left_small_toe → LSmallToe (20)
    19:21, # left_heel → LHeel (21)
    20:22, # right_big_toe → RBigToe (22)
    21:23, # right_small_toe → RSmallToe (23)
    22:24, # right_heel → RHeel (24)
}

# 3. 从 dataset_info 自动生成 5 个 WholeBody 部分索引
BODY17 = list(range(0, 17))
FOOT6 = list(range(17, 23))
FACE68 = list(range(23, 91))
LHAND21 = list(range(91, 112))
RHAND21 = list(range(112, 133))



def init_pose3d_estimator(pose3d_config, pose3d_checkpoint, device):
    """Initialize MMPose 3D estimator and its visualizer config."""
    pose_estimator = init_model(
        pose3d_config, pose3d_checkpoint, device=device
    )
    # 开启可视化与 OKS 跟踪
    pose_estimator.cfg.model.test_cfg.mode = 'vis'
    pose_estimator.cfg.model.test_cfg.use_oks_tracking = True
    pose_estimator.cfg.model.test_cfg.tracking_thr = 0.6
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 2

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)
    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    return pose_estimator, visualizer





def predict_child_adult(model, fullbody_crop, transform, device):
    """
    输入：BGR 图（一个人的全身 ROI），输出：("child"/"adult", p_child, p_adult)
    """
    h, w, _ = fullbody_crop.shape
    if h == 0 or w == 0:
        # 如果裁剪区域无效，则返回成人
        return "adult", 0.0, 1.0
    img_rgb = cv2.cvtColor(fullbody_crop, cv2.COLOR_BGR2RGB)
    x = transform(img_rgb).unsqueeze(0).to(device) # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x) # (1, 2)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() # (2,)
    idx = int(probs.argmax())
    return ("child" if idx == 1 else "adult"), float(probs[1]), float(probs[0])

def predict_child_skeleton(kpts3d, model, device):
    """
    kpts3d: numpy array of shape (17,3)
    返回: (p_child, p_adult)
    """
    # transpose → (3,17), expand batch → (1,3,17)
    x = torch.from_numpy(kpts3d.T.astype(np.float32)) \
             .unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x) # (1,2)
        probs = torch.softmax(logits, 1) # (1,2)
    p_child = probs[0,1].item()
    p_adult = probs[0,0].item()
    return p_child, p_adult




def refine_pose2d_results(pose2d_results):
    """
    对 pose2d_results 进行原地修正：将模型空间坐标还原为图像绝对坐标。
    """
    for res in pose2d_results:
        # 1. 提取元数据
        if 'input_center' not in res.metainfo:
            continue  # 如果没有元数据，跳过
            
        center = res.metainfo['input_center']
        scale = res.metainfo['input_scale']
        input_size = res.metainfo['input_size']
        
        # 2. 计算变换矩阵及其逆矩阵
        # get_warp_matrix 得到的是 [原图 -> 模型输入]
        warp_mat = get_warp_matrix(center, scale, rot=0, output_size=input_size)
        # cv2.invertAffineTransform 得到 [模型输入 -> 原图]
        inv_warp_mat = cv2.invertAffineTransform(warp_mat)

        # 3. 提取并还原坐标
        # 获取原始坐标 (N, 3)，包含 x, y, score
        raw_kpts = res.pred_instances.keypoints.copy()
        
        # 处理可能的维度问题 (Batch 维度)
        if raw_kpts.ndim == 3:
            # 针对 [1, 17, 3] 这种格式
            coords_to_transform = raw_kpts[0, :, :2].reshape(-1, 1, 2)
            abs_coords = cv2.transform(coords_to_transform, inv_warp_mat).squeeze(1)
            
            # 将还原后的坐标写回
            res.pred_instances.keypoints[0, :, :2] = abs_coords
        else:
            # 针对 [17, 3] 这种格式
            coords_to_transform = raw_kpts[:, :2].reshape(-1, 1, 2)
            abs_coords = cv2.transform(coords_to_transform, inv_warp_mat).squeeze(1)
            
            # 将还原后的坐标写回
            res.pred_instances.keypoints[:, :2] = abs_coords
            
    return pose2d_results







# ============ Gaze 模型相关 ============
from ultralytics import YOLO
from src.modeling.sharingan import Sharingan
from src.utils.common import spatial_argmax2d, square_bbox
from boxmot import OCSORT
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.cm as cm
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]
DET_THR = 0.0 # 不再用 confidence threshold

def load_gaze_models(sharingan_ckpt, device):
    # 虽然不跑 YOLO 检测，但还要构造一个 dummy head_det 用于接口统一
    # face_det = YOLO(yolo_ckpt).to(device).eval()
    # 加载 Sharingan
    sharingan = Sharingan( # 同你原来的参数
      patch_size=16, token_dim=768, image_size=224,
      gaze_feature_dim=512, encoder_depth=12, encoder_num_heads=12,
      encoder_num_global_tokens=0, encoder_mlp_ratio=4.0,
      encoder_use_qkv_bias=True, encoder_drop_rate=0.0,
      encoder_attn_drop_rate=0.0, encoder_drop_path_rate=0.0,
      decoder_feature_dim=128, decoder_hooks=[2,5,8,11],
      decoder_hidden_dims=[48,96,192,384], decoder_use_bn=True
    )
    ckpt = torch.load(sharingan_ckpt, map_location="cpu")
    sd = {k.replace("model.",""):v for k,v in ckpt["state_dict"].items()}
    sharingan.load_state_dict(sd, strict=True)
    sharingan.to(device).eval()
    return sharingan

def _predict_gaze(frame: Image.Image, sharingan, head_det, tracker, device: torch.device):
    img_np = np.array(frame)
    # 1) 检测 heads
    results = head_det(img_np)
    boxes = results[0].boxes.xyxy.cpu().numpy() # [N,4]
    confs = results[0].boxes.conf.cpu().numpy() # [N]
    dets = np.concatenate([boxes, confs[:,None]], axis=1) if len(boxes) else np.zeros((0,5))
    # 2) tracking
    tracks = tracker.update(
      np.concatenate([dets, np.zeros((dets.shape[0],1))], axis=1), # pad cls col
      img_np
    )
    if len(tracks)==0:
        return torch.empty((0,2)), torch.empty((0,3)), torch.empty((0,)), torch.empty((0,4)), torch.empty((0,224,224)), np.array([],int)
    pids = (tracks[:,4]-1).astype(int)
    head_bboxes= torch.from_numpy(tracks[:,:4]).float()
    tb = square_bbox(head_bboxes, *img_np.shape[:2][::-1])
    # 3) crop & normalize heads
    heads = []
    for bb in tb:
        crop = frame.crop(bb.numpy().astype(int))
        head = TF.resize(TF.to_tensor(crop),(224,224))
        heads.append(head)
    heads = TF.normalize(torch.stack(heads), mean=IMG_MEAN, std=IMG_STD)
    # 4) full image
    img_t = TF.normalize(TF.resize(TF.to_tensor(frame),(224,224)), mean=IMG_MEAN, std=IMG_STD)
    tb = tb / torch.tensor([img_np.shape[1],img_np.shape[0]]*2,dtype=torch.float32)
    sample={"image":img_t.unsqueeze(0).to(device),
            "heads":heads.unsqueeze(0).to(device),
            "head_bboxes":tb.unsqueeze(0).to(device)}
    with torch.no_grad():
        gv,ghm,inouts = sharingan(sample)
    ghm = ghm.squeeze(0).cpu()
    gv = gv.squeeze(0).cpu()
    gp = spatial_argmax2d(ghm,normalize=True)
    inouts = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()
    return gp, gv, inouts, head_bboxes, ghm, pids

def _ray_box_intersection(Cx, Cy, dx, dy, x1, y1, x2, y2):
    """
    Cx,Cy: 射线起点；dx,dy: 单位方向；x1,y1,x2,y2: box 坐标
    返回第一个正向相交点 (Px,Py)
    """
    ts = []
    # left & right
    if dx>0:
        t = (x2 - Cx)/dx
        ts.append(t)
    elif dx<0:
        t = (x1 - Cx)/dx
        ts.append(t)
    # top & bottom
    if dy>0:
        t = (y2 - Cy)/dy
        ts.append(t)
    elif dy<0:
        t = (y1 - Cy)/dy
        ts.append(t)
    # 找最小正 t
    t_pos = [t for t in ts if t>0]
    if not t_pos:
        return Cx, Cy
    t_min = min(t_pos)
    return Cx + dx*t_min, Cy + dy*t_min
def _gen_new_color(existing_colors):
    # 简单随机，或者用 HSV 均匀采样
    while True:
        c = tuple(np.random.choice(range(50,256), size=3).tolist())
        if c not in existing_colors:
            return c



def run_2d_detection_and_tracking(
    detector,
    frame,
    frame_idx,
    args,
    track_kf,
    track_last_seen,
    track_age,
    score_hist,
    label_lock,
    max_age,
    next_tid,
    missing_info_records,
):
    """Run section 5.1: 2D detection + Kalman-based association/tracking."""
    det_result = inference_detector(detector, frame)
    det_instances = det_result.pred_instances.cpu().numpy()
    keep_mask = np.logical_and(
        det_instances.labels == 0,
        det_instances.scores > args.bbox_thr
    )
    raw_bboxes = det_instances.bboxes[keep_mask] # (N,4)

    # 如果这一帧没人，执行关键帧重连检查
    if len(raw_bboxes) == 0:
        # 当前帧无检测，超过 max_age 的 track 直接删除
        for tid in list(track_last_seen.keys()):
            if frame_idx - track_last_seen[tid] > max_age:
                track_kf.pop(tid, None)
                track_last_seen.pop(tid, None)
                track_age.pop(tid, None)
                score_hist.pop(tid, None)
                label_lock.pop(tid, None)

        return None, None, next_tid, missing_info_records, True, True

    # 5.1.1 用卡尔曼预测所有活跃 track 到当前帧位置
    preds = {}  # {tid: [cx, cy, w, h]}
    for tid, kf in list(track_kf.items()):
        kf.predict()
        cx, cy, w, h = kf.x[0, 0], kf.x[1, 0], kf.x[2, 0], kf.x[3, 0]
        preds[tid] = [cx, cy, w, h]

    # 5.1.2 匹配：用 IoU 关联 raw_bboxes 和 preds
    unmatched_raw = set(range(len(raw_bboxes)))
    unmatched_tids = set(preds.keys())
    matches = {}  # {raw_idx: matched_tid}

    raw_xywh = []
    for box in raw_bboxes:
        x1, y1, x2, y2 = box.astype(int)
        w, h = x2 - x1, y2 - y1
        raw_xywh.append([x1 + w / 2, y1 + h / 2, w, h])
    raw_xywh = np.array(raw_xywh)

    iou_mat = np.zeros((len(raw_xywh), len(preds)), dtype=float)
    tids_list = list(preds.keys())
    for i, (cx, cy, w, h) in enumerate(raw_xywh):
        x1_r, y1_r = cx - w / 2, cy - h / 2
        x2_r, y2_r = cx + w / 2, cy + h / 2
        for j, tid in enumerate(tids_list):
            cx_p, cy_p, w_p, h_p = preds[tid]
            x1_p, y1_p = cx_p - w_p / 2, cy_p - h_p / 2
            x2_p, y2_p = cx_p + w_p / 2, cy_p + h_p / 2
            xx1 = max(x1_r, x1_p)
            yy1 = max(y1_r, y1_p)
            xx2 = min(x2_r, x2_p)
            yy2 = min(y2_r, y2_p)
            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            inter = inter_w * inter_h
            area_r = w * h
            area_p = w_p * h_p
            union = area_r + area_p - inter
            if union > 0:
                iou_mat[i, j] = inter / union

    iou_thr = 0.3
    for _ in range(min(len(raw_xywh), len(preds))):
        idx_flat = np.argmax(iou_mat)
        i, j = np.unravel_index(idx_flat, iou_mat.shape)
        if iou_mat[i, j] < iou_thr:
            break
        matched_tid = tids_list[j]
        matches[i] = matched_tid
        unmatched_raw.discard(i)
        unmatched_tids.discard(matched_tid)
        iou_mat[i, :] = -1
        iou_mat[:, j] = -1

    # 5.1.3 为 unmatched_raw 分配新 ID，为 unmatched_tids 增加 age
    for i in unmatched_raw:
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.x[:4] = np.array(raw_xywh[i]).reshape(4, 1)
        kf.F = np.eye(7)
        kf.F[0, 4] = 1
        kf.F[1, 5] = 1
        kf.F[2, 6] = 1
        kf.H = np.zeros((4, 7))
        kf.H[0, 0] = 1
        kf.H[1, 1] = 1
        kf.H[2, 2] = 1
        kf.H[3, 3] = 1
        kf.P *= 10.0
        kf.R *= 5.0
        kf.Q *= 0.01

        new_tid = next_tid
        next_tid += 1
        track_kf[new_tid] = kf
        track_last_seen[new_tid] = frame_idx
        track_age[new_tid] = 0
        matches[i] = new_tid

    for lost_tid in list(unmatched_tids):
        track_age[lost_tid] += 1
        if track_age[lost_tid] > max_age:
            track_kf.pop(lost_tid, None)
            track_last_seen.pop(lost_tid, None)
            track_age.pop(lost_tid, None)
            score_hist.pop(lost_tid, None)
            label_lock.pop(lost_tid, None)

    # 5.1.4 用 matches 确定当前帧 raw_box 对应的 tid，并 update KF
    final_tids = []
    for i, _ in enumerate(raw_bboxes):
        cx, cy, w, h = raw_xywh[i]
        assigned_tid = matches[i]
        final_tids.append(assigned_tid)
        kf = track_kf[assigned_tid]
        kf.update(np.array([cx, cy, w, h]).reshape(4, 1))
        track_age[assigned_tid] = 0
        track_last_seen[assigned_tid] = frame_idx

    associated_tids = final_tids
    return raw_bboxes, associated_tids, next_tid, missing_info_records, False, False


def run_pose2d_inference(
    pose_estimator,
    frame,
    raw_bboxes,
):
    """Run section 5.2: topdown pose inference and keypoint-based head debug outputs."""
    xywh_bboxes = []
    for box in raw_bboxes:
        x1, y1, x2, y2 = box.astype(int)
        w, h = x2 - x1, y2 - y1
        xywh_bboxes.append([
            x1 + w / 2,
            y1 + h / 2,
            w,
            h
        ])
    xywh_bboxes = np.array(xywh_bboxes)

    pose2d_results = inference_topdown(
        pose_estimator,
        frame,
        raw_bboxes,
        bbox_format='xyxy'
    )
    
    return (
        xywh_bboxes,
        pose2d_results
    )





def get_head_bbox_wholebody(res, bbox, conf_thr = 0.4):
    kpts = res.pred_instances.keypoints[0] # (133, 3)

    main_face_data = kpts[0:5,:2]
    main_face_conf = kpts[0:5,2]

    
    # 提取 Wholebody 中的面部 68 点 (索引 23 到 90)
    face_68_data = kpts[23:91, :2]
    face_68_conf = kpts[23:91, 2]
    
    # 2. Correctly merge valid points using concatenation
    valid_main = main_face_data[main_face_conf > conf_thr]
    valid_68 = face_68_data[face_68_conf > conf_thr]
    
    if len(valid_main) == 0 and len(valid_68) == 0:
        return None
    
    # Combine all points into one array for min/max calculation
    all_valid_pts = np.concatenate([valid_main, valid_68], axis=0)
    
    if len(all_valid_pts) < 5: 
        return None
    
    x1, y1 = np.min(all_valid_pts, axis=0)
    x2, y2 = np.max(all_valid_pts, axis=0)

    # 以面部宽度为基准，左右各扩充 50%
    w = x2 - x1
    pad_w = w * 0.5
    x1 = max(0, x1 - pad_w)
    x2 = x2 + pad_w
    
    
    # 稍微向上扩充一点点（因为 68 点最高只到眉毛，不到头顶）
    h = y2 - y1
    y1 -= h * 0.3 

    #以当前高度为基准，上扩充50%，下扩充20%
    pad_h_top = (y2 - y1) * 0.5
    pad_h_bottom = (y2 - y1) * 0.2
    y2 = y2 + pad_h_bottom
    y1 = max(0, y1 - pad_h_top)

    # final could not exceed bbox bounds
    x1 = max(x1, bbox[0])
    y1 = max(y1, bbox[1])
    x2 = min(x2, bbox[2])
    y2 = min(y2, bbox[3])
    
    return np.array([x1, y1, x2, y2])






#========== Adult & child classification ==========

def build_binary_resnet50(num_classes=2):
    """
    基于 torchvision 的 ResNet50 构建一个二分类模型，将最后一层 fc 替换为 (in_feats, num_classes)。
    """
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model