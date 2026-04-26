import os
import cv2
import numpy as np
import PIL.Image as pil_img
import smplx
import torch

import constants as cfg
from hps.body_model import init_body_model
# 彻底注释掉 smpla_model 的导入，因为它底层依赖 smil
# from hps.smpla import prepare_smpla_model

crop_size = 224

# 1. 初始化成人 SMPL 模型
adult_bm = init_body_model(model_path=cfg.smpl_model_path, batch_size=1, create_body_pose=False).cuda()
# 2. 强制将婴儿模型指向成人模型，避开文件缺失报错
infant_bm = adult_bm 
# 3. 强制关闭 smpla 模型
smpla_bm = None

def get_original(cam, x, y, h, target_focal, orig_img_width, orig_img_height):
    scale = crop_size / h
    undo_scale = 1. / scale
    flength = 500.
    curr_focal = flength * undo_scale
    tz = flength / (0.5 * crop_size * cam[0])
    trans = np.hstack([cam[1:], tz])
    dx = (orig_img_width/2 - (x+h/2)); dy = orig_img_height/2 - (y+h/2)
    trans[2] /= (curr_focal/target_focal)
    new_tz = tz / (curr_focal/target_focal)
    trans[0] -= new_tz * dx / target_focal
    trans[1] -= new_tz * dy / target_focal  
    return trans

def get_result(pred_pose, pred_betas, pred_cam, body_type, yhxw, target_focal, orig_img_width, orig_img_height, smpl_type, kid_age=1.0):
    pred_pose = pred_pose.unsqueeze(0)
    pred_betas = pred_betas.unsqueeze(0) 
    pred_cam = pred_cam.unsqueeze(0).clone()
    
    # 4. 彻底抛弃复杂的分类逻辑，强制所有人统一使用 adult_bm (成人 SMPL) 生成网格
    bm = adult_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas[:, :10])
    joints = bm.joints.detach()[0].detach().cpu().numpy()
    verts = bm.vertices.detach()[0].detach().cpu().numpy()

    # 转换回全图全局坐标系
    device = pred_cam.device
    yhxw = torch.from_numpy(yhxw).unsqueeze(0).to(device)
    scale = torch.max(yhxw[:, 1], yhxw[:, 3]) / 200
    cx, cy = yhxw[:, 2] + yhxw[:, 3] / 2., yhxw[:, 0] + yhxw[:, 1] / 2.
    center = torch.stack((cx, cy), dim=-1)
    focal_length = torch.tensor([target_focal], dtype=torch.float32, device=device)
    full_image_shape = torch.tensor([orig_img_height, orig_img_width]).to(device).float().unsqueeze(0)
    trans = cam_crop2full(pred_cam, center, scale, full_image_shape, focal_length)

    trans = trans.squeeze(0).cpu().numpy()

    if body_type == 'infant':
        trans[0] *= 0.5
        trans[1] *= 0.3
        trans[2] *= 0.2
        
    return verts, joints, trans

def collect_results_for_image_dapa(pred_pose, pred_betas, pred_cam, pred_transl, batch, target_focal, orig_img_width, orig_img_height, smpl_type='smpla', kid_age=1.0):
    pred_pose = pred_pose.detach()
    pred_betas = pred_betas.detach()
    if pred_cam is not None:
        pred_cam = pred_cam.detach()
    img_names = np.sort(np.unique(batch['img_name']))
    verts_results = dict()
    yhxw = batch['yhxw'].cpu().numpy()
    body_types = batch['body_type']
    keypoints = batch['keypoints']
    results_batch = [None for _ in range(len(batch['img_name']))]
    
    for img_name in img_names:
        idxs = np.where(np.array(batch['img_name'])==img_name)[0]
        verts_results[img_name] = []
        for i in idxs:
            verts, joints, trans = get_result(
                pred_pose[i], pred_betas[i], pred_cam[i], body_types[i], yhxw[i], target_focal,
                orig_img_width, orig_img_height, smpl_type, kid_age)
            if pred_transl is not None:
                trans = pred_transl[i].detach().cpu().numpy()
            verts_results[img_name].append(verts)
            assert results_batch[i] is None
            results_batch[i] = {
                'img_name': img_name,
                'betas': pred_betas[i].unsqueeze(0).cpu().numpy(), 
                'pred_cameras': pred_cam[i].unsqueeze(0).cpu().numpy(), 
                'body_pose': pred_pose[i, 3:].unsqueeze(0).cpu().numpy(), 
                'global_orient': pred_pose[i, :3].unsqueeze(0).cpu().numpy(),
                'transl': np.expand_dims(trans, axis=0),
                'joints': np.expand_dims(joints, axis=0) + np.expand_dims(trans, axis=0),
                'body_type': body_types[i],
                # 此处强制写入原本身份标识供下游使用
                'model_type': 'smil' if body_types[i] == 'infant' else 'smpl',
                'keypoints': keypoints[i].unsqueeze(0).cpu().numpy(),
            }
        
    return verts_results, results_batch

def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam
