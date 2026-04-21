import os
import torch
import numpy as np
import joblib
import collections
from torch.utils.data import DataLoader
from torchgeometry import rotation_matrix_to_angle_axis

from cmd_parser import parse_config
import constants as cfg
import hps
from dataset import Dataset
from results import Results
from hps.smpl_utils import collect_results_for_image_dapa, cam_crop2full
from postprocess.temporal_smplify import TemporalSMPLify
from postprocess.one_euro_filter import OneEuroFilter


TOUCH_THRESH_3D = 0.25
TOUCH_THRESH_2D_RATIO = 0.03
FOCAL_FACTOR = 5
IMAGE_HEIGHT = 1080 


def unbatch(ts, target_dim):
    if len(ts.shape) != target_dim: 
        assert ts.shape[0] == 1
        ts = ts[0]
    assert len(ts.shape) == target_dim
    return ts


def get_valid_idxs(results, pids, filter_by_2dkp=False, min_2dkp=4):
    infant_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smil', 'infant']]
    adult_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smpl', 'adult']]
    
    valid_infant_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in infant_pids])
    valid_adult_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in adult_pids])

    if filter_by_2dkp:
        if len(infant_pids) > 0:
            valid_infant_idxs = valid_infant_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in infant_pids])
        if len(adult_pids) > 0:
            valid_adult_idxs = valid_adult_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in adult_pids])

    valid_infant_idxs = valid_infant_idxs.astype(bool).flatten()
    valid_adult_idxs = valid_adult_idxs.astype(bool).flatten()
    return infant_pids, adult_pids, valid_infant_idxs, valid_adult_idxs


class DynamicTouchPipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size
        self.cam_focal_length = args.camera_focal

        if self.args.hps == 'dapa':
            self.adult_model = hps.get_dapa_model(cfg.dapa_adult_model, cfg.smpl_mean_params).to(self.device).eval()
            self.infant_model = hps.get_dapa_model(cfg.dapa_child_model, cfg.smpl_mean_params).to(self.device).eval()
        elif self.args.hps == 'cliff':
            from hps.cliff import cliff_hr48, strip_prefix_if_present
            self.cliff_model = cliff_hr48(cfg.smpl_mean_params).to(self.device)
            state_dict = torch.load(cfg.cliff_hr48_model)['model']
            state_dict = strip_prefix_if_present(state_dict, prefix="module.")
            self.cliff_model.load_state_dict(state_dict, strict=True)
            self.cliff_model.eval()
        else:
            raise NotImplementedError('Unknown hps model')

    def run_3d_estimation(self, dataset, results_holder):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        camera_center = dataset.camera_center
        orig_img_width, orig_img_height = camera_center[0] * 2, camera_center[1] * 2

        smplify_runner = None
        if self.args.run_smplify:
            smplify_runner = TemporalSMPLify(
                step_size=1e-2, num_iters=self.args.smplify_iters, focal_length=self.cam_focal_length,
                use_lbfgs=False, device=self.device, max_iter=20, ground_weight=self.args.ground_weight
            )

        adult_tracks, infant_tracks = dataset.get_sorted_track_by_body_type()
        tracks_to_be_fitted = adult_tracks + infant_tracks

        # Stage 1: 3D Body Fitting
        for track_id in tracks_to_be_fitted:
            pidxs = dataset.track_to_id[track_id]
            body_type = dataset.track_body_types[track_id][0]

            for batch_start_i in range(0, len(pidxs), self.batch_size):
                batch_pidxs = pidxs[batch_start_i: batch_start_i + self.batch_size]
                cur_batch_size = len(batch_pidxs)
                batch = dataloader.collate_fn([dataset[i] for i in batch_pidxs])
                yhxw = batch['yhxw']
                cx, cy = yhxw[:, 2] + yhxw[:, 3] / 2., yhxw[:, 0] + yhxw[:, 1] / 2.

                if self.args.hps == 'dapa':
                    model = self.adult_model if body_type == 'adult' else self.infant_model
                    with torch.no_grad():
                        pred_rotmat, pred_betas, pred_camera = model(batch['norm_cropped_img'].to(self.device))
                elif self.args.hps == 'cliff':
                    norm_img = batch['norm_cropped_img'].to(self.device)
                    b = torch.max(yhxw[:, 1], yhxw[:, 3])
                    bbox_info = torch.stack([cx - orig_img_width / 2., cy - orig_img_height / 2., b], dim=-1).to(self.device).float()
                    bbox_info[:, :2] = bbox_info[:, :2] / self.cam_focal_length * 2.8
                    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * self.cam_focal_length) / (0.06 * self.cam_focal_length)
                    with torch.no_grad():
                        pred_rotmat, pred_betas, pred_camera = self.cliff_model(norm_img, bbox_info)

                init_betas = pred_betas
                pred_rotmat_hom = torch.cat([
                    pred_rotmat.view(-1, 3, 3),
                    torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).view(1, 3, 1).expand(cur_batch_size*24, -1, -1)
                ], dim=-1)
                pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(cur_batch_size, -1)
                pred_pose[torch.isnan(pred_pose)] = 0.0
                
                init_global_orient = pred_pose[:,:3]
                init_pose = pred_pose[:, 3:]

                focal_length = torch.tensor([self.cam_focal_length], dtype=torch.float32, device=self.device).expand(cur_batch_size)
                full_image_shape = torch.tensor([orig_img_height, orig_img_width]).to(self.device).float().unsqueeze(0).expand(cur_batch_size, -1)
                scale = torch.max(yhxw[:, 1], yhxw[:, 3]).to(self.device) / 200
                center = torch.stack((cx, cy), dim=-1).to(self.device)
                init_transl = cam_crop2full(pred_camera, center, scale, full_image_shape, focal_length)

                if self.args.hps == 'cliff' and body_type == 'infant':
                    init_transl[:, 0] *= 0.5
                    init_transl[:, 1] *= 0.3
                    init_transl[:, 2] *= 0.2

                if not self.args.run_smplify:
                    _, results_batch = collect_results_for_image_dapa(
                        pred_pose, pred_betas, pred_camera, None, batch, self.cam_focal_length, orig_img_width, orig_img_height,
                        smpl_type=self.args.smpl_model, kid_age=self.args.kid_age)
                else:
                    model_type = 'smil' if body_type == 'infant' else 'smpl'
                    init_betas = init_betas[:1, ...]
                    smplify_results, _ = smplify_runner(
                        model_type, init_global_orient, init_pose, init_betas, init_transl, 
                        camera_center, batch['keypoints'], None, None)
                    refined_thetas = smplify_results['theta']
                    refined_transl = refined_thetas[:, :3]
                    refined_pose = refined_thetas[:, 3:-10]
                    refined_betas = refined_thetas[:, -10:]
                    
                    _, results_batch = collect_results_for_image_dapa(
                        refined_pose, refined_betas, pred_camera, refined_transl, batch, self.cam_focal_length, orig_img_width, orig_img_height,
                        smpl_type=self.args.smpl_model, kid_age=self.args.kid_age)

                results_holder.update_results(batch['idx'].numpy(), results_batch)

        # Stage 2: Temporal Smoothing
        if self.args.run_smplify:
            for track_id, pidxs in dataset.track_to_id.items():
                pred_pose = np.stack([results_holder.results[i]['body_pose'][0] for i in pidxs])
                pred_orient = np.stack([results_holder.results[i]['global_orient'][0] for i in pidxs])
                pred_transl = np.stack([results_holder.results[i]['transl'][0] for i in pidxs])

                pose_filter = OneEuroFilter(np.zeros_like(pred_pose[0]), pred_pose[0], min_cutoff=0.004, beta=0.7)
                orient_filter = OneEuroFilter(np.zeros_like(pred_orient[0]), pred_orient[0], min_cutoff=0.004, beta=0.7)
                transl_filter = OneEuroFilter(np.zeros_like(pred_transl[0]), pred_transl[0], min_cutoff=0.004, beta=0.7)
                
                for i in range(1, len(pred_pose)):
                    pred_pose[i] = pose_filter(np.ones_like(pred_pose[i]) * i, pred_pose[i])
                    pred_orient[i] = orient_filter(np.ones_like(pred_orient[i]) * i, pred_orient[i])
                    pred_transl[i] = transl_filter(np.ones_like(pred_transl[i]) * i, pred_transl[i])

                for i, idx in enumerate(pidxs):
                    results_holder.results_smoothed[idx] = results_holder.results[idx]
                    results_holder.results_smoothed[idx]['body_pose'] = pred_pose[i].reshape(1, -1)
                    results_holder.results_smoothed[idx]['global_orient'] = pred_orient[i].reshape(1, -1)
                    results_holder.results_smoothed[idx]['transl'] = pred_transl[i].reshape(1, -1)
        
        return results_holder

    def calculate_touch_labels(self, dataset, results):
        img_to_pid = collections.defaultdict(list)
        img_list = []
        for pid, (img_name, _) in dataset.person_to_img.items():
            img_to_pid[img_name].append(pid)
            img_list.append(img_name)
        img_list = np.sort(np.unique(img_list))

        labels = {}
        for img_name in img_list:
            pids = img_to_pid[img_name]
            infant_pids, adult_pids, valid_infant_idxs, valid_adult_idxs = get_valid_idxs(results, pids, filter_by_2dkp=False)
            exist_dyad = sum(valid_infant_idxs) != 0 and sum(valid_adult_idxs) != 0

            touch = 2 
            if exist_dyad:
                infant_pid = np.array(infant_pids)[valid_infant_idxs][0]
                adult_pids_valid = np.array(adult_pids)[valid_adult_idxs]
                
                infant_joints_3d = unbatch(results.results[infant_pid]['joints'], 2)[:25]
                
                dist_adult = [np.sqrt(((infant_joints_3d - unbatch(results.results[a_pid]['joints'], 2)[:25])**2).sum(1)).mean() for a_pid in adult_pids_valid]
                adult_pid = adult_pids_valid[np.argmin(dist_adult)]
                adult_joints_3d = unbatch(results.results[adult_pid]['joints'], 2)

                joint_dist_2d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:2] - infant_joints_3d[:,:2]).sum(1)) for i in range(25)])
                joint_dist_3d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:] - infant_joints_3d).sum(1)) for i in [0]]) # Reference to original calc_downstream.py behavior
                
                touch_2d = int(joint_dist_2d.min() > TOUCH_THRESH_2D_RATIO * IMAGE_HEIGHT)
                if not touch_2d:
                    touch = int(joint_dist_3d.min() > TOUCH_THRESH_3D)
                else:
                    touch = touch_2d

            labels[img_name] = {'touch': int(touch)}
            
        return labels

    def process(self, images_folder, out_folder):
        dataset = Dataset(images_folder, out_folder=out_folder, tracker_type=self.args.tracker_type, pipeline=self.args.pipeline, cfg=cfg)
        results_holder = Results()

        results_holder = self.run_3d_estimation(dataset, results_holder)
        
        target_results = results_holder if not self.args.run_smplify else results_holder # Actually original code falls back to raw results dict inside Results object
        
        touch_labels = self.calculate_touch_labels(dataset, target_results)
        
        return touch_labels

if __name__ == '__main__':
    args = parse_config()
    pipeline = DynamicTouchPipeline(args)
    final_labels = pipeline.process(args.images, args.out_folder)
    print("Dynamic Touch Labels:", final_labels)
