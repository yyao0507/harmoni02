# Harmoni Baseline: Dynamic Touch Pipeline

## Introduction
This project provides a baseline pipeline to reproduce and evaluate dynamic touch between an infant and an adult (Dyad) based on images/videos. By combining 3D human body reconstruction techniques with temporal smoothing algorithms, this pipeline estimates the 3D poses and meshes of the adult and infant in the scene separately, and calculates the final interactive touch labels using multi-dimensional spatial distances.

## Pipeline Overview
The core implementation of the entire project is encapsulated in the DynamicTouchPipeline class, which mainly consists of the following three execution stages:

1. Stage 1: 3D Body Fitting
   - Basic Prediction: Supports extracting adult SMPL model parameters and infant SMIL model parameters using DAPA or CLIFF (HR48) networks.
   - Alignment & Optimization: The TemporalSMPLify module can be optionally enabled. This module uses upstream 2D keypoints to perform backward iterative optimization, further aligning the 3D mesh with the actual image performance.
2. Stage 2: Temporal Smoothing
   - Introduces the OneEuroFilter. After completing the 3D fitting, it applies smoothing to the predicted body_pose, global_orient, and spatial translation (transl) of both the adult and infant to ensure the coherence of the motion sequence and eliminate jitter caused by frame-by-frame reconstruction.
3. Stage 3: Touch Label Calculation
   - For valid adult-infant dyads, the pipeline extracts their joint coordinates in both 3D space and the 2D projection plane. The final touch status is output by comparing the shortest distance between joints against predefined thresholds.

## Touch Detection Rules
In calculate_touch_labels, the touch labels are primarily determined by the following hard thresholds:

- 2D Threshold Criterion (TOUCH_THRESH_2D_RATIO = 0.03): First, it calculates the Euclidean distance of the target joints on the 2D plane. If the minimum 2D distance is greater than 3% of the image height, it is preliminarily judged as No Touch.
- 3D Threshold Criterion (TOUCH_THRESH_3D = 0.25): If the 2D touch condition is met, it further checks the minimum distance of the 3D joints in space. If the 3D distance > 0.25 meters, it is considered No Touch; otherwise, it is judged as Touch.
- Output Definitions:
  - 0 / 1: Binary classification results for dynamic touch (1 means far distance/no touch, 0 means close distance/touch).
  - 2: There is no valid adult or infant detection in the current frame (e.g., missing at least 4 2D keypoints), making it impossible to form a valid Dyad.

## Dependencies
- Python >= 3.7
- PyTorch (CUDA environment recommended for GPU inference)
- numpy, joblib, torchgeometry
- Pre-trained Models: You need to download and place the SMPL/SMIL mean parameter templates and the DAPA/CLIFF model weights in advance (see constants.py for specific paths).

## Usage
The main entry point for execution is yy_dynamic_touch_pipeline_rep.py.

Basic Execution Example (using DAPA):
```bash
python yy_dynamic_touch_pipeline_rep.py \
    --images ./data/input_images \
    --out_folder ./data/outputs \
    --hps dapa \
    --batch_size 16

## Related Resources
We borrowed code from the below amazing resources:
- [PARE](https://github.com/mkocabas/PARE) for HMR-related helpers.
- [PHALP](https://github.com/brjathu/PHALP) for tracking.
- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation.
- [Panoptic DeepLab](https://github.com/bowenc0221/panoptic-deeplab) for segmentation.
- [size_depth_disambiguation](https://github.com/nicolasugrinovic/size_depth_disambiguation) estimating ground normal.
- [OpenPose](https://github.com/Hzzone/pytorch-openpose) for 2D keypoint estimation.



