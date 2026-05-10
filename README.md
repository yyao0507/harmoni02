# HARMONI 精简 Pipeline

本目录在 HARMONI 仓库内提供一条**可配置的**处理链：视频抽帧、与论文一致的 **touch / visibility 几何**（见 `downstream/calc_downstream.py`）、以及可选的**官方预处理 / 全量 `main.py`**。编排逻辑集中在 **`pipeline.py`**；各步骤在需要时才 `import` 同目录模块或父仓库模块。

## 目录结构

| 文件 / 目录 | 说明 |
|-------------|------|
| `pipeline.py` | 主入口：`run()` / `main()`，三种 `run_mode` 分支 |
| `video_io.py` | 视频 / GIF 抽帧 |
| `touch_geometry.py` | 双人 25 关节 touch、visibility（NumPy，无 skspatial） |
| `procedural_joints.py` | `joints_source: synthetic` 时的演示关节序列 |
| `configs/default.yaml` | 默认配置 |
| `configs/harmoni_bundle.yaml` | 传给官方 `main.py` 的 `--config` 轻量覆盖 |
| `requirements.txt` | 几何分支依赖；完整 HARMONI 另需父仓库视觉环境 |

## 安装

在仓库根目录执行：

```bash
pip install -r pipeline/requirements.txt
```

- **`run_mode: geometry`**：仅需上述依赖（NumPy、OpenCV、PyYAML、Pillow、imageio、joblib）。
- **`harmoni_preprocess` / `harmoni_full`**：还需按仓库根目录 [README](../README.md) 安装 **`install_visual.sh`** 对应环境（PyTorch、loguru、omegaconf、`data/ckpts` 与 body 模型等）。

## 运行模式（`run_mode`）

在 `configs/default.yaml` 中设置 `run_mode`，或用命令行 **`--run_mode`** 覆盖。

### 1. `geometry`（默认）

轻量流程：抽帧（或使用 `--skip_video` + `demo_frames`）→ 每帧关节（**synthetic** 或 **npz**）→ 输出 `labels.json` / `summary.json`。不跑 OpenPose / DAPA，便于在无 GPU 环境验证几何与配置。

```bash
# 无视频：纯几何烟测
python pipeline/pipeline.py --skip_video --demo_frames 32 --out_dir pipeline/outputs/demo_geom

# 有视频：抽帧 + synthetic 关节 + touch
python pipeline/pipeline.py --video path/to/video.mp4 --out_dir pipeline/outputs/from_video
```

### 2. `harmoni_preprocess`

抽帧（或配置中的 `harmoni.images_dir`）→ 父仓库 **`dataset.Dataset`**（OpenPose、身体类型分类、`tracker_type` 为 `dummy` 或 `phalp`）→ 写出与官方一致的 **`dataset.pt`**。

```bash
python pipeline/pipeline.py --run_mode harmoni_preprocess \
  --video path/to/video.mp4 --out_dir results/prep
```

### 3. `harmoni_full`

在 HARMONI 仓库根目录下调用官方 **`main.main`**（DAPA、可选 SMPLify、渲染等），等价于带参运行根目录 `main.py`。默认 **`pass_video_to_main: true`** 时会把 `--video` 直接交给 `main`；否则由本 pipeline 先抽帧再传 `--images`。

```bash
python pipeline/pipeline.py --run_mode harmoni_full \
  --video path/to/video.mp4 --out_dir results/full_run
```

预处理与全量模式会在执行相关步骤时**临时 `chdir` 到仓库根**，以便 `data/ckpts`、`detectors/` 等相对路径正确解析。

## 命令行参数

| 参数 | 含义 |
|------|------|
| `--config` | YAML 配置路径，默认 `pipeline/configs/default.yaml` |
| `--video` | 输入视频（多数模式需要，或与 `harmoni.images_dir` 二选一） |
| `--out_dir` | 输出目录 |
| `--skip_video` | 不抽帧；`geometry` 下用配置里的 `demo_frames` |
| `--demo_frames` | 覆盖配置中的 `demo_frames`（与 `--skip_video` 联用） |
| `--run_mode` | `geometry` \| `harmoni_preprocess` \| `harmoni_full`，覆盖 YAML |

运行结束后会在 `out_dir` 写入 **`resolved_config.yaml`**（实际使用的配置快照）。

## 配置要点（`configs/default.yaml`）

- **`joints_source`**：`synthetic`（默认）或 `npz`（需提供 `npz.path`）。
- **`touch`**：`touch_thresh_3d`、`touch_thresh_2d_ratio`、`reference_image_height` 等与 HARMONI downstream 一致的含义。
- **`harmoni`**（非 `geometry` 时）：
  - `config_yaml`：传给 `main` 的配置文件（相对路径相对**仓库根**）。
  - `tracker_type`：`dummy` 或 `phalp`。
  - `pipeline`：`1`（OpenPose 管线）或 `2`（Grounded DINO 等，见父仓库说明）。
  - `pass_video_to_main` / `images_dir` / `main_extra_argv`：见 YAML 内注释。

### 路径解析（与 `pipeline.py` 一致）

| 配置项 | 规则 |
|--------|------|
| `harmoni.config_yaml` | 绝对路径照用；否则 = **仓库根** + 相对路径。 |
| `npz.path` | 若已是存在的文件则规范化绝对路径；否则依次在 **当前工作目录 → 仓库根 → `pipeline/` 目录** 下拼接查找。 |
| `harmoni.images_dir` | 同上逻辑，但要求为**已存在目录**。 |

## `npz` 关节格式（`joints_source: npz`）

`.npz` 需包含：

- `infant`：形状 `(F, 25, 3)`
- `adult`：形状 `(F, 25, 3)`
- `frame_names`：长度 `F` 的对象数组，字符串文件名建议与抽帧得到的 `frame_*.jpg` 一致

`F` 必须与当前模式下的帧数一致（与视频抽帧条数或 `demo_frames` 对齐）。

## 作为 Python 包调用

在仓库根目录且将当前目录加入 `PYTHONPATH` 时：

```python
import pipeline

pipeline.run(
    config_path="pipeline/configs/default.yaml",
    video_path="path/to/video.mp4",
    out_dir="pipeline/outputs/my_run",
    skip_video=False,
    demo_frames_override=None,
    run_mode_override=None,  # 或 "geometry" / "harmoni_preprocess" / "harmoni_full"
)
```

## 与官方 HARMONI 的关系

- **几何 touch**：阈值与成对逻辑对齐 `downstream/calc_downstream.py`；默认 synthetic 仅用于跑通流程，**不能**替代真实 SMPL 关节。
- **全量重建**：请仍以根目录 [README](../README.md) 为准（权重、`data/`、SMPL 许可等）；`harmoni_full` 只是从本目录发起同一套 `main.py` 调用。


