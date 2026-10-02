# EvolSplat4D Waymo preprocessing

This directory converts Waymo TFRecords into the scene layout consumed by EvolSplat4D. It has two entry points:

* `extract_waymo.py` reads raw TFRecords and writes images, poses, calibration, LiDAR, dynamic masks, objects, and optional CasTrack predictions.
* `prepare.py` computes DINOv2 + 16-D PCA features, semantic sky labels, one selected depth prior, static Drop50/Drop80 point clouds, and dynamic object tracks.

Only Waymo is supported. `--mode dynamic` writes `track_info.pth` and dynamic masks; `--mode static` omits trajectory data and is compatible with the static EvolSplat4D parser. The raw dataset and the generated output must be different directories; neither script writes intermediates to the input dataset.

## Environment

The preprocessing models can use a separate environment from the main `EVolSplat4D` environment:

```bash
conda create -n EvolSplat4D-preprocess --clone EVolSplat4D
conda activate EvolSplat4D-preprocess
python -m pip install -r Preprocess/requirements.txt
python -m pip install --only-binary=:all: --no-deps \
  'torch-cluster==1.6.3+pt25cu121' \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

The clone keeps the project environment unchanged. The tested stack uses Python 3.10, PyTorch 2.5.1 + CUDA 12.1, NumPy 1.26, and torchvision 0.20.1. `xformers` is required by UniDepthV2. A GPU is required for the neural priors.

Pretrained weights and the checkpoint-compatible PCA transform are not included in this repository. Download or point to these external assets before running `prepare.py`:

| Asset | Argument | Purpose |
| --- | --- | --- |
| DINOv2 repository | `--dino-repo` | ViT-S/14 image features |
| `dinov2_pca.joblib` | `--pca-path` | The checkpoint-compatible 384-to-16 PCA |
| `nvi_sem/checkpoints/cityscapes_ocrnet...pth` | `--semantic-checkpoint` | Sky/semantic filtering |
| Metric3D giant checkpoint | `--metric3d-checkpoint` | Metric3D depth |
| UniDepthV2 ViT-L/14 directory or Hub ID | `--unidepth-model` | UniDepthV2 depth |
| Prior-Depth-Anything checkpoints | `--prior-depth-checkpoint`, `--prior-depth-backbones` | LiDAR-completed depth |

The Prior-Depth-Anything backbone directory contains `depth_anything_v2_vitb.pth`; when no local directory is supplied, that frozen backbone is downloaded from `Rain729/Prior-Depth-Anything`. The conditioned prior checkpoint is `prior_depth_anything_vitb.pth`.

## 1. Extract raw Waymo

`--scene-ids` refer to the ordered segment list in `splits/segment_list_train.txt` or `segment_list_val.txt`.

```bash
python Preprocess/extract_waymo.py \
  --raw-root /path/to/Waymo/validation \
  --output-root /path/to/waymo_extracted/val \
  --split val --scene-ids 121
```

For a smoke check, add `--max-frames 8`. For predicted tracks, add `--track-file /path/to/castrack.json`; the JSON must contain the selected segment name.

For dynamic-object 3D bounding-box estimation and tracking, we follow [Street Gaussians](https://github.com/zju3dv/street_gaussians#preprocess-the-data). Refer to their preprocessing instructions for tracking predictions, then pass the resulting CasTrack JSON through `--track-file`.

## 2. Prepare a scene

Metric3D is the default depth route when an explicit checkpoint is supplied:

```bash
python Preprocess/prepare.py \
  --data-root /path/to/waymo_extracted/val \
  --output-root /path/to/evolsplat4d_data \
  --scene-id 121 --start-idx 0 --num-frames 60 \
  --mode dynamic --tracks gt --depth-model metric3d \
  --pca-path /path/to/dinov2_pca.joblib \
  --dino-repo /path/to/facebookresearch_dinov2_main \
  --metric3d-checkpoint /path/to/metric_depth_vit_giant2_800k.pth \
  --prior-python /path/to/EvolSplat4D-preprocess/bin/python
```

The other depth choices use the same command and replace only the depth arguments:

```bash
# UniDepthV2
--mode static --depth-model unidepth \
--unidepth-model /path/to/unidepth/v2_vitl14

# LiDAR + Prior-Depth-Anything
--mode dynamic --tracks predicted --depth-model lidar_depth \
--prior-depth-checkpoint /path/to/prior_depth_anything_vitb.pth \
--prior-depth-backbones /path/to/prior_depth_backbones
```

`--tracks gt` uses the extracted Waymo objects. `--tracks predicted` uses the extracted CasTrack files; for the LiDAR route, predicted object points come from LiDAR, while Metric3D and UniDepthV2 back-project their predicted depth maps. The default `--sparsities Drop50 Drop80` creates both point-cloud priors. Drop50 follows the existing training convention (every second frame); Drop80 follows the existing dynamic inference convention (every fifth frame).

The resulting directory is directly usable as a training scene or an inference scene:

```text
scene_121_000_060/
├── transforms.json
├── front_images/  feature_map/  semantic/  depth/
├── fg_mask/       input_pcd/Drop50/Static.npz
├── input_pcd/Drop80/Static.npz
└── input_pcd/track_info.pth       # dynamic scenes only
```

Run dynamic inference with `config/evolsplat4d_dynamic.yaml` and `drop80`; run static inference with `config/evolsplat4d_static.yaml` and `drop50`. The main repository's [`README.md`](../README.md) documents those commands.

## Licenses and data

The vendored model implementations retain their upstream license files. Waymo data and pretrained model weights remain subject to their respective terms. Do not commit large model checkpoints or generated scenes; pass their paths through the command-line options.
