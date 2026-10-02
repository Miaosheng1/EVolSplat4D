"""Run each pretrained prior in a separate process to release its GPU memory."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


def run(args):
    meta = json.loads((args.scene / "transforms.json").read_text())
    images = [args.scene / frame["file_path"] for frame in meta["frames"]]
    if args.kind == "dino":
        from dataset_methods.dinov2 import DinoFeatureExtractor

        model = DinoFeatureExtractor(IMG_H=meta["h"], IMG_W=meta["w"], repo=args.dino_repo)
        model.load_pca_model(args.pca_path)
        output = args.scene / "feature_map"
        output.mkdir()
        for image in tqdm(images, desc="DINOv2 + PCA"):
            with Image.open(image) as rgb:
                torch.save(model.transform(rgb.convert("RGB")), output / (image.stem + ".pt"))
        return
    if args.kind == "semantic":
        sys.path.insert(0, str(ROOT / "dataset_methods/nvi_sem"))
        from config import cfg

        cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
        cfg.MODEL.HRNET_CHECKPOINT = ""
        cfg.MODEL.N_SCALES = [1.0]
        cfg.OPTIONS.TORCH_VERSION = 2.5
        cfg.DATASET.NUM_CLASSES = 19
        from network.ocrnet import HRNet_Mscale

        model = HRNet_Mscale(num_classes=19, criterion=None).eval().cuda()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        from torchvision.transforms import Compose, Normalize, ToTensor

        transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        output = args.scene / "semantic/instance"
        output.mkdir(parents=True)
        for image in tqdm(images, desc="Cityscapes sky labels"):
            with Image.open(image) as rgb, torch.inference_mode():
                logits = model({"images": transform(rgb.convert("RGB")).unsqueeze(0).cuda()})["pred"]
            labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(labels).save(output / image.name)
        return
    output = args.scene / "depth"
    output.mkdir()
    K = np.array([[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1]], dtype=np.float32)
    if args.kind == "metric3d":
        sys.path.insert(0, str(ROOT / "dataset_methods/metric3d"))
        from mmcv import Config
        from mono.model.monodepth_model import get_configured_monodepth_model
        from mono.utils.do_test import get_prediction, transform_test_data_scalecano

        cfg = Config.fromfile(str(ROOT / "dataset_methods/metric3d/mono/configs/HourglassDecoder/vit.raft5.giant2.py"))
        model = get_configured_monodepth_model(cfg).cuda().eval()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        loaded = model.load_state_dict(state["model_state_dict"], strict=False)
        # These DINO parameters are unused by the giant model's multi-output inference path.
        unused = {"depth_model.encoder.mask_token", "depth_model.encoder.norm.weight", "depth_model.encoder.norm.bias"}
        if set(loaded.missing_keys) - unused or loaded.unexpected_keys:
            raise ValueError(f"Incompatible Metric3D checkpoint: {loaded}")
        del state
        model = torch.nn.DataParallel(model)
        for image in tqdm(images, desc="Metric3D"):
            rgb = cv2.imread(str(image))[:, :, ::-1].copy()
            tensor, cameras, pad, scale = transform_test_data_scalecano(
                rgb, [meta[k] for k in ["fl_x", "fl_y", "cx", "cy"]], cfg.data_basic
            )
            with torch.inference_mode():
                depth, _, _ = get_prediction(
                    model, tensor, cameras, pad, scale, None, cfg.data_basic.depth_range[1], list(rgb.shape[:2])
                )
            # The trained model predicts normalized depth; retain the original canonical conversion.
            np.save(output / (image.stem + ".npy"), depth.cpu().numpy())
    elif args.kind == "unidepth":
        sys.path.insert(0, str(ROOT / "dataset_methods/UniDepth-main"))
        from unidepth.models import UniDepthV2

        model = UniDepthV2.from_pretrained(args.checkpoint or "lpiccinelli/unidepth-v2-vitl14").cuda().eval()
        for image in tqdm(images, desc="UniDepthV2"):
            rgb = torch.from_numpy(np.array(Image.open(image).convert("RGB"))).permute(2, 0, 1)
            with torch.inference_mode():
                depth = model.infer(rgb, torch.from_numpy(K))["depth"].squeeze()
            np.save(output / (image.stem + ".npy"), depth.cpu().numpy())
    else:
        sys.path.insert(0, str(ROOT / "dataset_methods/Prior-Depth-Anything"))
        from prior_depth_anything import PriorDepthAnything

        model = PriorDepthAnything(
            device="cuda:0", fmde_dir=args.backbone_dir, cmde_dir=args.backbone_dir, ckpt_dir=args.checkpoint
        ).eval()
        for image in tqdm(images, desc="LiDAR + Prior Depth Anything"):
            frame_id = int(image.stem.split("_")[0])
            prior = args.scene / "lidar_depth" / f"{frame_id:03d}_0.npy"
            if not prior.is_file():
                raise FileNotFoundError(prior)
            with torch.inference_mode():
                depth = model.infer_one_sample(image=str(image), prior=str(prior))
            np.save(output / (image.stem + ".npy"), depth.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["dino", "semantic", "metric3d", "unidepth", "lidar_depth"])
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone-dir")
    parser.add_argument("--pca-path", type=Path)
    parser.add_argument("--dino-repo", type=Path)
    run(parser.parse_args())
