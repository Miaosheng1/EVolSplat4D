"""Build the Drop50/Drop80 static DINO point priors used by EvolSplat4D."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch


class WaymoPCDGenerator:
    def __init__(self, spars="Drop50", split_dynamic=False, bbx_min=None, bbx_max=None):
        if spars not in {"Drop50", "Drop80"}:
            raise ValueError(f"Invalid sparsity: {spars}")
        self.sparsity = spars
        self.split_dynamic = split_dynamic
        self.bbx_min = np.asarray(bbx_min)
        self.bbx_max = np.asarray(bbx_max)

    def forward(self, dir_name, poses, intrinsics, H, W, rgb_files, save_dir):
        scene = Path(dir_name)
        files = sorted((scene / "depth").glob("*_0.npy"))
        if len(files) != len(poses):
            raise ValueError("Depth files and camera frames do not match")
        stride, point_stride = (2, 5) if self.sparsity == "Drop50" else (5, 3)
        clouds = []
        # Preserve the original prior recipe: every 2nd/5th frame, excluding an off-stride final frame.
        for index in range(0, len(files), stride):
            name = files[index].stem
            depth = np.load(files[index])
            features = torch.load(Path(rgb_files) / (name + ".pt"), map_location="cpu", weights_only=False).numpy()
            labels = cv2.imread(str(scene / "semantic/instance" / (name + ".png")), cv2.IMREAD_UNCHANGED)
            if depth.shape != (H, W) or features.shape != (H, W, 16) or labels is None or labels.shape != (H, W):
                raise ValueError(f"Missing or misaligned depth/features/semantic labels: {name}")
            mask = (labels != 10) & np.isfinite(depth) & (depth > 0)
            if self.split_dynamic:
                dynamic = cv2.imread(str(scene / "dynamic_mask" / (name + ".png")), cv2.IMREAD_UNCHANGED)
                if dynamic is None or dynamic.shape != (H, W):
                    raise ValueError(f"Missing or misaligned dynamic mask: {name}")
                mask &= dynamic == 0
            y, x = np.nonzero(mask)
            z = depth[y, x]
            K = intrinsics[index]
            camera = np.column_stack(((x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z, np.ones(len(z))))
            world = (poses[index] @ camera.T).T[:, :3]
            clouds.append(np.concatenate((world, features[y, x]), axis=1))
        cloud = np.concatenate(clouds)
        points, features = cloud[:, :3], cloud[:, 3:]
        mask = ((points > self.bbx_min) & (points < self.bbx_max)).all(axis=1)
        points, features = points[mask], features[mask]
        if not len(points):
            raise ValueError("No static points remain inside the scene bounds")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, indices = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        points, features = points[indices][::point_stride], features[indices][::point_stride]
        target = Path(save_dir) / self.sparsity
        target.mkdir(exist_ok=True)
        np.savez(target / "Static.npz", points=points, features=features)
        return points
