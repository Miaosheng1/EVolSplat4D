"""Prepare Waymo scenes for EvolSplat4D training and feed-forward inference."""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from read_dataset.generate_waymo_pcd import WaymoPCDGenerator
from rich.console import Console
from tqdm import tqdm
from waymo.box_utils import bbox_to_corner3d, get_bound_2d_mask, inbbox_points
from waymo.objects import Objects
from waymo.track import get_obj_pose_tracking, inverse_rigid_trans
from waymo.waymo_sourceloader import WaymoLiDARSource, WaymoPixelSource
from waymo.waymo_utils import quaternion_to_matrix_numpy

CONSOLE = Console(width=120)
BILINEAR = Image.Resampling.BILINEAR
ROOT = Path(__file__).resolve().parent


class WaymoPreprocessor:
    bbx_min = np.array([-20, -9, -20])
    bbx_max = np.array([20, 3.8, 50])

    def __init__(self, args):
        self.args = args
        self.cam_id = 0
        self.downscale = args.downscale
        self.original_size = [1280, 1920]
        self.load_size = [size // self.downscale for size in self.original_size]
        self.H, self.W = self.load_size
        self.start_timestep = args.start_idx
        self.end_timestep = args.start_idx + args.num_frames
        self.dir_name = str(args.data_root / args.scene_id)
        self.save_dir = str(
            args.output_root / f"scene_{args.scene_id}_{self.start_timestep:03d}_{self.end_timestep:03d}"
        )
        self.input_pcd_dir = os.path.join(self.save_dir, "input_pcd")

    def run_prior(self, kind):
        command = [str(self.args.prior_python), str(ROOT / "infer_priors.py"), kind, "--scene", self.save_dir]
        options = {
            "dino": {"pca-path": self.args.pca_path, "dino-repo": self.args.dino_repo},
            "semantic": {"checkpoint": self.args.semantic_checkpoint},
            "metric3d": {"checkpoint": self.args.metric3d_checkpoint},
            "unidepth": {"checkpoint": self.args.unidepth_model},
            "lidar_depth": {
                "checkpoint": self.args.prior_depth_checkpoint,
                "backbone-dir": self.args.prior_depth_backbones,
            },
        }
        for name, value in options[kind].items():
            if value is not None:
                command += ["--" + name, str(value)]
        subprocess.run(command, check=True)

    def run(self):
        if not Path(self.dir_name).is_dir():
            raise FileNotFoundError(self.dir_name)
        for frame in range(self.start_timestep, self.end_timestep):
            for relative in [f"images/{frame:06d}_0.png", f"ego_pose/{frame:06d}.txt"]:
                if not (Path(self.dir_name) / relative).is_file():
                    raise FileNotFoundError(Path(self.dir_name) / relative)
        Path(self.save_dir).mkdir(parents=True, exist_ok=False)
        Path(self.input_pcd_dir).mkdir()
        self.load_calibrations()
        self.load_images()
        self.run_prior("dino")
        self.run_prior("semantic")
        if self.args.depth_model == "lidar_depth":
            self.gen_lidar_depth()
        self.run_prior(self.args.depth_model)
        if self.args.mode == "dynamic":
            if self.args.tracks == "gt":
                self.gen_init_objects()
            else:
                self.init_tracking()
        for sparsity in self.args.sparsities:
            generator = WaymoPCDGenerator(
                spars=sparsity, split_dynamic=self.args.mode == "dynamic", bbx_min=self.bbx_min, bbx_max=self.bbx_max
            )
            points = generator.forward(
                self.save_dir,
                self.cam_to_worlds.numpy(),
                self.intrinsics.numpy(),
                self.H,
                self.W,
                rgb_files=os.path.join(self.save_dir, "feature_map"),
                save_dir=self.input_pcd_dir,
            )
            if sparsity == "Drop50":
                self.gen_fg_mask(points)
        validate_scene(Path(self.save_dir), self.args.mode, self.args.sparsities)
        print(f"Prepared scene: {self.save_dir}")

    def gen_lidar_depth(self):
        ORIGINAL_SIZE = {
            "0": (1280, 1920),
            "1": (1280, 1920),
            "2": (1280, 1920),
            "3": (886, 1920),
            "4": (886, 1920),
        }
        downsample_factor = self.downscale

        pointcloud_path = os.path.join(self.dir_name, "pointcloud.npz")
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)["pointcloud"].item()

        for frame_idx in tqdm(range(self.start_timestep, self.end_timestep)):
            for cam_id in range(1):
                depth_path = f"{self.save_dir}/lidar_depth/{str(frame_idx).zfill(3)}_{str(cam_id)}.npy"
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                if not os.path.exists(depth_path):
                    frame_points = pts3d_dict[frame_idx]  # ego coordinate system
                    points = frame_points.astype(np.float32)

                    intrinsics_raw = np.loadtxt(os.path.join(self.dir_name, "intrinsics", f"{cam_id}.txt"))
                    fx, fy, cx, cy = intrinsics_raw[:4]
                    intrinsics = np.array(
                        [
                            [fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    )

                    cam_to_ego = np.loadtxt(os.path.join(self.dir_name, "extrinsics", f"{cam_id}.txt"))

                    ego_to_cam = np.linalg.inv(cam_to_ego)
                    target_size = (
                        ORIGINAL_SIZE[str(cam_id)][0] // downsample_factor,
                        ORIGINAL_SIZE[str(cam_id)][1] // downsample_factor,
                    )
                    _intrinsics = intrinsics.copy()
                    _intrinsics[0, 0] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 1] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    _intrinsics[0, 2] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 2] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    lidar2img = _intrinsics @ ego_to_cam
                    points_2d = (np.dot(lidar2img[:3, :3], points.T) + lidar2img[:3, 3:4]).T
                    depth_2d = points_2d[:, 2]
                    cam_coords = points_2d[:, :2] / (depth_2d[:, None] + 1e-6)
                    valid_mask = (
                        (cam_coords[:, 0] >= 0)
                        & (cam_coords[:, 0] < target_size[1])
                        & (cam_coords[:, 1] >= 0)
                        & (cam_coords[:, 1] < target_size[0])
                        & (depth_2d > 0)
                    )
                    valid_depth_points = depth_2d[valid_mask]
                    valid_cam_coords = cam_coords[valid_mask]
                    x_indices = valid_cam_coords[:, 0].astype(np.int32)
                    y_indices = valid_cam_coords[:, 1].astype(np.int32)
                    depth_sums = np.zeros(target_size)
                    depth_counts = np.zeros(target_size)

                    np.add.at(depth_sums, (y_indices, x_indices), valid_depth_points)
                    np.add.at(depth_counts, (y_indices, x_indices), 1)

                    depth_image = np.divide(depth_sums, depth_counts, where=depth_counts > 0)

                    depth_image[depth_counts == 0] = 0

                    print(
                        f"Frame {frame_idx}: Generated depth map with {valid_mask.sum()} projected points from {len(points)} total points"
                    )

                    np.save(depth_path, depth_image)

    def pose_normalization(self, poses):
        mid_frames = (self.end_timestep - self.start_timestep) // 2 - 2
        inv_pose = np.linalg.inv(poses[mid_frames])
        for i, pose in enumerate(poses):
            if i == mid_frames:
                poses[i] = np.eye(4)
            else:
                poses[i] = np.dot(inv_pose, poses[i])  ## Left-multiply by inv_pose.

        return poses, inv_pose

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        intrinsic = np.loadtxt(os.path.join(self.dir_name, "intrinsics", f"{self.cam_id}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        cam_to_ego = np.loadtxt(os.path.join(self.dir_name, "extrinsics", f"{self.cam_id}.txt"))

        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, distortions = [], []
        ego_frame_poses = []

        ego_to_world_start = np.loadtxt(os.path.join(self.dir_name, "ego_pose", f"{self.start_timestep:06d}.txt"))
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(self.dir_name, "ego_pose", f"{t:06d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        ego_pose_dir = os.path.join(self.dir_name, "ego_pose")
        ego_pose_paths = sorted(os.listdir(ego_pose_dir))
        for ego_pose_path in ego_pose_paths:
            if "_" not in ego_pose_path:
                ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
                ego_frame_poses.append(ego_frame_pose)

        ego_frame_poses = np.array(ego_frame_poses)

        cam_to_worlds, inv_pose = self.pose_normalization(cam_to_worlds)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.ego_frame_poses = ego_frame_poses
        self.cam_to_ego = cam_to_ego
        self.ego_to_world_start = np.linalg.inv(ego_to_world_start)
        self.inv_pose = inv_pose

    def load_images(self):
        self.img_filepaths = sorted(
            [f for f in glob.glob(os.path.join(self.dir_name, "images", "*.png")) if f.endswith("_0.png")]
        )[self.start_timestep : self.end_timestep]
        os.makedirs(os.path.join(self.save_dir, "front_images"), exist_ok=True)

        def listify_matrix(matrix):
            matrix_list = []
            if hasattr(matrix, "numpy"):
                matrix = matrix.numpy()
            for row in matrix:
                matrix_list.append([float(x) for x in row])
            return matrix_list

        self.rgbs = []
        out_data = {
            "fl_x": self.intrinsics[0][0][0].item(),
            "fl_y": self.intrinsics[0][1][1].item(),
            "cx": self.intrinsics[0][0][2].item(),
            "cy": self.intrinsics[0][1][2].item(),
            "w": self.W,
            "h": self.H,
            "Drop50_ply_file_path": "input_pcd/Drop50/Static.npz",
            "Drop80_ply_file_path": "input_pcd/Drop80/Static.npz",
            "bbx_min": self.bbx_min.tolist(),
            "bbx_max": self.bbx_max.tolist(),
            "track_file": "input_pcd/track_info.pth",
        }
        if self.args.mode == "static":
            out_data.pop("track_file")
        for sparsity in ["Drop50", "Drop80"]:
            if sparsity not in self.args.sparsities:
                out_data.pop(sparsity + "_ply_file_path")
        out_data["frames"] = []

        for ix, fname in tqdm(enumerate(self.img_filepaths), desc="Loading images"):
            rgb = Image.open(fname).convert("RGB")
            rgb = rgb.resize((self.load_size[1], self.load_size[0]), BILINEAR)
            rgb = cv2.undistort(
                np.array(rgb),
                self.intrinsics[ix].numpy(),
                self.distortions[ix].numpy(),
            )
            self.rgbs.append(rgb)
            rgb = Image.fromarray(rgb)

            if fname.endswith(".jpg"):
                fname = fname.replace(".jpg", ".png")

            rgb.save(os.path.join(self.save_dir, "front_images", Path(fname).name))
            img_name = os.path.join("front_images", Path(fname).name)
            frame_data = {
                "file_path": img_name,
                "transform_matrix": listify_matrix(self.cam_to_worlds[ix]),
                "dynamic_mask": f"dynamic_mask/{Path(fname).name}",
                "fg_mask": f"fg_mask/{Path(fname).name}",
                "depth": f"depth/{str(Path(fname).name).replace('.png', '.npy')}",
            }

            if self.args.mode == "static":
                frame_data.pop("dynamic_mask")
            out_data["frames"].append(frame_data)

        with open(f"{self.save_dir}/transforms.json", "w") as out_file:
            json.dump(out_data, out_file, indent=4)

    def init_tracking(self, min_pts_per_object=1000):
        """Initialize predicted moving tracks from LiDAR or monocular depth."""
        import open3d as o3d

        _, tracklets, objects = get_obj_pose_tracking(
            self.dir_name, [self.start_timestep, self.end_timestep - 1], self.ego_frame_poses, cameras=[0]
        )
        use_lidar = self.args.depth_model == "lidar_depth"
        if use_lidar:
            with np.load(Path(self.dir_name) / "pointcloud.npz", allow_pickle=True) as data:
                lidar = data["pointcloud"].item()
                projections = data["camera_projection"].item()
        mask_dir = Path(self.save_dir) / "dynamic_mask"
        mask_dir.mkdir()
        xyz = {key: [] for key in objects}
        colors = {key: [] for key in objects}
        for key, obj in objects.items():
            obj["poses"] = np.repeat(np.eye(4)[None], self.args.num_frames, axis=0)
            obj["frame_info"] = np.zeros(self.args.num_frames, dtype=bool)

        for i, image_path in enumerate(self.img_filepaths):
            frame = i + self.start_timestep
            K = self.intrinsics[i].numpy()
            bounds = np.zeros((self.H, self.W), dtype=bool)
            poses = {}
            for track in tracklets[i]:
                key = int(track[0])
                if key < 0:
                    continue
                obj = objects[key]
                pose = np.eye(4)
                pose[:3, :3] = quaternion_to_matrix_numpy(track[4:8])
                pose[:3, 3] = track[1:4]
                poses[key] = pose
                obj["poses"][i] = self.inv_pose @ self.ego_to_world_start @ self.ego_frame_poses[frame] @ pose
                obj["frame_info"][i] = True
                size = np.array([obj["length"], obj["width"], obj["height"]])
                corners = bbox_to_corner3d(np.stack([-size, size]) * 0.5)
                corners = corners @ pose[:3, :3].T + pose[:3, 3]
                bounds |= get_bound_2d_mask(corners, K, np.linalg.inv(self.cam_to_ego), self.H, self.W).astype(bool)
            Image.fromarray(bounds).save(mask_dir / Path(image_path).name)

            if use_lidar:
                projection = projections[frame]
                visible = projection[:, 0] == 0
                points_vehicle = lidar[frame][visible]
                u, v = projection[visible, 1], projection[visible, 2]
                raw_rgb = np.array(Image.open(image_path).convert("RGB")) / 255.0
                rgb = raw_rgb[v, u]
            else:
                depth = np.load(Path(self.save_dir) / "depth" / f"{frame:06d}_0.npy")
                labels = cv2.imread(str(Path(self.save_dir) / "semantic/instance" / Path(image_path).name), -1)
                mask = bounds & np.isin(labels, [13, 14, 15]) & np.isfinite(depth) & (depth > 0)
                v, u = np.nonzero(mask)
                z = depth[v, u]
                camera = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
                points_vehicle = camera @ self.cam_to_ego[:3, :3].T + self.cam_to_ego[:3, 3]
                # Sample the resized, undistorted image matching the depth and semantic pixels.
                rgb = self.rgbs[i][v, u] / 255.0
            for key, pose in poses.items():
                obj = objects[key]
                inverse = np.linalg.inv(pose)
                local = points_vehicle @ inverse[:3, :3].T + inverse[:3, 3]
                size = np.array([obj["length"], obj["width"], obj["height"]])
                selected = inbbox_points(local, bbox_to_corner3d(np.stack([-size, size]) * 0.5))
                xyz[key].append(local[selected])
                colors[key].append(rgb[selected])

        for key in list(objects):
            points = np.concatenate(xyz[key]) if xyz[key] else np.empty((0, 3))
            rgb = np.concatenate(colors[key]) if colors[key] else np.empty((0, 3))
            if len(points) < min_pts_per_object:
                del objects[key]
                continue
            if not use_lidar:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                _, indices = cloud.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.2)
                points, rgb = points[indices], rgb[indices]
                if len(points) > 25000:
                    indices = np.sort(np.random.choice(len(points), 25000, replace=False))
                    points, rgb = points[indices], rgb[indices]
            objects[key].update(pts=points, colors=rgb, num_pts=len(points))
        torch.save(objects, Path(self.input_pcd_dir) / "track_info.pth")

    def gen_fg_mask(self, pcd=None):
        mask_dir = os.path.join(self.save_dir, "fg_mask")
        os.makedirs(mask_dir, exist_ok=True)
        points = pcd  # [N, 3]
        poses = self.cam_to_worlds.numpy()
        Ks = self.intrinsics.numpy()
        img_filenames = self.img_filepaths
        H, W = self.H, self.W
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # [N, 4]

        for i, (pose, K, img_filename) in enumerate(zip(poses, Ks, img_filenames)):
            K = K[:3, :3]
            world2camera = inverse_rigid_trans(pose)
            points_cam = (world2camera @ points_homo.T).T  # [N, 4]
            points_cam = points_cam[:, :3]  # [N, 3]

            valid_mask = points_cam[:, 2] > 0
            points_cam = points_cam[valid_mask]

            if len(points_cam) == 0:
                mask = np.zeros((H, W), dtype=np.uint8)
            else:
                points_2d = (K @ points_cam.T).T  # [N, 3]
                points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # Normalize to [N, 2].

                mask = np.zeros((H, W), dtype=np.uint8)
                valid_points = (
                    (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
                )
                points_2d = points_2d[valid_points].astype(np.int32)

                if len(points_2d) > 0:
                    mask[points_2d[:, 1], points_2d[:, 0]] = 1

            mask_filename = Path(img_filename).name
            mask_path = os.path.join(mask_dir, mask_filename)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            cv2.imwrite(mask_path, mask * 255)

    def gen_init_objects(self):
        """Build GT boxes from per-frame LiDAR and dynamic masks."""
        assert os.path.exists(os.path.join(self.dir_name, "lidar")), "Lidar data is not found"
        assert os.path.exists(os.path.join(self.dir_name, "dynamic_mask")), "Dynamic mask data is not found"
        pixel_source = WaymoPixelSource(
            dataset_name="waymo",
            load_objects=True,
            data_path=self.dir_name,
            camera_list=[0],  ## forward camera
            start_timestep=self.start_timestep,
            end_timestep=self.end_timestep,
        )
        lidar_config = OmegaConf.create(
            {
                "lidar_downsample_factor": 4,
                "lidar_percentile": 0.02,
                "load_lidar": True,
                "only_use_top_lidar": False,
                "truncated_max_range": 80,
                "truncated_min_range": -2,
            }
        )
        lidar_source = WaymoLiDARSource(
            lidar_data_config=lidar_config,
            data_path=self.dir_name,
            start_timestep=self.start_timestep,
            end_timestep=self.end_timestep,
        )
        lidar_source.project_lidar_pts_on_images(pixel_source=pixel_source)

        Object = Objects(pixel_source, lidar_source=lidar_source)
        instance_dict = Object.get_init_objects(
            instance_max_pts=1000, only_moving=True, invpose=torch.from_numpy(self.inv_pose).float()
        )
        save_path = os.path.join(self.input_pcd_dir, "track_info.pth")
        torch.save(instance_dict, save_path)

        dynamic_mask_dir = os.path.join(self.save_dir, "dynamic_mask")
        os.makedirs(dynamic_mask_dir, exist_ok=True)
        for i in range(self.start_timestep, self.end_timestep):
            file_name = f"{i:06d}_0.png"
            dynamic_mask = cv2.imread(os.path.join(self.dir_name, "dynamic_mask", file_name), -1)
            dynamic_mask = cv2.resize(dynamic_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(dynamic_mask_dir, file_name), dynamic_mask)


def validate_scene(scene, mode, sparsities):
    """Check the file contract consumed by the EvolSplat4D data parsers."""
    meta = json.loads((scene / "transforms.json").read_text())
    shape = (meta["h"], meta["w"])
    for frame in meta["frames"]:
        name = Path(frame["file_path"]).stem
        features = torch.load(scene / "feature_map" / (name + ".pt"), map_location="cpu", weights_only=False)
        if features.shape != (*shape, 16) or not torch.isfinite(features).all():
            raise ValueError(f"Invalid DINO feature map: {name}")
        depth = np.load(scene / "depth" / (name + ".npy"))
        if depth.shape != shape or not np.isfinite(depth).all() or not (depth > 0).any():
            raise ValueError(f"Invalid depth: {name}")
        if "Drop50" in sparsities and not (scene / "fg_mask" / (name + ".png")).is_file():
            raise FileNotFoundError(f"Missing training foreground mask: {name}")
    for sparsity in sparsities:
        with np.load(scene / meta[sparsity + "_ply_file_path"]) as data:
            points, features = data["points"], data["features"]
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                raise ValueError(f"Empty or invalid {sparsity} point cloud")
            if features.shape != (len(points), 16) or not np.isfinite(points).all() or not np.isfinite(features).all():
                raise ValueError(f"Invalid {sparsity} point features")
    if mode == "dynamic":
        tracks = torch.load(scene / meta["track_file"], map_location="cpu", weights_only=False)
        if not tracks:
            raise ValueError("No moving objects survived filtering; use --mode static for static scenes")
        for item in tracks.values():
            if item["poses"].shape != (len(meta["frames"]), 4, 4) or len(item["frame_info"]) != len(meta["frames"]):
                raise ValueError("Track frames do not match the image sequence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted Waymo root containing scene IDs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True, help="Extracted scene directory, e.g. 121")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--mode", choices=["dynamic", "static"], required=True)
    parser.add_argument("--tracks", choices=["gt", "predicted"], default="gt")
    parser.add_argument("--depth-model", choices=["metric3d", "unidepth", "lidar_depth"], default="metric3d")
    parser.add_argument("--sparsities", nargs="+", choices=["Drop50", "Drop80"], default=["Drop50", "Drop80"])
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--pca-path", type=Path, required=True, help="The 16-D PCA fitted for the training checkpoint")
    parser.add_argument("--dino-repo", type=Path, help="Local DINOv2 repository; otherwise use torch.hub")
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=ROOT / "dataset_methods/nvi_sem/checkpoints/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth",
    )
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument(
        "--unidepth-model",
        default="lpiccinelli/unidepth-v2-vitl14",
        help="Local model directory or Hugging Face model ID",
    )
    parser.add_argument("--prior-depth-checkpoint", type=Path)
    parser.add_argument("--prior-depth-backbones", type=Path, help="Directory containing depth_anything_v2_vitb.pth")
    parser.add_argument(
        "--prior-python", type=Path, default=Path(sys.executable), help="Python in the isolated model environment"
    )
    args = parser.parse_args()
    if args.start_idx < 0 or args.num_frames < 4 or args.downscale < 1:
        parser.error("start-idx must be nonnegative; num-frames >= 4; downscale >= 1")
    if Path(args.scene_id).name != args.scene_id or args.scene_id in {".", ".."}:
        parser.error("scene-id must be a single directory name")
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root == args.data_root or args.output_root.is_relative_to(args.data_root):
        parser.error("output-root must be outside the read-only extracted dataset")
    required = [args.pca_path, args.semantic_checkpoint]
    if args.depth_model == "metric3d":
        if args.metric3d_checkpoint is None:
            parser.error("--metric3d-checkpoint is required for Metric3D")
        required.append(args.metric3d_checkpoint)
    for path in required:
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    np.random.seed(42)
    torch.manual_seed(42)
    WaymoPreprocessor(args).run()


if __name__ == "__main__":
    main()
