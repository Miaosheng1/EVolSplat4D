import json
import logging
import os
from typing import List

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm, trange
from waymo.dataset_meta import DATASETS_CONFIG
from waymo.lidar_source import SceneLidarSource
from waymo.pixel_source import ScenePixelSource

logger = logging.getLogger()


# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
# waymo coordinate system: x front, y left, z up
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# Waymo Camera List:
# 0: front_camera
# 1: front_left_camera
# 2: front_right_camera
# 3: left_camera
# 4: right_camera
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4]


class WaymoCameraData:
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        cam_id: int,
        # the start timestep to load
        start_timestep: int = 0,
        # the end timestep to load
        end_timestep: int = None,
        # the size to load the images
        downscale_when_loading: float = 2.0,
        # the device to move the camera to
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset_name = dataset_name
        self.cam_id = cam_id
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.device = device

        self.cam_name = DATASETS_CONFIG[dataset_name][cam_id]["camera_name"]
        self.original_size = DATASETS_CONFIG[dataset_name][cam_id]["original_size"]
        self.load_size = [
            int(self.original_size[0] / downscale_when_loading),
            int(self.original_size[1] / downscale_when_loading),
        ]

        # Load the images, dynamic masks, sky masks, etc.
        self.create_all_filelist()
        self.load_calibrations()
        self.load_images()
        # if load_dynamic_mask:
        #     self.load_dynamic_masks()
        # if load_sky_mask:
        #     self.load_sky_masks()
        # self.to(self.device)
        self.downscale_factor = 1.0

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # load camera intrinsics
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        # ====!! we did not use distortion parameters for simplicity !!====
        # to be improved!!
        intrinsic = np.loadtxt(os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        # scale intrinsics w.r.t. load size
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

        # load camera extrinsics
        cam_to_ego = np.loadtxt(os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt"))
        # because we use opencv coordinate system to generate camera rays,
        # we need a transformation matrix to covnert rays from opencv coordinate
        # system to waymo coordinate system.
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        # cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:06d}.txt"))
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{t:06d}.txt"))
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            # transformation:
            #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

    def create_all_filelist(self):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        # ---- define filepaths ---- #
        img_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            img_filepaths.append(os.path.join(self.data_path, "images", f"{t:06d}_{self.cam_id}.png"))
            # dynamic_mask_filepaths.append(
            #     os.path.join(
            #         self.data_path, dynamic_mask_dir, "all", f"{t:03d}_{self.cam_id}.png"
            #     )
            # )
            # sky_mask_filepaths.append(
            #     os.path.join(self.data_path, "sky_masks", f"{t:03d}_{self.cam_id}.png")
            # )
            # # depth_filepaths.append(
            # #     os.path.join(self.data_path, "depth", f"{t:06d}_{self.cam_id}.npy")
            # # )
            # depth_filepaths.append(
            #     os.path.join(self.data_path, "depth", f"{t:03d}_{self.cam_id}.npy")
            # )
        self.img_filepaths = np.array(img_filepaths)
        # self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        # self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        # self.depth_filepaths = np.array(depth_filepaths)

    def set_unique_ids(self, unique_cam_idx: int, unique_img_idx: Tensor):
        """
        unique id is the compact order of the camera index and frame index
        for example camera idx is [0, 2, 4]
        the camera index is [0, 1, 2]
        """
        self.unique_cam_idx = unique_cam_idx
        self.unique_img_idx = unique_img_idx.to(self.device)

    def load_images(self):
        images = []
        for ix, fname in tqdm(
            enumerate(self.img_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.img_filepaths),
        ):
            rgb = Image.open(fname).convert("RGB")
            # resize them to the load_size
            rgb = rgb.resize((self.load_size[1], self.load_size[0]), Image.BILINEAR)
            # undistort the images
            if ix == 0:
                print("undistorting rgb")
                rgb = cv2.undistort(
                    np.array(rgb),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            images.append(rgb)
        # normalize the images to [0, 1]
        self.images = images = torch.from_numpy(np.stack(images, axis=0)) / 255

    def load_time(
        self,
        normalized_time: Tensor,
    ):
        self.normalized_time = normalized_time.to(self.device)

    @property
    def num_frames(self) -> int:
        return self.cam_to_worlds.shape[0]

    @property
    def HEIGHT(self) -> int:
        return self.load_size[0]

    @property
    def WIDTH(self) -> int:
        return self.load_size[1]

    def __len__(self):
        return self.num_frames


class WaymoPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        load_objects: bool,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        camera_list: List[int],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, load_objects, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.camera_list = camera_list
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()

        for idx, cam_id in enumerate(self.camera_list):
            logger.info(f"Loading camera {cam_id}")
            camera = WaymoCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                device=self.device,
                downscale_when_loading=2.0,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera

    def load_objects_info(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        instances_info_path = os.path.join(self.data_path, "objects", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "objects", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)

        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:06d}.txt"))
        for idx, (k, v) in enumerate(instances_info.items()):
            for frame_idx, obj_to_world, box_size in zip(
                v["frame_annotations"]["frame_idx"],
                v["frame_annotations"]["obj_to_world"],
                v["frame_annotations"]["box_size"],
            ):
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_idx, idx] = np.array(obj_to_world)
                instances_size[frame_idx, idx] = np.array(box_size)

        # get frame valid instances
        # shape (num_frames, num_instances)
        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
        for frame_idx, valid_instances in frame_instances.items():
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(instances_pose[self.start_timestep : self.end_timestep]).float()
        instances_size = torch.from_numpy(instances_size[self.start_timestep : self.end_timestep]).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        per_frame_instance_mask = torch.from_numpy(
            per_frame_instance_mask[self.start_timestep : self.end_timestep]
        ).bool()

        # filter out the instances that are not visible in selected frames
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]

        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id


class WaymoLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(os.path.join(self.data_path, "lidar", f"{t:03d}.bin"))
        self.lidar_filepaths = np.array(lidar_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        # Note that in the Waymo Open Dataset, the lidar coordinate system is the same
        # as the vehicle coordinate system
        lidar_to_worlds = []

        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:06d}.txt"))
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{t:06d}.txt"))
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(np.stack(lidar_to_worlds, axis=0)).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        # flow/ground info are used for evaluation only
        flows, flow_classes, grounds = [], [], []
        # in waymo, we simplify timestamps as the time indices
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True):
            # each lidar_info contains an Nx14 array
            # from left to right:
            # origins: 3d, points: 3d, flows: 3d, flow_class: 1d,
            # ground_labels: 1d, intensities: 1d, elongations: 1d, laser_ids: 1d
            lidar_info = np.memmap(
                self.lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            lidar_origins = torch.from_numpy(lidar_info[:, :3]).float()
            lidar_points = torch.from_numpy(lidar_info[:, 3:6]).float()
            lidar_ids = torch.from_numpy(lidar_info[:, 13]).float()
            lidar_flows = torch.from_numpy(lidar_info[:, 6:9]).float()
            lidar_flow_classes = torch.from_numpy(lidar_info[:, 9]).long()
            ground_labels = torch.from_numpy(lidar_info[:, 10]).long()
            # we don't collect intensities and elongations for now

            # select lidar points based on a truncated ego-forward-directional range
            # this is to make sure most of the lidar points are within the range of the camera
            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (lidar_points[:, 0] > self.data_cfg.truncated_min_range)
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            lidar_flows = lidar_flows[valid_mask]
            lidar_flow_classes = lidar_flow_classes[valid_mask]
            ground_labels = ground_labels[valid_mask]
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_origins = (self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T + self.lidar_to_worlds[t][:3, 3:4]).T
            lidar_points = (self.lidar_to_worlds[t][:3, :3] @ lidar_points.T + self.lidar_to_worlds[t][:3, 3:4]).T
            # scene flows are in the lidar coordinate system, so we need to rotate them
            lidar_flows = (self.lidar_to_worlds[t][:3, :3] @ lidar_flows.T).T
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            # we use time indices as the timestamp for waymo dataset
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)
            flows.append(lidar_flows)
            flow_classes.append(lidar_flow_classes)
            grounds.append(ground_labels)
            # we use time indices as the timestamp for waymo dataset
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)
        self.pre_dynamic_masks = torch.zeros_like(self.ranges).squeeze().bool()
        # becasue the flows here are velocities (m/s), and the fps of the lidar is 10,
        # we need to divide the velocities by 10 to get the displacements/flows
        # between two consecutive lidar scans
        self.flows = torch.cat(flows, dim=0) / 10.0
        self.flow_classes = torch.cat(flow_classes, dim=0)
        self.grounds = torch.cat(grounds, dim=0).bool()

        # the underscore here is important.
        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

    def to(self, device: torch.device):
        super().to(device)
        self.flows = self.flows.to(device)
        self.flow_classes = self.flow_classes.to(device)
        self.grounds = self.grounds.to(self.device)

    def project_lidar_pts_on_images(self, pixel_source):
        """
        Project the lidar points on the images and attribute the color of the nearest pixel to the lidar point.

        Args:
            delete_out_of_view_points: bool
                If True, the lidar points that are not visible from the camera will be removed.
        """
        for cam in pixel_source.camera_data.values():
            for frame_idx in tqdm(
                range(len(cam)),
                desc="Projecting lidar pts on images for camera {}".format(cam.cam_name),
                dynamic_ncols=True,
            ):
                normed_time = pixel_source.normalized_time[frame_idx]

                # get lidar depth on image plane
                closest_lidar_idx = self.find_closest_timestep(normed_time)
                lidar_infos = self.get_lidar_rays(closest_lidar_idx)
                lidar_points = (
                    lidar_infos["lidar_origins"] + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                )

                # project lidar points to the image plane
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    cam.intrinsics[frame_idx].cpu().numpy(),
                    cam.distortions[frame_idx].cpu().numpy(),
                    (cam.WIDTH, cam.HEIGHT),
                    alpha=1,
                )
                intrinsic_4x4 = torch.nn.functional.pad(torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)).to(
                    self.device
                )

                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                lidar_points = (lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]).T  # (num_pts, 3)

                depth = lidar_points[:, 2]
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)  # (num_pts, 2)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                )  # (num_pts, )
                _cam_points = cam_points[valid_mask]

                # used to filter out the lidar points that are visible from the camera
                visible_indices = torch.arange(self.num_points, device=self.device)[lidar_infos["lidar_mask"]][
                    valid_mask
                ]

                self.visible_masks[visible_indices] = True

                # attribute the color of the nearest pixel to the lidar point
                points_color = cam.images[frame_idx][_cam_points[:, 1].long(), _cam_points[:, 0].long()]
                self.colors[visible_indices] = points_color

        self.visible_masks = self.visible_masks
        self.delete_invisible_pts()
