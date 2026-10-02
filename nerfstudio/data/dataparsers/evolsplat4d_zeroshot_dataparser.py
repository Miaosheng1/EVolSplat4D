# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class EvolSplat4DZeroShotDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: EvolSplat4DZeroShotDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    eval_mode: Literal["manner", "drop80", "drop50", "all"] = "drop50"
    """
    Use drop50 or drop80 to select context frames and evaluate the remaining frames;
    all uses every frame for both splits.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = True
    """Whether to load the static point-cloud prior and its DINO features."""
    use_dynamic: bool = True
    """Load object tracks; disable alongside the model's use_dynamic for static scenes."""
    pcd_ration: int = 1
    """ the downscale ration of input pointcloud """
    load_sky_mask: bool = False
    """whether or not to include loading of Sky Mask"""
    include_dynamic_mask: bool = False
    """whether or not to include loading of Dynamic Mask"""
    include_depth: bool = True
    """whether or not to include loading of Metric Depth"""
    num_scenes: int = 1
    """Number of scenes to load; inference processes one scene at a time."""
    start_timestep: int = 0
    """Start Timestep"""
    num_images_per_scene: int = 60
    """Number of Images per Scene"""
    camera_list: List[int] = field(default_factory=lambda: [0])
    interpolate_pose: bool = False
    """Whether to interpolate the pose"""
@dataclass
class EvolSplat4DZeroShotDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: EvolSplat4DZeroShotDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.num_scenes > 1:
            self.config.num_scenes = 1
        CONSOLE.log("[yellow] Load Scene: ", self.config.num_scenes)

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        dynamic_masks = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        seed_point = []
        aabbs = []
        instance_dicts = []
        
     
        data_dir =  self.config.data 
        meta = load_from_json(data_dir / "transforms.json")
        
        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        ## Read the input pointcloud; Drop50 is the densest available prior in this dataset layout.
        if self.config.load_3D_points and split == "train":
            if self.config.eval_mode == "drop80":
                ply_file_path = data_dir / Path(meta["Drop80_ply_file_path"])
            elif self.config.eval_mode == "drop50" or self.config.eval_mode == "all" or self.config.interpolate_pose:
                ply_file_path = data_dir / Path(meta["Drop50_ply_file_path"])
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            

            sparse_points,aabb = self._load_3D_points(ply_file_path, ratio=self.config.pcd_ration,meta=meta)
            instance_dict = {}
            if self.config.use_dynamic:
                assert "track_file" in meta, "track_info not found in meta"
                insrance_file = os.path.join(data_dir,meta["track_file"])
                if os.path.exists(insrance_file):
                    instance_dict = torch.load(insrance_file)
                else:
                    insrance_file = os.path.join(data_dir,"instance_gt.pth")
                    instance_dict = torch.load(insrance_file)
        
            instance_dicts.append(instance_dict)
            seed_point.append(sparse_points)
            aabbs.append(aabb)

        for idx,frame in enumerate(frames):
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))



        if self.config.include_depth:
            depth_filepath = data_dir/Path('feature_map')
            depth_filenames += [os.path.join(depth_filepath, f) for f in sorted(os.listdir(depth_filepath))]


                

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
 
        if self.config.interpolate_pose:
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            i_train = i_all
            i_eval = i_all
        elif self.config.eval_mode == "drop50":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            i_train = np.array([i for i in range(num_images)if (i % 2 == 0) or i == num_images-1])
            
            i_eval = np.setdiff1d(i_all, i_train)[:-1]
        elif self.config.eval_mode == "drop80":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            i_train = np.array([i for i in range(num_images)if (i % 5 == 0) or i == num_images-1])
            i_eval = np.setdiff1d(i_all, i_train)[:-1]
        elif self.config.eval_mode == "all":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            i_train = i_all
            i_eval = i_all
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
            CONSOLE.log(f"Train View {self.config.eval_mode}:  {indices}\n" + f"Train View Num: {len(i_train)}")
        elif split in ["val", "test"]:
            indices = i_eval
            CONSOLE.log(f"Test View {self.config.eval_mode}: {indices}\n" + f"Test View Num: {len(i_eval)}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        dynamic_masks = [dynamic_masks[i] for i in indices] if len(dynamic_masks) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        times = torch.tensor(indices,dtype=torch.long) % self.config.num_images_per_scene
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]



        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={},
        )
 
        # reinitialize metadata for dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            dataparser_scale=1.0,
            instances_dicts=instance_dicts,
            # dynamic_masks=dynamic_masks,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "input_pnt": seed_point,
                "aabbs": aabbs,
            },
        )
       
        
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, ratio: int = 1,meta=None):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        pcd = np.load(ply_file_path,allow_pickle=True)
        points3D = torch.from_numpy(pcd['points'])[::ratio,:]
        points3D_rgb = torch.from_numpy(pcd['features'])[::ratio,:]
        if len(points3D) == 0:
            return None

        Boundingbox_min, Boundingbox_max = self.get_aabb_box(meta)

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }

        return out,torch.stack([Boundingbox_min,Boundingbox_max],dim=0)

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file."""
        return data_dir / filepath
    

    def crop_pointcloud(self,bbx_min, bbx_max, points, color):
        mask = (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) & \
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) & \
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])
        return points[mask], color[mask]
    
    
    def get_aabb_box(self,meta=None):
        assert "bbx_min" in meta and "bbx_max" in meta, "bbx_min and bbx_max not found in meta"
        aabbb_min = torch.tensor(meta["bbx_min"],dtype=torch.float32)
        aabbb_max = torch.tensor(meta["bbx_max"],dtype=torch.float32)
        return aabbb_min, aabbb_max
