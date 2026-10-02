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
from tqdm import tqdm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class EvolSplat4DDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: EvolSplat4DDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    eval_mode: Literal["manner", "drop80", "drop50", "all"] = "manner"
    """
    The training parser uses manner: train on all frames and evaluate frame index 10 of each scene.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the static point-cloud prior and its DINO features."""
    pcd_ration: int = 1
    """ the downscale ration of input pointcloud """
    load_fg_mask: bool = True
    """whether or not to include loading of Vechile Box"""
    include_depth: bool = True
    """whether or not to include loading of Metric Depth"""
    include_dynamic_mask: bool = False
    """whether or not to include loading of Dynamic Mask"""
    num_scenes: int = 1
    """Number of training scenes to load."""
    camera_list: List[int] = field(default_factory=lambda: [0])


@dataclass
class EvolSplat4DDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: EvolSplat4DDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        train_path = sorted(os.listdir(Path(f"{self.config.data}")))
        assert self.config.num_scenes <= len(train_path), f"the number of scenes : {len(train_path)} is less than the number of scenes: {self.config.num_scenes}"
        train_path = train_path[:self.config.num_scenes]
        self.config.num_scenes = len(train_path)
        CONSOLE.log("[yellow] Load Scene: ", self.config.num_scenes)

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
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
        scene_dict = {}
        frame_offset = 0  # Cumulative frame offset.
        times = []

        for i in tqdm(range(self.config.num_scenes)):
            data_dir =  self.config.data / Path(train_path[i])
            meta = load_from_json(data_dir / "transforms.json")
         
            distort_fixed = False
         

            # sort the frames by fname
            fnames = []
            for frame in meta["frames"]:
                filepath = Path(frame["file_path"])
                fname = self._get_fname(filepath, data_dir)
                fnames.append(fname)
            inds = np.argsort(fnames)
            frames = [meta["frames"][ind] for ind in inds]

            frames_idx = np.arange(len(frames)) + frame_offset  # Apply offset.
            scene_dict[f"scene_{i:03d}"] = list(frames_idx)
            frame_offset += len(frames)  # Advance offset.
            times.append(np.arange(len(frames)))
        

            ## Read the input pointcloud; We use the Drop50% Pointcloud for pretraining
            if self.config.load_3D_points and split == "train":
                ply_file_path = data_dir / Path(meta["Drop50_ply_file_path"])
                sparse_points,aabb = self._load_3D_points(ply_file_path, ratio=self.config.pcd_ration,meta=meta)
                assert "track_file" in meta, "track_info not found in meta"
                insrance_file = os.path.join(data_dir,meta["track_file"])
                # insrance_file = os.path.join(data_dir,"instance_gt.pth")
                instance_dict = torch.load(insrance_file)
                if len(instance_dict)==0:
                    CONSOLE.print(f"*****************instance_dict is None for {insrance_file}****************")
           
                instance_dicts.append(instance_dict)
                seed_point.append(sparse_points)
                aabbs.append(aabb)

            for idx,frame in enumerate(frames):
                filepath = Path(frame["file_path"])
                fname = self._get_fname(filepath, data_dir)

                assert "fl_x" in meta, "fx not specified in meta"
                fx.append(float(meta["fl_x"]))
            
                assert "fl_y" in meta, "fy not specified in meta"
                fy.append(float(meta["fl_y"]))
            
                assert "cx" in meta, "cx not specified in meta"
                cx.append(float(meta["cx"]))
            
                assert "cy" in meta, "cy not specified in meta"
                cy.append(float(meta["cy"]))
            
                assert "h" in meta, "height not specified in meta"
                height.append(int(meta["h"]))
            
                assert "w" in meta, "width not specified in meta"
                width.append(int(meta["w"]))
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

                if self.config.load_fg_mask:
                    mask_filepath = Path('fg_mask') / Path(filepath).name
                    mask_fname = self._get_fname(mask_filepath, data_dir)
                    mask_filenames.append(mask_fname)

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

        if self.config.eval_mode == "manner":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            
            # Evaluate frame index 10 from each scene.
            i_eval = []
            for scene_key, scene_frames in scene_dict.items():
                eval_frame_idx = scene_frames[10]
                i_eval.append(eval_frame_idx)
            
            i_eval = sorted(i_eval)
            i_train = i_all
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
            print(f"train indices: {indices}")
        elif split in ["val", "test"]:
            indices = i_eval
            print(f"eval indices: {indices}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
      

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        times = torch.tensor(np.concatenate(times),dtype=torch.long)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]
        fx =  torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy =  torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx =  torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy =  torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type= CameraType.PERSPECTIVE,
            times=times,
            metadata={},
        )
 
        # reinitialize metadata for dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=None,
            instances_dicts=instance_dicts,
            scene_dict=scene_dict,
            dataparser_scale=1.0,
            metadata={
                "input_pnt": seed_point,
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "aabbs": aabbs,
                "fg_mask": mask_filenames,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path, ratio: int = 3,meta=None):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        # import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        # pcd = o3d.io.read_point_cloud(str(ply_file_path))
        # if no points found don't read in an initial point cloud
        pcd = np.load(ply_file_path)

        # points3D = np.asarray(pcd.points, dtype=np.float32)[::ratio,:]


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
    
    def get_aabb_box(self,meta=None):
        assert "bbx_min" in meta and "bbx_max" in meta, "bbx_min and bbx_max not found in meta"
        aabbb_min = torch.tensor(meta["bbx_min"],dtype=torch.float32)
        aabbb_max = torch.tensor(meta["bbx_max"],dtype=torch.float32)
        return aabbb_min, aabbb_max
    
