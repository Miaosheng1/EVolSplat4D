# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import fpsample
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track
from torch.nn import Parameter
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.datamanagers.utils import (
    eval_source_images_from_current_imageid,
    get_source_images_from_current_imageid,
    read_mask_filename,
    read_rgb_filename,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.evolsplat4d_dataparser import EvolSplat4DDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

logger = logging.getLogger()
@dataclass
class EvolSplat4DDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: EvolSplat4DDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=EvolSplat4DDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    """Specifies which sampling strategy is used to generate train cameras, 'random' means sampling 
    uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifacts 
    due to oversampling subsets of cameras that are very close to each other."""
    train_cameras_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    fps_reset_every: int = 100
    """The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every
    samples from the pool of all training cameras without replacement before a new round of sampling starts."""
    num_source_image: int = 3
    """The number of source images from the near poses."""


class EvolSplat4DDataManager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: EvolSplat4DDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: EvolSplat4DDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset() # type: ignore
        self.eval_dataset = self.create_eval_dataset() # type: ignore
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = self.sample_train_cameras()
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
    
        
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"
        super().__init__()
 

    def sample_train_cameras(self):
        """Return a list of camera indices sampled using the strategy specified by
        self.config.train_cameras_sampling_strategy"""
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            if not hasattr(self, "random_generator"):
                self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            self.random_generator.shuffle(indices)
            return indices
        elif self.config.train_cameras_sampling_strategy == "fps":
            if not hasattr(self, "train_unsampled_epoch_count"):
                np.random.seed(self.config.train_cameras_sampling_seed)  # fix random seed of fpsample
                self.train_unsampled_epoch_count = np.zeros(num_train_cameras)
            camera_origins = self.train_dataset.cameras.camera_to_worlds[..., 3].numpy()
            # We concatenate camera origins with weighted train_unsampled_epoch_count because we want to
            # increase the chance to sample camera that hasn't been sampled in consecutive epochs previously.
            # We assume the camera origins are also rescaled, so the weight 0.1 is relative to the scale of scene
            data = np.concatenate(
                (camera_origins, 0.1 * np.expand_dims(self.train_unsampled_epoch_count, axis=-1)), axis=-1
            )
            n = self.config.fps_reset_every
            if num_train_cameras < n:
                CONSOLE.log(
                    f"num_train_cameras={num_train_cameras} is smaller than fps_reset_ever={n}, the behavior of "
                    "camera sampler will be very similar to sampling random without replacement (default setting)."
                )
                n = num_train_cameras
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(data, n, h=3)

            self.train_unsampled_epoch_count += 1
            self.train_unsampled_epoch_count[kdline_fps_samples_idx] = 0
            return kdline_fps_samples_idx.tolist()
        else:
            raise ValueError(f"Unknown train camera sampling strategy: {self.config.train_cameras_sampling_strategy}")

    
    def sample_camId_from_multiscene(self,scene_id,all_pose,train_cameras):
        """
        Return:
            train_poses: ALl cameras pose of Curent Scene
            train_cameras: all cameras of Curent Scene
        
        """  
        cur_scene_indices = self.train_dataset.scene_dict[f"scene_{scene_id.item():03d}"]
        start_index = cur_scene_indices[0]
        end_index = cur_scene_indices[-1]
        image_index = random.choice(cur_scene_indices) - start_index
        # image_index = 54

        train_poses = all_pose[start_index:end_index+1,:,:]
        cur_image_filenames = self.train_dataset.image_filenames[start_index:end_index+1]
        cur_depth_filenames = self.train_dataset.metadata['depth_filenames'][start_index:end_index+1]
        cur_fg_mask_filenames = self.train_dataset.metadata['fg_mask'][start_index:end_index+1]

        return image_index, cur_image_filenames, train_poses, train_cameras[start_index:end_index+1], cur_fg_mask_filenames,cur_depth_filenames
       

    @cached_property
    def cached_train(self):
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) :
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("eval", cache_images_device=self.config.cache_images)

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ):
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            # data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            data = {}
            dataset.cameras[idx].reshape(())
      
            # if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)

        return undistorted_images
         

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[EvolSplat4DDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is EvolSplat4DDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is EvolSplat4DDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is EvolSplat4DDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch"""
        down_scale = 1.0
        ## Sample random sceneId for evolsplat4d
        scene_id = torch.randint(self.dataparser.config.num_scenes,size=(1,))
        # scene_id = torch.tensor([16])

        ## Load RGB and depth for this scene only.
        _ret = self.sample_camId_from_multiscene(scene_id=scene_id,
                                                 all_pose=self.train_dataset.cameras.camera_to_worlds,
                                                 train_cameras=self.train_dataset.cameras,
                                                 )
        image_idx, cur_rgbs, train_pose, train_cameras = _ret[:4]
        cur_fg_mask_filenames = _ret[4] if len(_ret) > 4 else None
        depth_filenames = _ret[5] if len(_ret) > 5 else None
        # image_idx = 10 
        camera = train_cameras[image_idx : image_idx + 1].to(self.device)
      
        tar_image = read_rgb_filename(image_filename=cur_rgbs[image_idx],down_scale = down_scale)

        ## select the nearest camera pose
        src_poses, target_pose, source_ids, src_depths = get_source_images_from_current_imageid(image_id = image_idx,
                                                                                                depths=depth_filenames,
                                                                                                all_pose = train_pose.to(self.device),
                                                                                                num_select=self.config.num_source_image,
                                                                                        )
        source_images = [read_rgb_filename(image_filename=cur_rgbs[i],down_scale = down_scale) for i in source_ids]
        # source_dynamic_masks = [read_mask_filename(mask_filename=dynamic_mask_filenames[i],down_scale = down_scale) for i in source_ids]
        source_images = torch.stack(source_images,dim=0)
        # source_dynamic_masks = torch.stack(source_dynamic_masks,dim=0)
        intrinsics = camera.get_normal_intrinsics_matrices().cuda()

        # source_timestamp = [train_cameras[i:i+1].times.item() % self.num_images_per_scene for i in source_ids]
        source_indices = [train_cameras[i:i+1].times.item() for i in source_ids]
        cur_timestamp = camera.times[0].cpu()

        mask_image = read_mask_filename(mask_filename=cur_fg_mask_filenames[image_idx],down_scale=1).cuda() if cur_fg_mask_filenames is not None else None
         ## apply downsample
        # tar_image, source_images, intrinsics, _ ,_ = self.apply_downsample(target_image=tar_image, 
        #                                                               source_image=source_images, 
        #                                                               intrinsics=intrinsics, 
        #                                                               depth=src_depths,
        #                                                               scale_factor=0.5)
        

        example = {
                    "source": {
                        "extrinsics": src_poses,
                        "intrinsics": intrinsics.repeat(self.config.num_source_image,1,1),
                        "image": source_images.cuda(),
                        "dino_feat": src_depths,
                        "source_timestamp":source_indices,
                    },
                    "target": {
                        "extrinsics": target_pose,
                        "intrinsics": intrinsics,
                        "image": tar_image.unsqueeze(0).cuda(),
                    },
                    "scene_id":scene_id,
                    "image_index": torch.cat([cur_timestamp,torch.tensor(source_indices)]),
                    "mask":mask_image
                }
        return camera, example # type: ignore

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch
            Each scene have 1 test views
        Returns a Camera instead of raybundle

        For the eval setting, the image_idx is the global index in the eval dataset"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        ## Visualize training images only.
        scene_id = torch.tensor(image_idx // 1)
        if self.dataparser.config.num_scenes == 1:
            scene_id = torch.tensor(0)
        _ret = self.sample_camId_from_multiscene(scene_id=scene_id,
                                                 all_pose=self.train_dataset.cameras.camera_to_worlds,
                                                 train_cameras=self.train_dataset.cameras,
                                                 )
        _, cur_rgbs, train_pose, train_cameras = _ret[:4]
        depth_filenames = _ret[5] if len(_ret) > 5 else None
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)


        tar_image = read_rgb_filename(image_filename=self.eval_dataset.image_filenames[image_idx],down_scale = 1.0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        src_poses, source_ids, src_depths = eval_source_images_from_current_imageid(rgbs=cur_rgbs,
                                                                                    depths=depth_filenames,
                                                                                    all_pose = train_pose.to(self.device),
                                                                                    num_select=self.config.num_source_image,
                                                                                    eval_pose=camera.camera_to_worlds,
                                                                                    tar_id=int(camera.times.item())
                                                                                    )
        source_images = [read_rgb_filename(image_filename=cur_rgbs[i],down_scale = 1.0) for i in source_ids]
        source_images = torch.stack(source_images,dim=0)
   
        eye = torch.tensor([0., 0., 0., 1.]).to(self.device).unsqueeze(0)
        target_pose = torch.cat([camera.camera_to_worlds,eye.unsqueeze(0)],dim=1)

        intrinsics = camera.get_normal_intrinsics_matrices().cuda()
        source_timestamp = [train_cameras[i:i+1].times.item() for i in source_ids]


        example = {
                    "source": {
                        "extrinsics": src_poses,
                        "intrinsics": intrinsics.repeat(self.config.num_source_image,1,1),
                        "image": source_images.cuda(),
                        "dino_feat": src_depths,
                        "source_timestamp":source_timestamp,
          
                    },
                    "target": {
                        "extrinsics": target_pose,
                        "intrinsics": intrinsics,
                        "image": tar_image.unsqueeze(0).cuda(),
                    },
                     "scene_id": scene_id,
                }
        
        return camera, example # type: ignore
    
    ## For feedforward inference on a novel scene
    def next_fixed_eval_image(self, image_idx: int) -> Tuple[Cameras, Dict]:
        """ return a assigned cameraID 3D GS outputs, For feedfoward-inference"""

        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        # self.num_images_per_scene = len(self.train_dataset)
        # _ret = self.sample_camId_from_multiscene(scene_id=scene_id,
        #                                          all_pose=self.train_dataset.cameras.camera_to_worlds,
        #                                          train_cameras=self.train_dataset.cameras,
        #                                          )
        cur_rgbs = self.train_dataset.image_filenames
        train_pose = self.train_dataset.cameras.camera_to_worlds
        depth_filenames = self.train_dataset.metadata['depth_filenames']

        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        tar_image = read_rgb_filename(image_filename=self.eval_dataset.image_filenames[image_idx],down_scale = 1.0)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"

        src_poses, source_ids, src_depths = eval_source_images_from_current_imageid(rgbs=cur_rgbs,
                                                                                    depths=depth_filenames,
                                                                                    all_pose = train_pose.to(self.device),
                                                                                    num_select=self.config.num_source_image,
                                                                                    eval_pose=camera.camera_to_worlds,
                                                                                    tar_id=-1
                                                                                    )
        source_images = [read_rgb_filename(image_filename=cur_rgbs[i],down_scale = 1.0) for i in source_ids]
        source_images = torch.stack(source_images,dim=0)

        eye = torch.tensor([0., 0., 0., 1.]).to(self.device).unsqueeze(0)
        target_pose = torch.cat([camera.camera_to_worlds,eye.unsqueeze(0)],dim=1)

        intrinsics = camera.get_normal_intrinsics_matrices().cuda()
        source_timestamp = [self.train_dataset.cameras[i:i + 1].times.item() for i in source_ids]


        example = {
                    "source": {
                        "extrinsics": src_poses,
                        "intrinsics": intrinsics.repeat(self.config.num_source_image,1,1),
                        "image": source_images.cuda(),
                        "dino_feat": src_depths,
                        "source_id":source_ids,
                        "source_timestamp": source_timestamp,
                    },
                    "target": {
                        "extrinsics": target_pose,
                        "intrinsics": intrinsics,
                        "image": tar_image.unsqueeze(0).cuda(),
                    },
                     "scene_id": torch.tensor([0]).to(self.device)
                }
        return camera, example
    

    


    def apply_downsample(self, target_image=None, source_image=None, intrinsics=None, depth=None,scale_factor=0.5,dymask=None):
        """
        Apply downsample to the image, intrinsics, and depth
        """
        if target_image is not None:
            target_image = F.interpolate(
                target_image.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
                scale_factor=scale_factor, mode='bilinear', align_corners=True
            ).squeeze(0).permute(1, 2, 0)  # Restore HWC layout.
        
        # Reshape and downsample source images.
        source_image = F.interpolate(
            source_image.permute(0, 3, 1, 2),  # (B, C, H, W)
            scale_factor=scale_factor, mode='bilinear', align_corners=True
        ).permute(0, 2, 3, 1)  # Restore BHWC layout.
        
        if depth is not None:
            depth = F.interpolate(
                depth.unsqueeze(1), 
                scale_factor=scale_factor, mode='bilinear', align_corners=True
            ).squeeze()  

        if dymask is not None:
            dymask = F.interpolate(
                dymask.unsqueeze(0),  # (1, 1, H, W)
                scale_factor=scale_factor, mode='bilinear', align_corners=True
            ).squeeze(0)  # Restore HW layout.
        
        # Scale intrinsics.
        intrinsics = intrinsics.clone()
        intrinsics[:, :2, :3] *= scale_factor  # Scale focal lengths and principal point.
        
        return target_image, source_image, intrinsics, depth,dymask
