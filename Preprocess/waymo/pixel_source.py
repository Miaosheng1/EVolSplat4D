import abc
import logging
from typing import Dict, Tuple

import torch
from torch import Tensor

logger = logging.getLogger()


class ScenePixelSource(abc.ABC):
    """
    The base class for all pixel sources of a scene.
    """

    # the dataset name, choose from ["waymo", "kitti", "nuscenes", "pandaset", "argoverse"]
    dataset_name: str = None
    # the dict of camera data
    _normalized_time: Tensor = None
    # timestep indices of frames, shape: (num_frames,)
    _timesteps: Tensor = None
    # the dict of camera data
    camera_data: Dict = {}

    # -------------- object annotations
    # (num_frame, num_instances, 4, 4)
    instances_pose: Tensor = None
    # (num_instances, 3)
    instances_size: Tensor = None
    # (num_instances, )
    instances_true_id: Tensor = None
    # (num_instances, )
    instances_model_types: Tensor = None
    # (num_frame, num_instances)
    per_frame_instance_mask: Tensor = None

    def __init__(self, dataset_name, load_objects, device: torch.device = torch.device("cpu")) -> None:
        # hold the config of the pixel data
        self.dataset_name = dataset_name
        self.load_objects = load_objects
        self.device = device

    @abc.abstractmethod
    def load_cameras(self) -> None:
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Load the images, dynamic masks, sky masks, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_objects_info(self) -> None:
        """
        Load the object annotations.
        """
        raise NotImplementedError

    def load_data(self) -> None:
        """
        A general function to load all data.
        """
        self.load_cameras()
        logger.info("[Pixel] All Pixel Data loaded.")

        if self.load_objects:
            self.load_objects_info()
            logger.info("[Pixel] All Object Annotations loaded.")

    def to(self, device: torch.device) -> "ScenePixelSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_time is not None:
            self._normalized_time = self._normalized_time.to(device)
        if self.instances_pose is not None:
            self.instances_pose = self.instances_pose.to(device)
        if self.instances_size is not None:
            self.instances_size = self.instances_size.to(device)
        if self.per_frame_instance_mask is not None:
            self.per_frame_instance_mask = self.per_frame_instance_mask.to(device)
        if self.instances_model_types is not None:
            self.instances_model_types = self.instances_model_types.to(device)
        return self

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            We compute the coarse aabb by using the front camera positions / trajectories. We then
            extend this aabb by 40 meters along horizontal directions and 20 meters up and 5 meters
            down along vertical directions.
        """
        front_camera_trajectory = self.front_camera_trajectory

        # compute the aabb
        aabb_min = front_camera_trajectory.min(dim=0)[0]
        aabb_max = front_camera_trajectory.max(dim=0)[0]

        # extend aabb by 40 meters along forward direction and 40 meters along the left/right direction
        # aabb direction: x, y, z: front, left, up
        aabb_max[0] += 40
        aabb_max[1] += 40
        # when the car is driving uphills
        aabb_max[2] = min(aabb_max[2] + 20, 20)

        # for waymo, there will be a lot of waste of space because we don't have images in the back,
        # it's more reasonable to extend the aabb only by a small amount, e.g., 5 meters
        # we use 40 meters here for a more general case
        aabb_min[0] -= 40
        aabb_min[1] -= 40
        # when a car is driving downhills
        aabb_min[2] = max(aabb_min[2] - 5, -5)
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Pixel] Auto AABB from camera: {aabb}")
        return aabb

    @property
    def front_camera_trajectory(self) -> Tensor:
        """
        Returns:
            the front camera trajectory.
        """
        front_camera = self.camera_data[0]
        assert front_camera.cam_to_worlds is not None, (
            "Camera poses not loaded, cannot compute front camera trajectory."
        )
        return front_camera.cam_to_worlds[:, :3, 3]

    def parse_img_idx(self, img_idx: int) -> Tuple[int, int]:
        """
        Parse the image index to the camera index and frame index.
        Args:
            img_idx: the image index.
        Returns:
            cam_idx: the camera index.
            frame_idx: the frame index.
        """
        unique_cam_idx = img_idx % self.num_cams
        frame_idx = img_idx // self.num_cams
        return unique_cam_idx, frame_idx

    def get_image(self, img_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given image index.
        Args:
            img_idx: the image index.
        Returns:
            a dict containing the rays for rendering the given image index.
        """
        unique_cam_idx, frame_idx = self.parse_img_idx(img_idx)
        for cam_id in self.camera_list:
            if unique_cam_idx == self.camera_data[cam_id].unique_cam_idx:
                return self.camera_data[cam_id].get_image(frame_idx)

    @property
    def num_frames(self) -> int:
        """
        Returns:
            the number of frames in the dataset
        """
        return len(self._timesteps)

    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of image timesteps in the dataset
        """
        return len(self._timesteps)

    @property
    def num_imgs(self) -> int:
        """
        Returns:
            the number of images in the dataset
        """
        return self.num_cams * self.num_frames

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of all images,
            shape: (num_imgs,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of images
            while timesteps are the integer timestep indices of images.
        """
        return self._timesteps

    @property
    def normalized_time(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of all images
            (normalized to the range [0, 1]),
            shape: (num_imgs,)
        """
        return self._normalized_time

    def register_normalized_timestamps(self) -> None:
        # normalized timestamps are between 0 and 1
        normalized_time = (self._timesteps - self._timesteps.min()) / (self._timesteps.max() - self._timesteps.min())

        self._normalized_time = normalized_time.to(self.device)
        self._unique_normalized_timestamps = self._normalized_time.unique()

    def find_closest_timestep(self, normed_timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        return torch.argmin(torch.abs(self._normalized_time - normed_timestamp))
