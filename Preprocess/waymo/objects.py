import logging

import torch

logger = logging.getLogger()


class Objects:
    def __init__(self, pixel_source, lidar_source):
        """ "" Config for the loading object"""
        # (num_frames, num_instances, 4, 4)
        self.instances_pose: torch.Tensor = pixel_source.instances_pose
        # (num_instances, 3)
        self.instances_size: torch.Tensor = pixel_source.instances_size
        # (num_frames, num_instances)
        self.per_frame_instance_mask: torch.Tensor = pixel_source.per_frame_instance_mask
        """Mask for the instances in each frame."""
        # (num_instances)
        self.instances_true_id: torch.Tensor = pixel_source.instances_true_id
        """Model id for the instances."""
        self.lidar_source = lidar_source
        self.frame_num = pixel_source.num_frames
        self.instance_dict = {}
        logger.info(f"Init the objects with {self.instances_pose.shape[1]} instances")

    @property
    def instance_num(self):
        return len(self.instances_pose[0])

    def get_init_objects(
        self,
        instance_max_pts: int = 1000,  ## Minimum point count for moving objects.
        only_moving: bool = True,
        traj_length_thres: float = 0.5,
        invpose: torch.Tensor = None,
    ):
        """
        return:
            instances_dict: Dict[int, Dict[str, Tensor]]
                keys: instance_id
                values: Dict[str, Tensor]
                    keys: "pts", "colors", "num_pts", "flows"(Optional)
                    values: Tensor

        NOTE: pts are in object coordinate system
        """
        instance_dict = {}
        for fi in range(self.frame_num):
            lidar_dict = self.lidar_source.get_lidar_rays(fi)
            lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
            for ins_id in range(self.instance_num):
                instance_active = self.per_frame_instance_mask[fi, ins_id]
                if not instance_active:
                    continue
                if ins_id not in instance_dict:
                    instance_dict[ins_id] = {
                        "pts": [],
                        "colors": [],
                        # "flows": [],
                    }
                # get the pose of the instance at the given frame
                o2w = self.instances_pose[fi, ins_id]
                o_size = self.instances_size[ins_id]
                # convert the lidar points to the instance's coordinate system
                w2o = torch.inverse(o2w)
                o_pts = self.transform_points(lidar_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                valid_pts = o_pts[mask]
                valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][mask]
                # valid_flows = lidar_dict["lidar_flows"][mask]
                instance_dict[ins_id]["pts"].append(valid_pts)
                instance_dict[ins_id]["colors"].append(valid_colors)
                # instance_dict[ins_id]["flows"].append(valid_flows)

        logger.info(f"Aggregating lidar points across {self.frame_num} frames")
        for ins_id in list(instance_dict.keys()):
            instance_dict[ins_id]["pts"] = torch.cat(instance_dict[ins_id]["pts"], dim=0)
            instance_dict[ins_id]["colors"] = torch.cat(instance_dict[ins_id]["colors"], dim=0)
            # instance_dict[ins_id]["flows"] = torch.cat(instance_dict[ins_id]["flows"], dim=0)
            instance_dict[ins_id]["num_pts"] = instance_dict[ins_id]["pts"].shape[0]

            if instance_dict[ins_id]["num_pts"] < 200:
                logger.info(f"Instance {ins_id} has {instance_dict[ins_id]['num_pts']} lidar sample points, skip it")
                del instance_dict[ins_id]
                continue

        if only_moving:
            # consider only the instances with non-zero flows
            logger.info("Filtering out the instances with non-moving trajectories")
            new_instance_dict = {}
            for k, v in instance_dict.items():
                if v["num_pts"] > instance_max_pts:
                    # flows = v["flows"]
                    # if flows.norm(dim=-1).mean() > moving_thres:
                    #     v.pop("flows")
                    #     new_instance_dict[k] = v
                    #     logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
                    frame_info = self.per_frame_instance_mask[:, k]
                    instances_pose = self.instances_pose[:, k]
                    instances_trans = instances_pose[:, :3, 3]
                    valid_trans = instances_trans[frame_info]
                    traj_length = valid_trans[1:] - valid_trans[:-1]
                    traj_length = torch.norm(traj_length, dim=-1).sum()
                    if traj_length > traj_length_thres:
                        new_instance_dict[k] = v
                        logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
            instance_dict = new_instance_dict

        # get instance info
        for ins_id in instance_dict:
            instance_dict[ins_id]["poses"] = invpose @ self.instances_pose[:, ins_id]
            instance_dict[ins_id]["size"] = self.instances_size[ins_id]
            instance_dict[ins_id]["frame_info"] = self.per_frame_instance_mask[:, ins_id]
            print(f"Instance {ins_id} has {instance_dict[ins_id]['num_pts']} lidar sample points")

        return instance_dict

    def transform_points(self, points, transform_matrix):
        """
        Apply a 4x4 transformation matrix to 3D points.

        Args:
            points: (N, 3) tensor of 3D points
            transform_matrix: (4, 4) transformation matrix

        Returns:
            (N, 3) tensor of transformed 3D points
        """
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
        homo_points = torch.cat([points, ones], dim=1)  # N x 4
        transformed_points = torch.matmul(homo_points, transform_matrix.T)
        return transformed_points[:, :3]
