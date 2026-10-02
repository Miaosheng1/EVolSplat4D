from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from pytorch3d.transforms import matrix_to_quaternion
from torch import nn
from torch.nn import Parameter
from tqdm import tqdm

from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.projection import Projector
from nerfstudio.models.basics import interpolate_quats, k_nearest_sklearn


class RigidNodes(nn.Module):
    def __init__(
        self,
        num_scenes: int = 1,
        instances_dicts: Dict[str, Any] = None,
        interpolate_pose: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.step = 0

        self._means = [] # type: ignore
        self.init_scales = [] # type: ignore
        self.instances_fv = []
        self.instances_size = []
        self.point_ids = []
        self.instances_quats = []
        self.instances_trans = []
        self.interpolate_pose = interpolate_pose

        self.device = device
        self.in_test_set = False
        self.instances_dicts = instances_dicts
        self.projector = Projector()
        self.dim_in = 3*9*2
        ## geometry decoder
        self.rigid_decoder = MLP(
            in_dim=self.dim_in,
            num_layers=2,
            layer_width=128,
            out_dim=11,  ## opacity(1) +scale(3) +quat(4) + color(3)
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        )

        ## Initialize each scene.
        for i in tqdm(range(num_scenes),desc=f"Creating {num_scenes} rigid nodes"):
            self.create_from_pcd(self.instances_dicts[i],scene_id=i)

    def get_pts_valid_mask(self,scene_id: int):
        """
        get the mask for valid points
        """
        return self.instances_fv[scene_id][int(self.cur_frame)][self.point_ids[scene_id][..., 0]]
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor],scene_id: int) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        # collect all instances
        init_means = []
        instances_pose = []
        instances_fv = []
        point_ids = []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            if isinstance(v["pts"],np.ndarray):
                init_means.append(torch.from_numpy(v["pts"]).to(self.device).float())
            else:
                init_means.append(v["pts"])
            if isinstance(v["poses"],np.ndarray):
                instances_pose.append(torch.from_numpy(v["poses"]).unsqueeze(1).float())
            else:
                instances_pose.append(v["poses"].unsqueeze(1))
            if isinstance(v["frame_info"],np.ndarray):
                instances_fv.append(torch.from_numpy(v["frame_info"]).unsqueeze(1))
            else:
                instances_fv.append(v["frame_info"].unsqueeze(1))
            point_ids.append(torch.full((v["num_pts"], 1), id_in_model, dtype=torch.long))
            
        init_means = torch.cat(init_means, dim=0).to(self.device) # (N, 3)
        
        self._means.append(init_means)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device) # (num_frame, num_instances, 4, 4)
        self.instances_fv.append(torch.cat(instances_fv, dim=1).to(self.device)) # (num_frame, num_instances)
        self.point_ids.append(torch.cat(point_ids, dim=0).to(self.device))
        instances_quats = self.get_instances_quats(instances_pose,scene_id=scene_id)
        instances_trans = instances_pose[..., :3, 3]
        
        # Initialize scales from point spacing.
        distances, _ = k_nearest_sklearn(init_means, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self.init_scales.append(torch.log(avg_dist.repeat(1, 3)))
        
        # pose refinement
        self.instances_quats.append(self.quat_act(instances_quats)) # (num_frame, num_instances, 4)
        self.instances_trans.append(instances_trans)            # (num_frame, num_instances, 3)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["rigid_decoder"] = list(self.rigid_decoder.parameters())
        return param_groups
    
    def get_instances_quats(self, instances_pose: torch.Tensor, scene_id: int) -> torch.Tensor:
        """
        Convert the pose to quaternion for all frames and instances
        """
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames*num_instances, 4, device=self.device)
        
        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = self.instances_fv[scene_id].view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)
        
        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def get_scaling(self,scales: torch.Tensor,scene_id: int):
        return torch.exp(scales + self.init_scales[scene_id])

    def transform_means(self, means: torch.Tensor,scene_id: int) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids[scene_id].shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        num_frames = self.instances_fv[scene_id].shape[0]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < num_frames
        ):
            # use the previous and next frame to interpolate the pose
            _quats_prev_frame = self.instances_quats[scene_id][self.cur_frame - 1]
            _quats_next_frame = self.instances_quats[scene_id][self.cur_frame + 1]
            _quats_cur_frame = self.instances_quats[scene_id][self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame,fraction=0.5)
            
            inter_valid_mask = self.instances_fv[scene_id][self.cur_frame - 1] & self.instances_fv[scene_id][self.cur_frame + 1]
            quats_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_quats, _quats_cur_frame
            )
        else:
            quats_cur_frame = self.instances_quats[scene_id][self.cur_frame] # (num_instances, 4)
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )                                                          # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[scene_id][..., 0]]        # (num_points, 3, 3)
        
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < num_frames
        ):
            _prev_ins_trans = self.instances_trans[scene_id][self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[scene_id][self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[scene_id][self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[scene_id][self.cur_frame - 1] & self.instances_fv[scene_id][self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[scene_id][self.cur_frame] # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[scene_id][..., 0]]
        
        # transform the means to world space
        means = torch.bmm(
            rot_per_pts, means.unsqueeze(-1)
        ).squeeze(-1) + trans_per_pts
        return means
    
    def transform_means_interp(self, means: torch.Tensor,scene_id: int,timestamps) -> torch.Tensor:
        """
        interpolate the rot and trans of instances to world space.
        NOTE: Pose interpolation is buggy when objects appear or disappear.
        """
        assert means.shape[0] == self.point_ids[scene_id].shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        start_time, end_time = timestamps[0], timestamps[-1]
  
        # use the previous and next frame to interpolate the pose
        _quats_prev_frame = self.instances_quats[scene_id][start_time]
        _quats_next_frame = self.instances_quats[scene_id][end_time]
        try:
            fraction = (self.cur_frame - start_time) / (end_time - start_time)
        except (IndexError, RuntimeError, ZeroDivisionError):
            fraction = 0.0
        print("source_timestamp:",timestamps,"fraction:",fraction)

        interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame,fraction=fraction)
        quats_cur_frame = interpolated_quats
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )                                                          # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[scene_id][..., 0]]        # (num_points, 3, 3)

        _prev_ins_trans = self.instances_trans[scene_id][start_time]
        _next_ins_trans = self.instances_trans[scene_id][end_time]
        trans_cur_frame = (1 - fraction) * _prev_ins_trans + fraction * _next_ins_trans
            
        trans_per_pts = trans_cur_frame[self.point_ids[scene_id][..., 0]]
        
        # transform the means to world space
        means = torch.bmm(
            rot_per_pts, means.unsqueeze(-1)
        ).squeeze(-1) + trans_per_pts
        return means

    def get_gaussians(self, cam, scene_id: int,source_images, source_extrinsics, intrinsics, timestamps) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.interpolate_pose:
            world_means = self.transform_means_interp(self._means[scene_id],scene_id=scene_id,timestamps=timestamps)
          
        else:
            world_means = self.transform_means(self._means[scene_id],scene_id=scene_id)

        ## get colors from ibr module, NOTE: Return the current time stamp
        query_rgbs,rigid_mask = self.ibr_query(source_images, scene_id,source_extrinsics, intrinsics,timestamps=timestamps)
        query_rgbs = torch.cat(query_rgbs, dim=1).reshape(world_means.shape[0],self.dim_in)
        self.set_cur_frame(cam.times.item())

        ## geometry decoder
        self.geometry_feat = self.rigid_decoder(query_rgbs)
        opacities, scales, quats, colors = self.geometry_feat.split([1, 3, 4, 3], dim=-1)
        rgbs = torch.sigmoid(colors)

        valid_mask = self.get_pts_valid_mask(scene_id=scene_id)
            
        activated_opacities = torch.sigmoid(opacities)
        activated_scales = self.get_scaling(scales=scales,scene_id=scene_id)
        activated_rotations = self.quat_act(quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[valid_mask],
            _opacities=activated_opacities[valid_mask],
            _rgbs=actovated_colors[valid_mask],
            _scales=activated_scales[valid_mask],
            _quats=activated_rotations[valid_mask],
        )

        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
        
        return gs_dict, rigid_mask

    def ibr_query(self, source_images, scene_id, source_extrinsics, intrinsics,timestamps):
        """
        query the colors of the gaussians in IBR manner
        """
        num_neighbors = source_images.shape[0]
        sampled_feats = []
        rigid_masks = []
        for i in range(num_neighbors):
            ## get the camera pose
            cam_pose = source_extrinsics[i:i+1]
            ## get the camera intrinsics
            cam_intrinsics = intrinsics[i:i+1]
            ## get the image
            image = source_images[i:i+1]
            frame_timestamp = timestamps[i]

            self.set_cur_frame(frame_timestamp)
            cur_means = self.transform_means(self._means[scene_id],scene_id=scene_id).reshape(-1,3)
            point_mask = self.get_pts_valid_mask(scene_id=scene_id)
            sampled_feat,rigid_mask = self.projector.dynamic_ibr(xyz = cur_means, 
                                        train_imgs = image,                
                                        train_cameras = cam_pose,     
                                        train_intrinsics= cam_intrinsics,
                                        point_mask = point_mask,
                                        )

            sampled_feats.append(sampled_feat.squeeze(0))
            rigid_masks.append(rigid_mask)

        return sampled_feats,torch.stack(rigid_masks,dim=0)

    @torch.no_grad()
    def ibr_query_with_occlusion(
        self, source_images, scene_id, source_extrinsics, intrinsics, timestamps, source_dino_features,
    ):
        if source_dino_features is None or source_dino_features.ndim != 4:
            raise ValueError("Dynamic IBR occlusion requires source DINO features [V, C, H, W]")
        if source_dino_features.shape[0] != source_images.shape[0]:
            raise ValueError("Source DINO features and RGB images must have the same view count")
        anchors = []
        for instance_id, instance in self.instances_dicts[scene_id].items():
            if "features" not in instance:
                raise ValueError(f"Dynamic IBR occlusion requires instance {instance_id} features aligned with pts")
            features = torch.as_tensor(instance["features"], device=source_images.device,
                                       dtype=source_dino_features.dtype)
            if features.shape != (instance["num_pts"], source_dino_features.shape[1]):
                raise ValueError(f"Instance {instance_id} features must have shape [num_pts, source DINO channels]")
            anchors.append(features)
        anchor_features = torch.cat(anchors, dim=0)
        if not torch.isfinite(anchor_features).all() or not torch.isfinite(source_dino_features).all():
            raise ValueError("Dynamic IBR occlusion requires finite point and image DINO features")
        anchor_features = F.normalize(anchor_features, p=2, dim=-1)

        sampled_rgbs, rigid_masks = self.ibr_query(
            source_images, scene_id, source_extrinsics, intrinsics, timestamps
        )
        similarities, valid_views = [], []
        h, w = source_images.shape[-2:]
        for i, timestamp in enumerate(timestamps):
            self.set_cur_frame(timestamp)
            xyz = self.transform_means(self._means[scene_id], scene_id=scene_id)
            pixels, in_front, _ = self.projector.compute_projections(
                xyz, source_extrinsics[i:i + 1], intrinsics[i:i + 1]
            )
            valid = in_front & self.projector.inbound(pixels, h, w) & self.get_pts_valid_mask(scene_id)
            sampled = F.grid_sample(
                source_dino_features[i:i + 1], self.projector.normalize(pixels, h, w).unsqueeze(1),
                align_corners=False,
            )[0, :, 0, :].T
            similarities.append((F.normalize(sampled, p=2, dim=-1) * anchor_features).sum(dim=-1))
            valid_views.append(valid.squeeze(0))

        # Match the static branch's cosine similarity and softmax temperature.
        logits = torch.stack(similarities) / 0.1
        valid = torch.stack(valid_views)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=0).masked_fill(~valid, 0)
        # Keep the existing 54-channel decoder: suppress less reliable RGB, retaining the best view's amplitude.
        weights = weights / weights.amax(dim=0, keepdim=True).clamp_min(1e-8)
        sampled_rgbs = [
            rgb * weight.reshape((-1,) + (1,) * (rgb.ndim - 1))
            for rgb, weight in zip(sampled_rgbs, weights)
        ]
        return sampled_rgbs, rigid_masks

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> Dict:
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        destination.update({
            prefix+"points_ids": self.point_ids,
            prefix+"instances_size": self.instances_size,
            prefix+"instances_fv": self.instances_fv,
        })
        return destination
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
 
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
