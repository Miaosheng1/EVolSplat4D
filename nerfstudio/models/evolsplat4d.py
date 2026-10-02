# ruff: noqa: E741
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

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import kornia.morphology as morph
import torch
import torch.nn.functional as F
from einops import rearrange
from gsplat.rendering import rasterization
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from nerfstudio.background.backgroud_node import BackgroundNode
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.projection import Projector
from nerfstudio.model_components.sparse_conv import SparseCostRegNet, construct_sparse_tensor, sparse_to_dense_volume
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.basics import dataclass_gs, get_viewmat, k_nearest_sklearn, resize_image
from nerfstudio.RigidNode.rigid import RigidNodes
from nerfstudio.utils import colormaps


@dataclass
class EvolSplat4DModelConfig(ModelConfig):
    """Configuration for EvolSplat4D volume-based Gaussian prediction."""
    _target: Type = field(default_factory=lambda: EvolSplat4DModel)
    validate_every: int = 8000
    """period of steps where gaussians are culled and densified"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    entropy_loss: float = 0.1
    """weight of Entropy loss"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.
    """
    freeze_volume: bool = False
    """Whether to freeze the volume"""
    num_source_image: int = 2
    """Number of nearest neighbors to use for neighbor selection"""
    sparseConv_outdim: int = 16
    """Output dimension of the sparse convolution"""
    offset_max: float = 0.1
    """Maximum offset for the 3D offset"""
    local_radius: int = 1  ## project window radius range: 3×3
    """Radius of the local window"""
    background_res: int = 800
    """Resolution of the background model"""
    background_radius: int = 80
    """Distance of the background model"""
    background_center: List[float] = field(default_factory=lambda: [0,3.8,5.6])
    """Center of the background model"""
    voxel_size: float = 0.1
    """Voxel size of the foreground model"""
    use_dynamic: bool = True
    """Whether the model is dynamic"""
    interpolate_pose: bool = False
    """Whether to interpolate the pose"""



class EvolSplat4DModel(Model):
    """EvolSplat4D for static and dynamic urban scene synthesis.

    Args:
        config: EvolSplat4D configuration to instantiate the model.
    """

    config: EvolSplat4DModelConfig

    def __init__(
        self,
        *args,
        seed_points: List,
        instances_dicts: Dict[str, Any],
        **kwargs,
    ):
        self.seed_points = seed_points
        self.instances_dicts = instances_dicts
        self.num_scenes = len(seed_points) # type: ignore
        self.aabbs = kwargs.get("aabbs", None)
        super().__init__(*args, **kwargs)

    def populate_modules(self):

        ## Important: input the 3D point of the scene. All scenes data should be stroed as List as the unsame point number
        self.means = [] # type: ignore
        self.anchor_feats = []
        self.scales = [] # type: ignore
        self.offset = [] # type: ignore
        if self.seed_points is not None:
            for i in tqdm(range(self.num_scenes),desc="Loading scenes"):
                means = self.seed_points[i]['points3D_xyz'].float()
                anchors_feat =   self.seed_points[i]['points3D_rgb'].float() 
                offsets = torch.zeros_like(means)
                distances, _ = k_nearest_sklearn(means.data, 3)
                distances = torch.from_numpy(distances)
                avg_dist = distances.mean(dim=-1, keepdim=True)
                scales = torch.log(avg_dist.repeat(1, 3))
                
                ## stack the parameters into list
                self.means.append(means)
                self.anchor_feats.append(anchors_feat)
                self.scales.append(scales)
                self.offset.append(offsets)
       



        ## load mannul param:
        self.local_radius = self.config.local_radius
        self.sparseConv_outdim = self.config.sparseConv_outdim
        self.offset_max = self.config.offset_max 
        self.num_neibours = self.config.num_source_image

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        ## config the projecter
        self.projector = Projector()

         ## construct the sparse tensor
        self.sparse_conv = SparseCostRegNet(d_in=16, d_out=self.sparseConv_outdim).cuda()
        self.feature_dim_in = 4*self.num_neibours*(2*self.local_radius+1)**2
        # self.gs_color = [t.clone() for t in self.anchor_feats]
        
        ## gaussian MLP
        self.mlp_color = MLP(
                in_dim= self.feature_dim_in,
                num_layers=3,
                layer_width=128,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="torch",
            )
        
        self.mlp_conv = MLP(
                in_dim= self.sparseConv_outdim + 1,
                num_layers=2,
                layer_width=64,
                out_dim=3+4,
                activation=nn.Tanh(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_opacity = MLP(
                in_dim=self.sparseConv_outdim + 1,
                num_layers=2,
                layer_width=64,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="torch",
            )
        
        self.mlp_offset = MLP(
                in_dim=self.sparseConv_outdim,
                num_layers=2,
                layer_width=64,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            )
        
        ## Rigid Node
        if self.config.use_dynamic: 
            self.rigid_node = RigidNodes(
                num_scenes=self.num_scenes,
                instances_dicts=self.instances_dicts,
                interpolate_pose=self.config.interpolate_pose,
            )
        else:
            self.rigid_node = None
  
        
      
        ## Background Model for sky & distant view
        self.background_node = BackgroundNode()

    def load_state_dict(self, model_dict, strict=False):  # type: ignore
        super().load_state_dict(model_dict, strict= strict)

    ## Run Validate set
    def after_train(self,step: int):
        if self.step <= 2000:
            return
        return

   
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                update_every_num_iters=self.config.validate_every,
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = {}
        ## add mlp decoder parameters
        gps['gaussianDecoder'] = list(self.mlp_color.parameters())
        gps['mlp_conv'] = list(self.mlp_conv.parameters())
        gps['mlp_opacity'] = list(self.mlp_opacity.parameters())
        gps['mlp_offset'] = list(self.mlp_offset.parameters())
        gps['sparse_conv'] = list(self.sparse_conv.parameters())
        gps['background_model'] = list(self.background_node.parameters())
        if self.config.use_dynamic and self.rigid_node is not None:
            gps.update(self.rigid_node.get_param_groups())
        return gps


    def _downscale_if_required(self, image):
        d = 1
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}


    def get_outputs(self, camera: Cameras,batch) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        scene_id = batch.get("scene_id", None)
        means = self.means[scene_id].cuda()
        scales = self.scales[scene_id].cuda()
        offset = self.offset[scene_id].cuda()
        anchors_feat = self.anchor_feats[scene_id].cuda()

        assert self.aabbs is not None, "self.aabbs cannot be None"
        Bbx_max = self.aabbs[scene_id][1]
        Bbx_min = self.aabbs[scene_id][0]
        
        optimized_camera_to_world = camera.camera_to_worlds

        source_images = batch['source']['image']
        dino_feature = batch['source']['dino_feat'].permute(0, 3, 1, 2) 
        source_images = rearrange(source_images[None,...],"b v h w c -> b v c h w")
        source_extrinsics = batch['source']['extrinsics'] 
        target_image = batch['target']['image'].squeeze(0)

        

        ## query 3d feature
        if not self.config.freeze_volume:
            sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(raw_coords=means.clone(),
                                                                                   feats=anchors_feat,
                                                                                   Bbx_max=Bbx_max,
                                                                                   Bbx_min=Bbx_min,
                                                                                   voxel_size=self.config.voxel_size
                                                                                   ) 
            feat_3d = self.sparse_conv(sparse_feat)
            dense_volume = sparse_to_dense_volume(sparse_tensor=feat_3d,coords=self.valid_coords,vol_dim=self.vol_dim).unsqueeze(dim=0)
            self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')
       

        
        ## query 2d feature
        with torch.no_grad():
            sampled_feat, projection_mask, vis_map = self.projector.sample_within_window(
                xyz=means,
                anchor_feat = anchors_feat,
                train_imgs=source_images.squeeze(0),  # [N_view,c,h,w]
                train_cameras=source_extrinsics,      # [N_view,4,4]
                train_intrinsics=batch['source']['intrinsics'],  # [N_view,4,4]
                image_feat=dino_feature,
                local_radius=self.local_radius,
                # dino_threshold=0.8,
            )  # [N_samples, N_views, C]
        N_samples = sampled_feat.shape[0]
        sampled_feat = torch.concat([sampled_feat,vis_map],dim=-1).reshape(N_samples,self.feature_dim_in)
        if torch.isnan(sampled_feat).any():
            raise ValueError("sampled_feat contains NaN")
        # sampled_feat = sampled_feat.reshape(-1,self.feature_dim_in)

        means_crop = means[projection_mask]
        vailid_scales = scales[projection_mask]
        last_offset = offset[projection_mask]


        ## Trilinear the feature volume
        grid_coords = self.get_grid_coords(means_crop + last_offset, voxel_size=self.config.voxel_size,bbx_min=Bbx_min)
        feat_3d = self.interpolate_features(grid_coords=grid_coords, feature_volume=self.dense_volume).permute(3, 4, 1, 0, 2).squeeze()

        with torch.no_grad():
            ob_view = means_crop - optimized_camera_to_world[0,:3,3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            # ob_view = ob_view / ob_dist

        ## Important: Learn the 3D scale, rotation and opacity 
        # input_feat = torch.cat([feat_3d,ob_dist,ob_view],dim=-1).squeeze(dim=1)
        input_feat = torch.cat([feat_3d,ob_dist],dim=-1).squeeze(dim=1)
        scales_crop, quats_crop = self.mlp_conv(input_feat).split([3,4],dim=-1)
        opacities_crop = self.mlp_opacity(input_feat)
        colors_crop = self.mlp_color(sampled_feat) 

        # color_crop = torch.tensor([1.0,1.0,1.0]).repeat(colors_crop.shape[0],1).to(self.device)
        # gs_color= torch.where(vis_map.unsqueeze(-1), colors_crop, anchors_feat[projection_mask]).float() 
        gs_color = colors_crop

        ## Optimize the 3D offset via MLP
        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        means_crop += offset_crop
        ## update the latest offset for each 3DGS; only save the tensor without grad
        if self.training:
            self.offset[scene_id][projection_mask] = offset_crop.detach().cpu()  

        static_scales = torch.exp(scales_crop + vailid_scales)
        static_quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
        static_opacities = torch.sigmoid(opacities_crop)


        ## collect the rigid node gaussians
        ## TODO: Frame id  should be related to time, single frame may contain 3 images
        rigid_node_gaussians = None
        if self.config.use_dynamic and self.rigid_node is not None:
            self.rigid_node.set_cur_frame(frame_id=camera.times.item()) # type: ignore
            rigid_node_gaussians,rigid_masks = self.rigid_node.get_gaussians(cam=camera,
                                                            scene_id=scene_id,
                                                            source_images=source_images.squeeze(0),
                                                            source_extrinsics= source_extrinsics,
                                                            intrinsics= batch['source']['intrinsics'],
                                                            timestamps=batch['source']['source_timestamp'])   
            fg_gaussians = dataclass_gs(
                _means=torch.cat((means_crop,rigid_node_gaussians['_means']),dim=0),
                _scales=torch.cat((static_scales,rigid_node_gaussians['_scales']),dim=0),
                _quats=torch.cat((static_quats,rigid_node_gaussians['_quats']),dim=0),
                _rgbs=torch.cat((gs_color,rigid_node_gaussians['_rgbs']),dim=0),
                _opacities=torch.cat((static_opacities,rigid_node_gaussians['_opacities']),dim=0),
                extras=None        # to save some extra information (TODO) more flexible way
            )

            ## Check static and rigid node points.
            # exit()
        else:
            rigid_masks = None
            fg_gaussians = dataclass_gs(
                _means=means_crop,
                _scales=static_scales,
                _quats=static_quats,
                _rgbs=gs_color,
                _opacities=static_opacities,
                extras=None        # to save some extra information (TODO) more flexible way
            )

        """  For Background, Fixed Background hemisphere Model """
        if rigid_masks is None:
            # Static scenes have no dynamic pixels to exclude from the background.
            rigid_masks = torch.zeros_like(source_images[0, :, 0], dtype=torch.bool)
        else:
            rigid_masks = rigid_masks.float().unsqueeze(1)
            rigid_masks = morph.erosion(rigid_masks.float(), kernel=torch.ones(3,3).cuda()).bool().squeeze(1)

        bg_gaussians = self.background_node.get_gaussians(source_images=source_images, 
                                                          source_extrinsics= source_extrinsics, 
                                                          intrinsics= batch['source']['intrinsics'],
                                                          dy_mask=rigid_masks,
                                                          )


    

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        viewmat = get_viewmat(optimized_camera_to_world)
        K = batch['target']['intrinsics'][...,:3,:3]
        H, W = target_image.shape[:2]
        self.last_size = (H, W)

        render_mode = "RGB+ED"
        def render_fn(gaussians,opaticy_mask=None):
            renders, alphas, info = rasterization(
                means=gaussians.means,
                quats=gaussians.quats,
                scales=gaussians.scales,
                opacities=gaussians.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gaussians.opacities.squeeze(),
                colors=gaussians.rgbs,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,  ## set True for more memory efficient
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
            )

            renders = renders[0]
            alphas = alphas[0].squeeze(-1)

            assert renders.shape[-1] == 4, "Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            
            return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            

        fg_render_rgb, fg_depth, fg_alpha = render_fn(gaussians=fg_gaussians)

        ## Background Model for sky & distant view
        bg_gaussians = dataclass_gs(
            _means=bg_gaussians['_means'],
            _scales=bg_gaussians['_scales'],
            _quats=bg_gaussians['_quats'],
            _rgbs=bg_gaussians['_rgbs'],
            _opacities=bg_gaussians['_opacities'],
        )
        bg_render_rgb, bg_depth, bg_alpha,  = render_fn(gaussians=bg_gaussians)
        render_rgb = fg_render_rgb + (1 - fg_alpha) * bg_render_rgb
        depth = fg_depth + (1 - fg_alpha) * bg_depth


        outputs ={
            "rgb": render_rgb.squeeze(0),  # type: ignore
            "depth": depth,  # type: ignore
            "accumulation": fg_alpha,  # type: ignore
            "background": (1 - fg_alpha) * bg_render_rgb,  # type: ignore
            "rigid_opacity": (
                rigid_node_gaussians['_opacities'] if rigid_node_gaussians is not None else static_opacities[:0]
            ),
            "fg_rgb":fg_render_rgb,
            "bg_acc":bg_alpha,
                }  # type: ignore
        
        if not self.training and rigid_node_gaussians is None:
            outputs['rigid_rgb'] = torch.ones_like(render_rgb)
            outputs['rigid_acc'] = torch.zeros_like(fg_alpha)
        elif not self.training:
            with torch.no_grad():
                rigid_gaussian = dataclass_gs(
                    _means=rigid_node_gaussians['_means'], # type: ignore
                    _scales=rigid_node_gaussians['_scales'],# type: ignore
                    _quats=rigid_node_gaussians['_quats'], # type: ignore
                    _rgbs=rigid_node_gaussians['_rgbs'], # type: ignore
                    _opacities=rigid_node_gaussians['_opacities'], # type: ignore
                )
            rigid_render_rgb, _ , rigid_alpha,  = render_fn(gaussians=rigid_gaussian)
            # green_background = torch.tensor([0.0, 177, 64],device=self.device) / 255.0
            green_background = torch.tensor([255, 255, 255],device=self.device) / 255.0
            outputs['rigid_rgb'] = rigid_render_rgb + (1 - rigid_alpha) * green_background
            outputs['rigid_acc'] = rigid_alpha



        return outputs
    
    
    def interpolate_features(self, grid_coords, feature_volume):
        grid_coords = grid_coords[None, None, None, ...]
        feature = F.grid_sample(feature_volume,
                                grid_coords,
                                mode='bilinear',
                                align_corners=True,
                                )
        return feature
   
    
    def get_grid_coords(self, position_w, voxel_size=0.1,bbx_min=None):
        bounding_min = bbx_min
        pts = position_w - bounding_min.to(position_w)
        x_index = pts[..., 0] / voxel_size
        y_index = pts[..., 1] / voxel_size
        z_index = pts[..., 2] / voxel_size
        """ Normalize the point coordinates to [-1,1]"""

        dhw = torch.stack([x_index, y_index, z_index], dim=1)

        # index = dhw.clone().long()
        dhw[..., 0] = dhw[..., 0] / self.vol_dim[0] * 2 - 1
        dhw[..., 1] = dhw[..., 1] / self.vol_dim[1] * 2 - 1
        dhw[..., 2] = dhw[..., 2] / self.vol_dim[2] * 2 - 1
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords
    

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        self._downscale_if_required(image)
        return image.to(self.device)


    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = batch['target']["image"].squeeze(0)
        metrics_dict = {}
        scene_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        scene_dict['scene_id'] = batch['scene_id']
        scene_dict['image_index'] = batch['image_index']
    
        return metrics_dict,scene_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        
        gt_img = batch['target']["image"].squeeze(0)
        pred_img = outputs["rgb"]
       
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        # Bce_loss =  0.1 * F.binary_cross_entropy_with_logits(outputs['accumulation'].squeeze().clip(1e-3, 1.0-1e-3),batch['mask'].float())
        Bce_loss = F.l1_loss(outputs['accumulation'].squeeze() * batch['mask'], batch['mask'].float())


        acc_fg, acc_bg = outputs['accumulation'], outputs['bg_acc']
        weights_ratio = acc_bg / torch.clamp(acc_fg + acc_bg, min=1e-9)
        entropy_loss = -(
          weights_ratio * torch.log(weights_ratio + 1e-9)
          + (1.0 - weights_ratio) * torch.log(1.0 - weights_ratio + 1e-9)
      )
        

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
             "bce_loss": 0.1 * Bce_loss,
             "entropy_loss":5e-4 * torch.mean(entropy_loss),
        }

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, batch=None, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        if self.config.use_dynamic:
            self.rigid_node.in_test_set = True
            outs = self.get_outputs(camera,batch=batch)
            self.rigid_node.in_test_set = False
        else:
            outs = self.get_outputs(camera,batch=batch)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = batch['target']["image"].squeeze(0)# type: ignore
        predicted_rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"]
            # accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        bg_color = outputs["background"]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, "background":bg_color}
        if 'rigid_rgb' in outputs:
            images_dict['rigid_rgb'] = outputs['rigid_rgb']
            images_dict['rigid_acc'] = colormaps.apply_colormap(outputs["rigid_acc"])

        return metrics_dict, images_dict
    
    @torch.no_grad()
    def init_volume(self, scene_id:int = 0):
        ## Foreground
        self.config.freeze_volume = True
        means = self.means[scene_id].cuda()
        anchors_feat = self.anchor_feats[scene_id].cuda()
        sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(raw_coords=means.clone(),
                                                                               feats=anchors_feat,
                                                                                Bbx_max=self.aabbs[scene_id][1],
                                                                                Bbx_min=self.aabbs[scene_id][0],
                                                                                   ) 
        feat_3d = self.sparse_conv(sparse_feat) # type: ignore
        dense_volume = sparse_to_dense_volume(sparse_tensor=feat_3d,coords=self.valid_coords,vol_dim=self.vol_dim).unsqueeze(dim=0)
        if hasattr(self, 'dense_volume'):
            del self.dense_volume
        self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')

        ## update the 3DGS location
        grid_coords = self.get_grid_coords(means,voxel_size=self.config.voxel_size,bbx_min=self.aabbs[scene_id][0])
        feat_3d = self.interpolate_features(grid_coords=grid_coords, feature_volume=self.dense_volume).permute(3, 4, 1, 0, 2).squeeze()

        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        self.offset[scene_id] = offset_crop.detach().cpu()
        print("Init the Static volume, Done.  \n")
