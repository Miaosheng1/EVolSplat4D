from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.background.unet import UNet


class BackgroundNode(nn.Module):
    def __init__(self
                 ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024)
        down_attention: Tuple[bool, ...] = (False, False, False, True, True)
        mid_attention: bool = True
        up_channels: Tuple[int, ...] = (1024, 512, 256, 128, 64)
        up_attention: Tuple[bool, ...] = (True, True, False, False,False)

        self.gs_channels = 12 

        self.unet = UNet(
            # 3+6+1+1, self.gs_channels, 
            3+6, self.gs_channels, 
            down_channels=down_channels,
            down_attention=down_attention,
            mid_attention=mid_attention,
            up_channels=up_channels,
            up_attention=up_attention,
        )
        self.conv = nn.Conv2d(self.gs_channels, self.gs_channels, kernel_size=1)
        self.conv_confidence = nn.Conv2d(self.gs_channels + 1, 1, kernel_size=1)

        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.exp(x) * 0.1
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.near = 0.01
        self.far = 200
        self.global_step = 0

    
    def get_ray_directions(self,
        H: int,
        W: int,
        focal: Union[float, Tuple[float, float]],
        principal: Optional[Tuple[float, float]] = None,
        use_pixel_centers: bool = True,
    ) -> Float[Tensor, "H W 3"]:
        """
        Get ray directions for all pixels in camera coordinate.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
        Outputs:
            directions: (H, W, 3), the direction of the rays in camera coordinate
        """
        pixel_center = 0.5 if use_pixel_centers else 0

        if isinstance(focal, float):
            fx, fy = focal, focal
            cx, cy = W / 2, H / 2
        else:
            fx, fy = focal
            assert principal is not None
            cx, cy = principal

        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device) + pixel_center,
            torch.arange(H, dtype=torch.float32, device=self.device) + pixel_center,
            indexing="xy",
        )

        directions: Float[Tensor, "H W 3"] = torch.stack(
            [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
        )

        return directions


    def get_rays(
            self,
        directions: Float[Tensor, "... 3"],
        c2w: Float[Tensor, "... 4 4"],
        keepdim=False,
        noise_scale=0.0,
        normalize=True,
    ) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
        # Rotate ray directions from camera coordinate to the world coordinate
        assert directions.shape[-1] == 3

        if directions.ndim == 2:  # (N_rays, 3)
            if c2w.ndim == 2:  # (4, 4)
                c2w = c2w[None, :, :]
            assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
            rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
            rays_o = c2w[:, :3, 3].expand(rays_d.shape)
        elif directions.ndim == 3:  # (H, W, 3)
            assert c2w.ndim in [2, 3]
            if c2w.ndim == 2:  # (4, 4)
                rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                    -1
                )  # (H, W, 3)
                rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
            elif c2w.ndim == 3:  # (B, 4, 4)
                rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                    -1
                )  # (B, H, W, 3)
                rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
        elif directions.ndim == 4:  # (B, H, W, 3)
            assert c2w.ndim == 3  # (B, 4, 4)
            rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

        # add camera noise to avoid grid-like artifect
        # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
        if noise_scale > 0:
            rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
            rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

        if normalize:
            rays_d = F.normalize(rays_d, dim=-1)
        if not keepdim:
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        return rays_o, rays_d
    
    
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def get_gaussians(self, source_images, source_extrinsics, intrinsics,dy_mask):

        target_H = source_images.shape[-2] //16 *16
        target_W = source_images.shape[-1] //16 *16
        source_images = source_images[...,0:target_H,0:target_W]
        dy_mask = dy_mask[...,0:target_H,0:target_W]

        B, N, C, origin_H, origin_W = source_images.size()
        ## ray_o, ray_d
        input_directions = []
        for i in range(N):
             fx, fy = intrinsics[i, 0, 0], intrinsics[i, 1, 1]
             cx, cy = intrinsics[i, 0, 2], intrinsics[i, 1, 2]
             direction = self.get_ray_directions(H=origin_H, W=origin_W,
                                           focal=(fx, fy), principal=(cx, cy))
             input_directions.append(direction)

        input_directions = torch.stack(input_directions)
        ## Convert OpenGL to OpenCV for static training.
        cv_extrinsics = source_extrinsics * torch.tensor([1,-1,-1,1],device=self.device)
        rays_o, rays_d = self.get_rays(input_directions, cv_extrinsics, keepdim=True, normalize=True)
        rays_o = rays_o[None, ...]
        rays_d = rays_d[None, ...]
        # plucker
        plucker = self.plucker_embedder(rays_o, rays_d)

        ## depths
        input_feat = torch.cat([source_images, plucker], dim=2).squeeze(0)
        x = self.unet(input_feat)
        x = self.conv(x)
        valid_mask =  ~rearrange(dy_mask, "v h w -> (v h w)")

        gaussians = rearrange(x, "(b v) (n c) h w -> b (v h w n) c",
                              b=1, v=2, n=1, c=self.gs_channels).squeeze(0)

         # Activate gaussian parameters
        w = gaussians[..., :1].sigmoid()
        opacities = self.opt_act(gaussians[..., 1:2])
        scales = self.scale_act(gaussians[..., 2:5])
        rotations = self.rot_act(gaussians[..., 5:9])
        rgbs = self.rgb_act(gaussians[..., 9:12])
        # rgbs = rearrange(source_images, "b v c h w-> b (v h w) c", b=1, v=2).squeeze(0)

        depth = self.near*(1-w) + self.far*w

        origins = rearrange(rays_o, "b v h w c ->  (b v h w) c")
        directions = rearrange(rays_d, "b v h w c ->  (b v h w) c")

        means = origins + directions * depth

        gs_dict = dict(
            _means=means[valid_mask],
            _opacities=opacities[valid_mask],
            _scales=scales[valid_mask],
            _quats=rotations[valid_mask],
            _rgbs=rgbs[valid_mask],
        )

        return gs_dict
    

   