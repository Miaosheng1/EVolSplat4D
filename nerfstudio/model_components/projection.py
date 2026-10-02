# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import torch.nn.functional as F


class Projector():
    def __init__(self):
        pass
    
    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras,train_intrinsics):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]  Opencv
        :param train_cameras: [n_views, 4, 4]  OpenGL
        :param camera intrinsics: [n_views, 4, 4]
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:1]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        # train_cameras = train_cameras * torch.tensor([1, -1, -1, 1],device="cuda")
        train_poses = train_cameras.reshape(-1, 4, 4)  # [n_views, 4, 4]

        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera

        depth = projections[..., 2].reshape((num_views, ) + original_shape)
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape),\
               depth
    

    def dynamic_ibr(self,  xyz, train_imgs, train_cameras, train_intrinsics,point_mask,local_radius=1,):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
    
        xyz = xyz.detach()
        n_samples = xyz.shape[0]    
        h, w = train_imgs.shape[2:]

        n_views, _ ,_ = train_cameras.shape
        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = self.generate_window_grid(-local_radius, local_radius,
                                                -local_radius, local_radius,
                                                local_h, local_w, device=xyz.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_views, 1, 1)


        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, _ = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())
        pixel_locations = pixel_locations.unsqueeze(dim=2) + window_grid.unsqueeze(dim=1)
        pixel_locations = pixel_locations.reshape(n_views,-1,2)  ## [N_view, N_points,2]
        mask_in_front = mask_in_front & point_mask.unsqueeze(0)
        mask_in_front = mask_in_front.unsqueeze(-1).repeat(1,1, local_h*local_w).reshape(n_views,-1)


        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0)  # [n_points, n_views, 3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 0)[..., None]   # [n_rays, n_samples, n_views, 1]

        rgb = rgb_sampled.masked_fill(mask==0, 0)

        projection_mask = mask[..., :].sum(dim=1) > 0
        rigid_mask = torch.zeros(train_imgs.shape[2:]).to(xyz.device)
        valid_pixel_locations = pixel_locations[0][projection_mask.squeeze()].long()
        rigid_mask[valid_pixel_locations[:,1],valid_pixel_locations[:,0]] = 1.0
        # return rgb, projection_mask.squeeze()
        return rgb.reshape(n_samples,n_views,local_w*local_h,3), \
                rigid_mask.bool()
    

    def sample_within_window(self,  xyz, anchor_feat, train_imgs, train_cameras, train_intrinsics, image_feat=None, local_radius = 2):
        '''
        :param xyz: [n_samples, 3]
        :param source_imgs: [ n_views, c, h, w]
        :param source_cameras: [ n_views, 4, 4], in OpnecGL
        :param source_intrinsics: [ n_views, 4, 4]
        :param image_feat: [ n_views, c, h, w]
        :param local_radius: local radius
        :param dino_threshold: dino threshold
        :return: rgb_feat_sampled: [n_samples,n_views,c],
                 mask: [n_samples,n_views,1]
        '''
        n_views, _ ,_ = train_cameras.shape
        
        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = self.generate_window_grid(-local_radius, local_radius,
                                                -local_radius, local_radius,
                                                local_h, local_w, device=xyz.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_views, 1, 1)

        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # sample within the window size
        pixel_locations, mask_in_front, project_depth = self.compute_projections(xyz, train_cameras,train_intrinsics.clone())
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front ).float()
        pixel_mask = torch.any(mask.bool(), dim=0)

        n_samples = pixel_mask.sum()

        pixel_locations = pixel_locations[:, pixel_mask, :]
        valid_samples = pixel_mask.sum()

        # if valid_samples == 0:
        #     return None, None, None

        ## Occlusion-Aware check for IBR:
        assert image_feat is not None
        feat_sampled = F.grid_sample(image_feat, self.normalize(pixel_locations, h, w).unsqueeze(dim=1), align_corners=False)
        feat_sampled = feat_sampled.squeeze().permute(0, 2, 1)
        anchor_feat_valid =  F.normalize(anchor_feat[pixel_mask], p=2,dim=-1)
        A_norm = F.normalize(feat_sampled, p=2, dim=-1)

        poins_sim = (A_norm * anchor_feat_valid.unsqueeze(0)).sum(dim=-1)  # [2, N]
        # normalized_sim = poins_sim / (poins_sim.sum(dim=0, keepdim=True) + 1e-8)
        normalized_sim = torch.softmax(poins_sim / 0.1, dim=0)

        visibility_map = normalized_sim.unsqueeze(-1).repeat(1,1, local_h*local_w) 
        visibility_map = visibility_map.reshape(n_views,n_samples,local_w*local_h) ##type: ignore
        
        # sim_diff = torch.abs(poins_sim[0] - poins_sim[1])
        # occlusion_mask = sim_diff > 0.3 
        # occlusion_view1 = (poins_sim[0] > poins_sim[1]) & occlusion_mask
        # occlusion_view0 = (poins_sim[1] > poins_sim[0]) & occlusion_mask

    

        pixel_locations = pixel_locations.unsqueeze(dim=2) + window_grid.unsqueeze(dim=1)
        pixel_locations = pixel_locations.reshape(n_views,-1,2)  ## [N_view, N_points,2]

        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(dim=1) # [n_views, 1, n_points, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=False)
        rgb = rgbs_sampled.permute(2, 3, 0, 1).squeeze(dim=0).reshape(valid_samples,n_views,local_w*local_h,3)  ##type:ignore
        # rgb[occlusion_view0, 0] = 0.0
        # rgb[occlusion_view1, 1] = 0.0
        # print(rgb.shape)



        # rgb = rgb.reshape(-1,local_w*local_w,n_views,3).permute(0,2,1,3)
        # mask = mask.reshape(-1,local_w*local_w,n_views,1).permute(0,2,1,3)
        # return rgb.reshape(n_samples,local_w*local_h*n_views,3), mask.reshape(n_samples,local_w*local_h*n_views)
        return rgb, \
               pixel_mask, \
               visibility_map.to(xyz.device).permute(1,0,2).unsqueeze(-1)
    


    
    def generate_window_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device=None):
        assert device is not None

        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                            torch.linspace(h_min, h_max, len_h, device=device)],
                            )
        grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

        return grid
