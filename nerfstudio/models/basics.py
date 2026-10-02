from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as tf

# Build the nearest neighbors model
from sklearn.neighbors import NearestNeighbors


def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

def interpolate_quats(q1, q2, fraction=0.5):
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    
    dot = (q1 * q2).sum(dim=-1)
    dot = torch.clamp(dot, -1, 1)
    
    neg_mask = dot < 0
    q2[neg_mask] = -q2[neg_mask]
    dot[neg_mask] = -dot[neg_mask]
    
    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)

    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    q_interp = (s1[..., None] * q1) + (s2[..., None] * q2)
    
    final_q_interp = torch.zeros_like(q1)
    final_q_interp[similar_mask] = q_interp_similar[similar_mask]
    final_q_interp[~similar_mask] = q_interp[~similar_mask]
    return final_q_interp

def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)

## for waymo dataset in opencv system
def get_viewmat(optimized_camera_to_world):
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    # R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

@dataclass
class dataclass_gs:
    _opacities: torch.Tensor
    _means: torch.Tensor
    _rgbs: torch.Tensor
    _scales: torch.Tensor
    _quats: torch.Tensor
    extras: Optional[Dict[str, torch.Tensor]] = None

    @property
    def opacities(self):
        return self._opacities
    
    @property
    def means(self):
        return self._means
    
    @property
    def rgbs(self):
        return self._rgbs
    
    @property
    def scales(self):
        return self._scales
    
    @property
    def quats(self):
        return self._quats
