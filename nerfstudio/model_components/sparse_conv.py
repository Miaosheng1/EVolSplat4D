import numpy as np
import torch
import torchsparse.nn as spnn
from torch import nn
from torchsparse.tensor import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class SparseResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride), spnn.BatchNorm(outc))
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SparseCostRegNet(nn.Module):
    def __init__(self, d_in, d_out=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F


def sparse_to_dense_volume(sparse_tensor, coords, vol_dim, default_val=0):
    c = sparse_tensor.shape[-1]
    coords = coords.to(torch.int64)
    ## clamp the coords to prevent the data overflow
    coords[:, 0] = coords[:, 0].clamp(0, vol_dim[0] - 1)
    coords[:, 1] = coords[:, 1].clamp(0, vol_dim[1] - 1)
    coords[:, 2] = coords[:, 2].clamp(0, vol_dim[2] - 1)
    device = sparse_tensor.device
    dense = torch.full([vol_dim[0], vol_dim[1], vol_dim[2], c], float(default_val), device=device)  # type: ignore
    dense[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_tensor
    return dense


def construct_sparse_tensor(raw_coords, feats, Bbx_min: torch.Tensor, Bbx_max: torch.Tensor, voxel_size=0.1):
    if isinstance(raw_coords, torch.Tensor) or isinstance(feats, torch.Tensor):
        raw_coords = raw_coords.cpu().numpy()
        feats = feats.cpu().numpy()

    # X_MIN, X_MAX = Bbx_min[0], Bbx_max[0]
    # Y_MIN, Y_MAX = Bbx_min[1], Bbx_max[1]
    # Z_MIN, Z_MAX = Bbx_min[2], Bbx_max[2]

    # bbx_max = np.array([X_MAX,Y_MAX,Z_MAX])
    # bbx_min = np.array([X_MIN,Y_MIN,Z_MIN])
    vol_dim = (Bbx_max - Bbx_min) / voxel_size
    vol_dim = vol_dim.int().tolist()

    raw_coords -= np.array([Bbx_min[0], Bbx_min[1], Bbx_min[2]]).astype(int)

    coords, indices = sparse_quantize(
        raw_coords, voxel_size, return_index=True
    )  ## voxelize the pnt to discrete formation
    coords = torch.tensor(coords, dtype=torch.int).cuda()
    zeros = torch.zeros(coords.shape[0], 1).cuda()
    ## Note: [B,X,Y,Z] in Torch sparsev 2.1
    coords = torch.cat((zeros, coords), dim=1).to(torch.int32)

    feats = torch.tensor(feats[indices], dtype=torch.float).cuda()
    sparse_feat = SparseTensor(feats, coords=coords)
    return sparse_feat, vol_dim, coords[:, 1:]


# class ResNetFeatureExtractor(nn.Module):
#     def __init__(self, feature_dim=64,pca_weights=None):
#         super(ResNetFeatureExtractor, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#         resnet = models.resnet18(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         self.feature_proj = nn.Linear(512, feature_dim)

#         if pca_weights is not None:
#             self.feature_proj.weight.data = pca_weights
#         else:
#             nn.init.xavier_uniform_(self.feature_proj.weight)

#         self.feature_proj.bias.data.zero_()

#         for param in self.feature_proj.parameters():
#             param.requires_grad = False

#     def forward(self, img):
#         input_img = transforms.functional.normalize(
#             img.permute(0, 3, 1, 2).float(),
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )

#         B, C, H, W = input_img.shape

#         with torch.no_grad():
#             features = self.backbone(input_img)  # [B, 512, H/32, W/32]
#             features_flat = features.permute(0, 2, 3, 1).reshape(-1, 512)  # [B*H*W, 512]
#             features_reduced = self.feature_proj(features_flat)  # [B*H*W, 64]
#             features_reduced = features_reduced.reshape(B, H//32, W//32, -1).permute(0, 3, 1, 2)  # [B, 64, H/32, W/32]

#         features_upsampled = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)

#         return features_upsampled  # [B, feature_dim, H, W]
