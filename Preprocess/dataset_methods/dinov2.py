import joblib
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn


class DinoFeatureExtractor(nn.Module):
    def __init__(self, n_components=16, IMG_H=640, IMG_W=960, repo=None):
        super(DinoFeatureExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = torch.hub.load(
            str(repo) if repo else "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
            source="local" if repo else "github",
        )
        self.feature_extractor = self.feature_extractor.to(self.device).eval()
        self.high_dim = self.feature_extractor.embed_dim
        self.patch_size = self.feature_extractor.patch_size

        self.n_components = n_components

        self.pca_model = None
        self.image_size = (IMG_H, IMG_W)
        self.preprocess = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _extract_high_dim_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image_tensor.shape
        resize_h = (H // self.patch_size) * self.patch_size
        resize_w = (W // self.patch_size) * self.patch_size
        if (H != resize_h) or (W != resize_w):
            image_tensor = F.interpolate(image_tensor, size=(resize_h, resize_w), mode="bilinear", align_corners=False)

        features_dict = self.feature_extractor.get_intermediate_layers(image_tensor, n=1, return_class_token=False)
        patch_features = features_dict[0]
        H_patch, W_patch = resize_h // self.patch_size, resize_w // self.patch_size
        coarse_map = patch_features.reshape(B, H_patch, W_patch, self.high_dim)
        coarse_map_chw = coarse_map.permute(0, 3, 1, 2)
        dense_map_chw = F.interpolate(coarse_map_chw, size=(H, W), mode="bilinear", align_corners=False)
        dense_map = dense_map_chw.permute(0, 2, 3, 1)
        return dense_map

    def transform(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            high_dim_features = self._extract_high_dim_features(input_tensor)
            B, H, W, D = high_dim_features.shape
            features_flat = high_dim_features.reshape(-1, D).cpu().numpy()
            reduced_features_flat = self.pca_model.transform(features_flat)
            reduced_features = torch.from_numpy(reduced_features_flat).reshape(H, W, self.n_components)
        # The existing training/inference loader restores this tensor directly on CUDA.
        return reduced_features.to(self.device)

    def load_pca_model(self, path: str):

        self.pca_model = joblib.load(path)
        self.n_components = self.pca_model.n_components
        if self.n_components != 16 or self.pca_model.n_features_in_ != self.high_dim:
            raise ValueError("Expected the checkpoint-compatible 384-to-16 DINOv2 PCA model")
