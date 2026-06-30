"""MADNet model architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFeatureEncoder(nn.Module):
    """Shared grayscale encoder for RGB and IR inputs."""

    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LocalCorrelation(nn.Module):
    """Local all-offset correlation volume over a square search window."""

    def __init__(self, radius: int = 8):
        super().__init__()
        if radius < 0:
            raise ValueError("radius must be non-negative")
        self.radius = radius
        self.num_offsets = (2 * radius + 1) ** 2

    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor) -> torch.Tensor:
        rgb_features = F.normalize(rgb_features, p=2, dim=1, eps=1e-6)
        ir_features = F.normalize(ir_features, p=2, dim=1, eps=1e-6)

        radius = self.radius
        if radius == 0:
            return (rgb_features * ir_features).sum(dim=1, keepdim=True)

        batch_size, channels, height, width = rgb_features.shape
        kernel_size = 2 * radius + 1
        ir_patches = F.unfold(ir_features, kernel_size=kernel_size, padding=radius)
        ir_patches = ir_patches.view(
            batch_size,
            channels,
            kernel_size * kernel_size,
            height * width,
        )
        rgb_flat = rgb_features.view(batch_size, channels, 1, height * width)
        corr = (rgb_flat * ir_patches).sum(dim=1)
        return corr.view(batch_size, kernel_size * kernel_size, height, width)


class MADNetV2Corr(nn.Module):
    """MADNet V2: shared encoder + local feature correlation for dx/dy regression."""

    def __init__(self, corr_radius: int = 8, feature_channels: int = 128):
        super().__init__()
        self.corr_radius = corr_radius
        self.encoder = SharedFeatureEncoder(out_channels=feature_channels)
        self.correlation = LocalCorrelation(radius=corr_radius)
        corr_channels = self.correlation.num_offsets

        self.regression_head = nn.Sequential(
            nn.Conv2d(corr_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    @staticmethod
    def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.size(1) == 1:
            return rgb
        weights = rgb.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        return (rgb * weights).sum(dim=1, keepdim=True)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        rgb_gray = self.rgb_to_gray(rgb)
        rgb_features = self.encoder(rgb_gray)
        ir_features = self.encoder(ir)
        corr_volume = self.correlation(rgb_features, ir_features)
        return self.regression_head(corr_volume)


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "madnet_v2_corr":
        return MADNetV2Corr(corr_radius=kwargs.get("corr_radius", 8))
    raise ValueError(f"Unknown model_name: {model_name}")
