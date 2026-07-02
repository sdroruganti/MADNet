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


class MADNetV1_2(nn.Module):
    """MADNet V1.2: 4-channel CNN baseline for dx/dy regression."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


class BasicResidualBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.relu(x)


class MADNetV13Res(nn.Module):
    """MADNet V1.3: residual direct-regression CNN with 8x8 spatial pooling."""

    def __init__(self, pool_size: int = 8, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")

        self.pool_size = int(pool_size)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            nn.Conv2d(4, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c1, c1, stride=1, num_blocks=2)
        self.stage2 = self._make_stage(c1, c2, stride=2, num_blocks=2)
        self.stage3 = self._make_stage(c2, c3, stride=2, num_blocks=2)
        self.stage4 = self._make_stage(c3, c4, stride=2, num_blocks=2)
        self.stage5 = self._make_stage(c4, c5, stride=1, num_blocks=1)

        # 224 -> 112 -> 112 -> 56 -> 28 -> 14 -> 14 -> AdaptiveAvgPool 8x8
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))

        regressor_layers: list[nn.Module] = [
            nn.Linear(c5 * self.pool_size * self.pool_size, 512),
            nn.ReLU(inplace=True),
        ]
        if self.dropout > 0:
            regressor_layers.append(nn.Dropout(self.dropout))
        regressor_layers.extend(
            [
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
            ]
        )
        self.regressor = nn.Sequential(*regressor_layers)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
    ) -> nn.Sequential:
        blocks = [BasicResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


class MADNetV13Lite(MADNetV13Res):
    """MADNet V1.3-lite: V1.3 residual architecture with fewer base channels."""

    def __init__(self, pool_size: int = 8, base_channels: int = 24, dropout: float = 0.0):
        super().__init__(pool_size=pool_size, base_channels=base_channels, dropout=dropout)


class MADNetV13LiteCoord(MADNetV13Lite):
    """MADNet V1.3-lite with normalized x/y coordinate channels."""

    def __init__(self, pool_size: int = 8, base_channels: int = 24, dropout: float = 0.0):
        super().__init__(pool_size=pool_size, base_channels=base_channels, dropout=dropout)
        self.stem = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def coordinate_channels(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape
        x_coords = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        y_coords = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return x_coords, y_coords

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        x_coords, y_coords = self.coordinate_channels(x)
        x = torch.cat([x, x_coords, y_coords], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


class MADNetV15CoarseToFine(nn.Module):
    """MADNet V1.5: two-stage V1.3-lite coarse-to-fine translation regression."""

    def __init__(
        self,
        pool_size: int = 8,
        base_channels: int = 24,
        dropout: float = 0.0,
        x_bound: float = 100.0,
        y_bound: float = 100.0,
    ):
        super().__init__()
        self.x_bound = float(x_bound)
        self.y_bound = float(y_bound)
        self.stage1 = MADNetV13Lite(
            pool_size=pool_size,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.stage2 = MADNetV13Lite(
            pool_size=pool_size,
            base_channels=base_channels,
            dropout=dropout,
        )

    @staticmethod
    def base_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, height, width, 2)

    def warp_ir(self, ir: torch.Tensor, coarse_norm: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = ir.shape
        coarse_px = coarse_norm * coarse_norm.new_tensor([self.x_bound, self.y_bound]).view(1, 2)
        dx_norm_grid = 2.0 * coarse_px[:, 0] / max(width - 1, 1)
        dy_norm_grid = 2.0 * coarse_px[:, 1] / max(height - 1, 1)

        grid = self.base_grid(batch_size, height, width, ir.device, ir.dtype).clone()
        offsets = torch.stack([dx_norm_grid, dy_norm_grid], dim=1).view(batch_size, 1, 1, 2)
        grid = grid - offsets.to(dtype=grid.dtype)
        return F.grid_sample(
            ir,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        coarse = self.stage1(rgb, ir)
        warped_ir = self.warp_ir(ir, coarse)
        residual = self.stage2(rgb, warped_ir)
        self.latest_coarse = coarse.detach()
        self.latest_residual = residual.detach()
        return coarse + residual


class MADNetV13LiteCorrHead(MADNetV13Lite):
    """MADNet V1.3-lite backbone with a local spatial correlation prediction head."""

    def __init__(
        self,
        pool_size: int = 8,
        base_channels: int = 24,
        dropout: float = 0.0,
        corr_radius: int = 20,
        corr_head_channels: int = 128,
        encoder_stride: int = 16,
    ):
        super().__init__(pool_size=pool_size, base_channels=base_channels, dropout=dropout)
        self.corr_radius = int(corr_radius)
        self.corr_head_channels = int(corr_head_channels)
        self.encoder_stride = int(encoder_stride)
        self.feature_corr_radius = max(
            1,
            (self.corr_radius + self.encoder_stride - 1) // self.encoder_stride,
        )
        self.correlation = LocalCorrelation(radius=self.feature_corr_radius)

        corr_channels = self.correlation.num_offsets
        self.corr_head = nn.Sequential(
            nn.Conv2d(corr_channels, self.corr_head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.corr_head_channels, self.corr_head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.corr_head_channels, 2),
        )
        del self.pool
        del self.regressor

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.stage5(x)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        zero_ir = ir.new_zeros(ir.shape)
        zero_rgb = rgb.new_zeros(rgb.shape)
        visible_input = torch.cat([rgb, zero_ir], dim=1)
        infrared_input = torch.cat([zero_rgb, ir], dim=1)

        visible_features = self.encode_features(visible_input)
        infrared_features = self.encode_features(infrared_input)
        corr_volume = self.correlation(visible_features, infrared_features)
        return self.corr_head(corr_volume)


class MADNetV13LiteConfidence(MADNetV13Lite):
    """MADNet V1.3-lite with separate translation and confidence heads."""

    def __init__(self, pool_size: int = 8, base_channels: int = 24, dropout: float = 0.0):
        super().__init__(pool_size=pool_size, base_channels=base_channels, dropout=dropout)

        c5 = base_channels * 16
        body_layers: list[nn.Module] = [
            nn.Linear(c5 * self.pool_size * self.pool_size, 512),
            nn.ReLU(inplace=True),
        ]
        if self.dropout > 0:
            body_layers.append(nn.Dropout(self.dropout))
        body_layers.extend(
            [
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
            ]
        )
        self.regressor_body = nn.Sequential(*body_layers)
        self.translation_head = nn.Linear(128, 2)
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        del self.regressor

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([rgb, ir], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x).flatten(1)
        features = self.regressor_body(x)
        return {
            "translation": self.translation_head(features),
            "confidence": self.confidence_head(features),
        }


class MADNetV14MultiScale(nn.Module):
    """MADNet V1.4: residual direct-regression CNN with multi-scale pooled features."""

    def __init__(self, pool_size: int = 8, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")

        self.pool_size = int(pool_size)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            nn.Conv2d(4, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c1, c1, stride=1, num_blocks=2)
        self.stage2 = self._make_stage(c1, c2, stride=2, num_blocks=2)
        self.stage3 = self._make_stage(c2, c3, stride=2, num_blocks=2)
        self.stage4 = self._make_stage(c3, c4, stride=2, num_blocks=2)
        self.stage5 = self._make_stage(c4, c5, stride=1, num_blocks=1)

        self.pool2 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool3 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool4 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool5 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))

        total_channels = c2 + c3 + c4 + c5
        total_features = total_channels * self.pool_size * self.pool_size
        regressor_layers: list[nn.Module] = [
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
        ]
        if self.dropout > 0:
            regressor_layers.append(nn.Dropout(self.dropout))
        regressor_layers.extend(
            [
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
            ]
        )
        self.regressor = nn.Sequential(*regressor_layers)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
    ) -> nn.Sequential:
        blocks = [BasicResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        features = torch.cat(
            [
                self.pool2(f2).flatten(1),
                self.pool3(f3).flatten(1),
                self.pool4(f4).flatten(1),
                self.pool5(f5).flatten(1),
            ],
            dim=1,
        )
        return self.regressor(features)


class MADNetV14BMultiScaleNorm(nn.Module):
    """MADNet V1.4b: stabilized multi-scale residual CNN with LayerNorm."""

    def __init__(self, pool_size: int = 8, base_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")

        self.pool_size = int(pool_size)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            nn.Conv2d(4, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c1, c1, stride=1, num_blocks=2)
        self.stage2 = self._make_stage(c1, c2, stride=2, num_blocks=2)
        self.stage3 = self._make_stage(c2, c3, stride=2, num_blocks=2)
        self.stage4 = self._make_stage(c3, c4, stride=2, num_blocks=2)
        self.stage5 = self._make_stage(c4, c5, stride=1, num_blocks=1)

        self.pool2 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool3 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool4 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.pool5 = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))

        total_channels = c2 + c3 + c4 + c5
        total_features = total_channels * self.pool_size * self.pool_size
        self.feature_norm = nn.LayerNorm(total_features)
        self.regressor = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
    ) -> nn.Sequential:
        blocks = [BasicResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        features = torch.cat(
            [
                self.pool2(f2).flatten(1),
                self.pool3(f3).flatten(1),
                self.pool4(f4).flatten(1),
                self.pool5(f5).flatten(1),
            ],
            dim=1,
        )
        features = self.feature_norm(features)
        return self.regressor(features)


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


class MADNetV3SoftArgmax(nn.Module):
    """MADNet V3: local correlation with explicit soft-argmax displacement."""

    def __init__(
        self,
        corr_radius: int = 8,
        feature_channels: int = 128,
        encoder_stride: float = 8.0,
        softmax_temperature: float = 0.1,
        x_bound: float = 100.0,
        y_bound: float = 100.0,
    ):
        super().__init__()
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        self.corr_radius = corr_radius
        self.feature_channels = feature_channels
        self.encoder_stride = float(encoder_stride)
        self.softmax_temperature = float(softmax_temperature)
        self.x_bound = float(x_bound)
        self.y_bound = float(y_bound)

        self.encoder = SharedFeatureEncoder(out_channels=feature_channels)
        self.correlation = LocalCorrelation(radius=corr_radius)

        offsets = [
            (dx, dy)
            for dy in range(-corr_radius, corr_radius + 1)
            for dx in range(-corr_radius, corr_radius + 1)
        ]
        offset_tensor = torch.tensor(offsets, dtype=torch.float32)
        self.register_buffer("offset_dx", offset_tensor[:, 0], persistent=False)
        self.register_buffer("offset_dy", offset_tensor[:, 1], persistent=False)

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

        corr_logits = corr_volume.mean(dim=(2, 3))
        offset_probs = F.softmax(corr_logits / self.softmax_temperature, dim=1)
        dx_feature = (offset_probs * self.offset_dx).sum(dim=1)
        dy_feature = (offset_probs * self.offset_dy).sum(dim=1)

        dx_image = dx_feature * self.encoder_stride
        dy_image = dy_feature * self.encoder_stride
        dx_norm = dx_image / self.x_bound
        dy_norm = dy_image / self.y_bound
        return torch.stack([dx_norm, dy_norm], dim=1)


class MADNetV4SpatialCorr(nn.Module):
    """MADNet V4: spatial confidence aggregation over local correlation offsets."""

    def __init__(
        self,
        corr_radius: int = 8,
        feature_channels: int = 64,
        encoder_stride: float = 8.0,
        x_bound: float = 100.0,
        y_bound: float = 100.0,
    ):
        super().__init__()
        self.corr_radius = corr_radius
        self.feature_channels = feature_channels
        self.encoder_stride = float(encoder_stride)
        self.x_bound = float(x_bound)
        self.y_bound = float(y_bound)

        self.encoder = SharedFeatureEncoder(out_channels=feature_channels)
        self.correlation = LocalCorrelation(radius=corr_radius)
        corr_channels = self.correlation.num_offsets

        self.offset_head = nn.Sequential(
            nn.Conv2d(corr_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, corr_channels, kernel_size=1),
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(corr_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        offsets = [
            (dx, dy)
            for dy in range(-corr_radius, corr_radius + 1)
            for dx in range(-corr_radius, corr_radius + 1)
        ]
        offset_tensor = torch.tensor(offsets, dtype=torch.float32)
        self.register_buffer(
            "offsets_dx", offset_tensor[:, 0].view(1, corr_channels, 1, 1), persistent=False
        )
        self.register_buffer(
            "offsets_dy", offset_tensor[:, 1].view(1, corr_channels, 1, 1), persistent=False
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

        offset_logits = self.offset_head(corr_volume)
        spatial_conf_logits = self.conf_head(corr_volume)

        offset_probs = F.softmax(offset_logits, dim=1)
        local_dx = (offset_probs * self.offsets_dx).sum(dim=1, keepdim=True)
        local_dy = (offset_probs * self.offsets_dy).sum(dim=1, keepdim=True)

        batch_size, _, height, width = spatial_conf_logits.shape
        spatial_weights = F.softmax(spatial_conf_logits.flatten(2), dim=2).view(
            batch_size, 1, height, width
        )

        dx_feature = (local_dx * spatial_weights).sum(dim=(2, 3))
        dy_feature = (local_dy * spatial_weights).sum(dim=(2, 3))

        dx_image = dx_feature * self.encoder_stride
        dy_image = dy_feature * self.encoder_stride
        dx_norm = dx_image / self.x_bound
        dy_norm = dy_image / self.y_bound
        return torch.cat([dx_norm, dy_norm], dim=1)


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "madnet_v1_2":
        return MADNetV1_2()
    if model_name == "madnet_v1_3_res":
        return MADNetV13Res(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 32),
            dropout=kwargs.get("dropout", 0.0),
        )
    if model_name == "madnet_v1_3_lite":
        return MADNetV13Lite(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 24),
            dropout=kwargs.get("dropout", 0.0),
        )
    if model_name == "madnet_v1_3_lite_coord":
        return MADNetV13LiteCoord(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 24),
            dropout=kwargs.get("dropout", 0.0),
        )
    if model_name == "madnet_v1_5_coarse_to_fine":
        return MADNetV15CoarseToFine(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 24),
            dropout=kwargs.get("dropout", 0.0),
            x_bound=kwargs.get("x_bound", 100.0),
            y_bound=kwargs.get("y_bound", 100.0),
        )
    if model_name == "madnet_v1_3_lite_corr_head":
        return MADNetV13LiteCorrHead(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 24),
            dropout=kwargs.get("dropout", 0.0),
            corr_radius=kwargs.get("corr_radius", 20),
            corr_head_channels=kwargs.get("corr_head_channels", 128),
            encoder_stride=kwargs.get("encoder_stride", 16),
        )
    if model_name == "madnet_v1_3_lite_confidence":
        return MADNetV13LiteConfidence(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 24),
            dropout=kwargs.get("dropout", 0.0),
        )
    if model_name == "madnet_v1_4_multiscale":
        return MADNetV14MultiScale(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 32),
            dropout=kwargs.get("dropout", 0.0),
        )
    if model_name == "madnet_v1_4b_multiscale_norm":
        return MADNetV14BMultiScaleNorm(
            pool_size=kwargs.get("pool_size", 8),
            base_channels=kwargs.get("base_channels", 16),
            dropout=kwargs.get("dropout", 0.1),
        )
    if model_name == "madnet_v2_corr":
        return MADNetV2Corr(
            corr_radius=kwargs.get("corr_radius", 8),
            feature_channels=kwargs.get("feature_channels", 128),
        )
    if model_name == "madnet_v3_softargmax":
        return MADNetV3SoftArgmax(
            corr_radius=kwargs.get("corr_radius", 8),
            feature_channels=kwargs.get("feature_channels", 128),
            encoder_stride=kwargs.get("encoder_stride", 8),
            softmax_temperature=kwargs.get("softmax_temperature", 0.1),
            x_bound=kwargs.get("x_bound", 100.0),
            y_bound=kwargs.get("y_bound", 100.0),
        )
    if model_name == "madnet_v4_spatial_corr":
        return MADNetV4SpatialCorr(
            corr_radius=kwargs.get("corr_radius", 8),
            feature_channels=kwargs.get("feature_channels", 64),
            encoder_stride=kwargs.get("encoder_stride", 8),
            x_bound=kwargs.get("x_bound", 100.0),
            y_bound=kwargs.get("y_bound", 100.0),
        )
    raise ValueError(f"Unknown model_name: {model_name}")
