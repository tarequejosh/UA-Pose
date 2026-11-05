"""SOCA-PRNet Backbone for MMPose.

Implements Spatially Oriented Attention-Infused Structured-Feature-Enabled
PoseResNet (SOCA-PRNet) with Global Context Blocks (GCB) and Spatially Oriented
Contextual Attention (SOCA) modules, built on top of MMPose's ResNet backbone.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule, constant_init, kaiming_init

from mmpose.models.backbones.resnet import ResNet
from mmpose.registry import MODELS


class GlobalContextBlock(BaseModule):
    """Global Context Block (GCNet / Non-Local + SE fusion).

    Captures long-range global spatial dependencies across the entire human
    body to help resolve joint ambiguities under self-occlusion.

    Args:
        in_channels (int): Number of input feature channels.
        reduction (int): Channel reduction ratio in the transform bottleneck.
            Defaults to 4.
    """

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        # Context modeling: 1x1 conv -> spatial softmax attention
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # Feature transform: 1x1 conv -> LayerNorm -> ReLU -> 1x1 conv
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
        )

    def init_weights(self) -> None:
        kaiming_init(self.conv_mask, mode='fan_in')
        constant_init(self.channel_add_conv[-1], 0)

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        # [B, C, H * W]
        input_x = x.view(batch, channels, height * width)
        # [B, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [B, 1, H, W] -> [B, 1, H * W]
        context_mask = self.conv_mask(x).view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        # [B, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [B, 1, C, 1] -> [B, C, 1, 1]
        context = torch.matmul(input_x, context_mask).view(
            batch, channels, 1, 1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.spatial_pool(x)
        channel_add_term = self.channel_add_conv(context)
        return x + channel_add_term


class SpatiallyOrientedContextualAttention(BaseModule):
    """Spatially Oriented Contextual Attention (SOCA) Module.

    Decomposes spatial attention along horizontal (H) and vertical (W)
    orientations (coordinate-aware directional pooling) combined with local
    spatial context (dilated 3x3 conv) to emphasize articulated limb structures
    and suppress background clutter.

    Args:
        in_channels (int): Number of input feature channels.
        reduction (int): Channel reduction factor. Defaults to 8.
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)

        # Directional 1D adaptive pooling along horizontal and vertical axes
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

        # Local spatial selectivity mask (max + avg channel pool -> 7x7 conv)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.size()

        # 1. Spatially oriented coordinate attention
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H + W, 1]
        y = self.shared_conv(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, mid, 1, W]

        attn_h = self.conv_h(x_h).sigmoid()
        attn_w = self.conv_w(x_w).sigmoid()

        oriented_feat = identity * attn_h * attn_w

        # 2. Local spatial focus mask
        avg_out = torch.mean(oriented_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(oriented_feat, dim=1, keepdim=True)
        spatial_mask = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))

        return oriented_feat * spatial_mask + identity


@MODELS.register_module()
class SOCAPRNet(ResNet):
    """SOCA-PRNet Backbone (Spatially Oriented Attention-Infused PoseResNet).

    Extends MMPose's ``ResNet`` (e.g., ResNet-34 or ResNet-50) by injecting
    Global Context Blocks (GCB) and Spatially Oriented Contextual Attention
    (SOCA) modules at the output of specified residual stages. Compatibility
    with standard ``torchvision://resnet34`` or ``torchvision://resnet50``
    pretrained weights is preserved since the residual stages remain intact.

    Args:
        depth (int): Network depth (18, 34, 50, 101, 152). Defaults to 34.
        soca_stages (Sequence[int]): Stage indices (0-indexed) where SOCA + GCB
            modules are applied. Defaults to ``(1, 2, 3)``.
        gcb_reduction (int): Reduction ratio for Global Context Blocks.
            Defaults to 4.
        soca_reduction (int): Reduction ratio for SOCA modules. Defaults to 8.
        dropout_prob (float): Optional 2D spatial dropout probability at the
            final stage for epistemic uncertainty / regularization.
            Defaults to 0.0.
    """

    def __init__(
        self,
        depth: int = 34,
        soca_stages: Sequence[int] = (1, 2, 3),
        gcb_reduction: int = 4,
        soca_reduction: int = 8,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(depth=depth, **kwargs)
        self.soca_stages = set(soca_stages)

        # Compute channel dimensions per stage
        stage_channels = [
            self.base_channels * self.expansion * (2**i)
            for i in range(self.num_stages)
        ]

        self.gcb_modules = nn.ModuleDict()
        self.soca_modules = nn.ModuleDict()

        for stage_idx in range(self.num_stages):
            if stage_idx in self.soca_stages:
                ch = stage_channels[stage_idx]
                self.gcb_modules[f'stage_{stage_idx}'] = GlobalContextBlock(
                    in_channels=ch, reduction=gcb_reduction)
                self.soca_modules[f'stage_{stage_idx}'] = (
                    SpatiallyOrientedContextualAttention(
                        in_channels=ch, reduction=soca_reduction))

        self.dropout = (
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )

    def init_weights(self) -> None:
        super().init_weights()
        for gcb in self.gcb_modules.values():
            gcb.init_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward function with GCB and SOCA enhancement."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            key = f'stage_{i}'
            if key in self.gcb_modules:
                x = self.gcb_modules[key](x)
                x = self.soca_modules[key](x)

            if i == len(self.res_layers) - 1:
                x = self.dropout(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
