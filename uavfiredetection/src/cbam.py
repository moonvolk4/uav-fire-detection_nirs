"""
CBAM (Convolutional Block Attention Module) для YOLOv8.

Источник: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention: какие каналы (признаки) важны."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Spatial Attention: где на картинке смотреть."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """
    CBAM: Channel Attention -> Spatial Attention.

    Использует lazy init: при первом forward определяет реальное
    количество каналов и создаёт слои. Это нужно потому что
    ultralytics масштабирует каналы (width multiplier 0.5),
    и значения из YAML не совпадают с реальными.
    """

    def __init__(self, c1=None, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.channel_attn = None
        self.spatial_attn = None
        if c1 is not None:
            self._build(c1)

    def _build(self, c1):
        self.channel_attn = ChannelAttention(c1)
        self.spatial_attn = SpatialAttention(self.kernel_size)

    def forward(self, x):
        if self.channel_attn is None:
            self._build(x.shape[1])
            self.to(x.device)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
