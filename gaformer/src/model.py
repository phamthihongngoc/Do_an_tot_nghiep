"""
GAFormer – kiến trúc CoAtNet-light cho dữ liệu GADF từ cảm biến IMU.

So sánh với CoAtNet-0 gốc (Bảng 3.6 trong luận án):
  S2-MBConv : L=2 (gốc 3)
  S3-TFM    : L=3 (gốc 5)

Input  : (B, 8, 224, 224)  – ảnh GADF 8 kênh
Output : (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MBConv – Inverted Residual Block (MobileNetV2 style)
# ---------------------------------------------------------------------------

class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 4) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.conv = nn.Sequential(
            # pointwise expand
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            # depthwise 3×3
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            # pointwise project
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Transformer block hoạt động trên feature map 2D
# ---------------------------------------------------------------------------

class TransformerBlock2D(nn.Module):
    """Multi-head self-attention trên không gian đặc trưng 2D (flatten H×W → seq)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)                  # (B, H*W, C)
        normed = self.norm1(seq)
        attn_out, _ = self.attn(normed, normed, normed)
        seq = seq + attn_out
        seq = seq + self.mlp(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# GAFormer (CoAtNet-light)
# ---------------------------------------------------------------------------

class GAFormer(nn.Module):
    """
    5 giai đoạn (mặc định = CoAtNet-light):
      S0 : Conv stem,   D=64,  L=2
      S1 : MBConv,      D=96,  L=2,         stride=2
      S2 : MBConv,      D=192, L=s2_blocks, stride=2
      S3 : Transformer, D=384, L=s3_blocks, stride=2
      S4 : Transformer, D=768, L=2,         stride=2

    CoAtNet-light (đề xuất) : s2_blocks=2, s3_blocks=3
    CoAtNet-0 (gốc)         : s2_blocks=3, s3_blocks=5
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
        s2_blocks: int = 2,
        s3_blocks: int = 3,
    ) -> None:
        super().__init__()

        # S0 – Stem conv (224 → 112)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # S1 – MBConv × 2, 64→96, stride=2  (112 → 56)
        self.s1 = self._mbconv_stage(64, 96, num_blocks=2, stride=2)

        # S2 – MBConv × s2_blocks, 96→192, stride=2  (56 → 28)
        self.s2 = self._mbconv_stage(96, 192, num_blocks=s2_blocks, stride=2)

        # S3 – Transformer × s3_blocks, 192→384, stride=2  (28 → 14)
        self.s3 = self._transformer_stage(192, 384, num_blocks=s3_blocks, stride=2, num_heads=6, dropout=dropout)

        # S4 – Transformer × 2, 384→768, stride=2  (14 → 7)
        self.s4 = self._transformer_stage(384, 768, num_blocks=2, stride=2, num_heads=8, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------

    @staticmethod
    def _mbconv_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(MBConv(
                in_ch=in_ch if i == 0 else out_ch,
                out_ch=out_ch,
                stride=stride if i == 0 else 1,
            ))
        return nn.Sequential(*blocks)

    @staticmethod
    def _transformer_stage(
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
        num_heads: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            # chiếu kênh: in_ch → out_ch
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if stride > 1:
            layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        for _ in range(num_blocks):
            layers.append(TransformerBlock2D(out_ch, num_heads=num_heads, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.stem(x)    # (B,64,112,112)
        x = self.s1(x)      # (B,96, 56, 56)
        x = self.s2(x)      # (B,192,28, 28)
        x = self.s3(x)      # (B,384,14, 14)
        x = self.s4(x)      # (B,768, 7,  7)
        feats = self.pool(x).flatten(1)   # (B,768)
        feats = self.drop(feats)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


def build_gaformer(num_classes: int, in_channels: int = 8, dropout: float = 0.1) -> GAFormer:
    """CoAtNet-light: S2=2 MBConv, S3=3 Transformer."""
    return GAFormer(in_channels=in_channels, num_classes=num_classes, dropout=dropout,
                    s2_blocks=2, s3_blocks=3)


def build_coatnet0(num_classes: int, in_channels: int = 8, dropout: float = 0.1) -> GAFormer:
    """CoAtNet-0 gốc: S2=3 MBConv, S3=5 Transformer."""
    return GAFormer(in_channels=in_channels, num_classes=num_classes, dropout=dropout,
                    s2_blocks=3, s3_blocks=5)
