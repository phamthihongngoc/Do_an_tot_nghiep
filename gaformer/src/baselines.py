"""
Các mô hình baseline cho so sánh:
  - CNN1D_BiLSTM : 1D-CNN + Bi-LSTM trên tín hiệu IMU thô
  - ResNet50_8ch : ResNet-50 nhận ảnh 8 kênh (GADF/GASF)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1D-CNN + Bi-LSTM
# ---------------------------------------------------------------------------

class CNN1D_BiLSTM(nn.Module):
    """
    Kiến trúc: Conv1D × 3  →  MaxPool × 2  →  BiLSTM × 2  →  FC
    Đầu vào : (B, T, 6)  – tín hiệu thô đã smooth + normalize
    Đầu ra  : (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 15,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)   # 128 * 2 directions

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.cnn(x)                 # (B, 256, T//4)
        x = x.transpose(1, 2)          # (B, T//4, 256)
        out, _ = self.lstm(x)           # (B, T//4, 256)
        out = self.drop(out[:, -1, :])  # lấy bước cuối: (B, 256)
        return self.classifier(out)


def build_cnn_bilstm(num_classes: int, in_channels: int = 6, dropout: float = 0.3) -> CNN1D_BiLSTM:
    return CNN1D_BiLSTM(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


# ---------------------------------------------------------------------------
# ResNet-50 với đầu vào 8 kênh
# ---------------------------------------------------------------------------

def build_resnet50_8ch(num_classes: int, dropout: float = 0.1) -> nn.Module:
    """ResNet-50 nhận ảnh 8 kênh GADF/GASF (không dùng pretrained vì khác RGB)."""
    try:
        from torchvision.models import resnet50
    except ImportError as exc:
        raise ImportError("Cần cài torchvision: pip install torchvision") from exc

    model = resnet50(weights=None)

    # Thay conv đầu tiên: 3 kênh → 8 kênh
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        8, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Thay lớp classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
