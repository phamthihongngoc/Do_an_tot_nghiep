"""Kiến trúc mô hình Deep Learning — tương thích với demo/predict.py.

Interface chuẩn cho mọi model:
    input  : (B, T, C)  — batch × timesteps × channels
    output : (B, num_classes)  — logits chưa qua softmax

Class names được import trực tiếp bởi demo/predict.py:
    from models import CNN1D, LSTMModel, TransformerModel
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. 1D-CNN  (Baseline Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    """Stack of Conv1D-BN-ReLU-MaxPool blocks + global avg pool + FC head."""

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 20,
        conv_configs: list[tuple[int, int]] | None = None,
        dropout: float = 0.40,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        if conv_configs is None:
            conv_configs = [(32, 7), (64, 5), (128, 3)]

        blocks: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, kernel in conv_configs:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        x = x.permute(0, 2, 1)   # → (B, C, T)
        x = self.features(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN + Bi-LSTM  (Hybrid Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """Lightweight CNN encoder → Bidirectional LSTM → FC classifier.

    CNN giảm chiều thời gian và trích xuất đặc trưng cục bộ;
    Bi-LSTM mô hình hoá phụ thuộc dài hạn theo cả 2 chiều.
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 20,
        cnn_channels: tuple[int, ...] = (32, 64),
        cnn_kernel: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.30,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        in_ch = input_channels
        cnn_blocks: list[nn.Module] = []
        for out_ch in cnn_channels:
            cnn_blocks += [
                nn.Conv1d(in_ch, out_ch, cnn_kernel, padding=cnn_kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        self.lstm = nn.LSTM(
            in_ch,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self._bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        x = x.permute(0, 2, 1)           # → (B, C, T)
        x = self.cnn(x)                   # → (B, C', T')
        x = x.permute(0, 2, 1)           # → (B, T', C')
        _, (hn, _) = self.lstm(x)
        # Ghép hidden state cuối của 2 chiều
        h = torch.cat([hn[-2], hn[-1]], dim=-1) if self._bidirectional else hn[-1]
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lightweight Transformer  (Advanced)
# ─────────────────────────────────────────────────────────────────────────────
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    """Pre-norm Transformer Encoder với global average pooling.

    Tham số ~0.2 M — đủ nhẹ cho inference real-time trên CPU.
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_channels, d_model)
        self.pos  = _PositionalEncoding(d_model, max_seq_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (ổn định hơn khi train)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        x = x.mean(dim=1)   # Global average pooling
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, num_classes: int, cfg: dict | None = None) -> nn.Module:
    """Factory function — dùng bởi cả training scripts và demo/predict.py."""
    name = name.lower()
    if cfg is None:
        cfg = {}
    m = cfg.get("models", {}).get(name, {})

    if name == "cnn":
        return CNN1D(num_classes=num_classes,
                     conv_configs=m.get("conv_configs"),
                     dropout=m.get("dropout", 0.40),
                     fc_hidden=m.get("fc_hidden", 128))
    if name == "lstm":
        return LSTMModel(num_classes=num_classes,
                         cnn_channels=tuple(m.get("cnn_channels", [32, 64])),
                         cnn_kernel=m.get("cnn_kernel", 5),
                         lstm_hidden=m.get("lstm_hidden", 128),
                         lstm_layers=m.get("lstm_layers", 2),
                         bidirectional=m.get("bidirectional", True),
                         dropout=m.get("dropout", 0.30),
                         fc_hidden=m.get("fc_hidden", 128))
    if name == "transformer":
        return TransformerModel(num_classes=num_classes,
                                d_model=m.get("d_model", 64),
                                nhead=m.get("nhead", 4),
                                num_layers=m.get("num_layers", 3),
                                dim_feedforward=m.get("dim_feedforward", 256),
                                dropout=m.get("dropout", 0.10),
                                max_seq_len=m.get("max_seq_len", 256))
    raise ValueError(f"Unknown model name: {name!r}. Use 'cnn', 'lstm', or 'transformer'.")
