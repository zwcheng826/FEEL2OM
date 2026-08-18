"""Final FEEL2OM model architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class ECABlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be a positive odd integer")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        weights = self.pool(x).transpose(-1, -2)
        weights = torch.sigmoid(self.conv(weights)).transpose(-1, -2)
        return x * weights.expand_as(x)


class Feel2OMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fm_proj = nn.Sequential(
            nn.Linear(cfg.rna_fm_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.fm_ln = nn.LayerNorm(cfg.hidden_dim)
        k1, k2, k3 = cfg.cnn_kernel_sizes
        self.cnn = nn.Sequential(
            nn.Conv1d(5, 64, k1, padding="same"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, k2, padding="same"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, cfg.hidden_dim, k3, padding="same"),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.eca = ECABlock1D(cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.bilstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.local_ln = nn.LayerNorm(cfg.hidden_dim)
        self.fused_dim = cfg.hidden_dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim,
            nhead=cfg.num_heads,
            dim_feedforward=self.fused_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.msd = nn.ModuleList([nn.Dropout(cfg.dropout * 0.8) for _ in range(cfg.msd_samples)])
        self.head = nn.Linear(64, 1)

    def forward(self, fm, onehot, mask):
        context = self.fm_ln(self.fm_proj(fm))
        local_channels = self.eca(self.cnn(onehot))
        local = local_channels.transpose(1, 2)
        self.bilstm.flatten_parameters()
        recurrent, _ = self.bilstm(local)
        local = self.local_ln(local + recurrent)
        fused = torch.cat([context, local], dim=-1)
        fused = self.transformer(fused, src_key_padding_mask=mask == 0)
        fused = fused * mask.unsqueeze(-1)
        center = fused[:, self.cfg.seq_len // 2, :]
        mean = fused.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        features = self.classifier(torch.cat([center, mean], dim=-1))
        return torch.stack([self.head(drop(features)) for drop in self.msd]).mean(dim=0)


def build_model(cfg) -> Feel2OMModel:
    return Feel2OMModel(cfg)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
