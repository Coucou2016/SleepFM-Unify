"""Optional channel-aware masked attention pooling (stub; not default forward).

Default SleepFM-Unify still zeros padded leads via ``channel_mask`` before the
fixed-in_ch 1D CNN. This module is a drop-in prototype for future variable-channel
work — do not treat it as production-ready.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAwareMaskedPool(nn.Module):
    """Softmax attention over channels with an additive mask for absent leads.

    Input:  x (B, C, T), channel_mask (B, C) with 1=present / 0=absent.
    Output: (B, T) pooled across channels (or (B, out_dim, T) if project=True).
    """

    def __init__(self, in_channels: int, out_dim: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.score = nn.Linear(1, 1, bias=True)
        self.project = None
        if out_dim is not None:
            self.project = nn.Conv1d(1, int(out_dim), kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, C, T)
        if x.ndim != 3:
            raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
        b, c, t = x.shape
        # Per-channel energy as a cheap attention key.
        energy = x.pow(2).mean(dim=-1, keepdim=True)  # (B, C, 1)
        logits = self.score(energy).squeeze(-1)  # (B, C)
        if channel_mask is not None:
            # Large negative for absent channels.
            logits = logits.masked_fill(channel_mask < 0.5, -1e4)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, C, 1)
        pooled = (x * weights).sum(dim=1, keepdim=True)  # (B, 1, T)
        if self.project is not None:
            return self.project(pooled)
        return pooled.squeeze(1)
