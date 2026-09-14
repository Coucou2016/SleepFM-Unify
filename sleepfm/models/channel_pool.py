"""Channel-aware masked attention pooling for padded PSG montages.

Default SleepFM still zeros absent leads via ``channel_mask`` before the fixed
``in_ch`` EffNet. When ``channel_aware_pool`` is enabled, each modality first
applies a learned soft attention over channels (masked so absent leads cannot
contribute), then keeps the same ``(B, C, T)`` layout expected by EffNet.

This is the production path for mask-aware channel attention; hard zeroing
remains available as ``mode=\"zero\"`` / config off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAwareMaskedPool(nn.Module):
    """Softmax attention over channels with an additive mask for absent leads.

    Modes
    -----
    ``reweight`` (default)
        Return ``(B, C, T)`` = ``x * softmax(scores)`` (broadcast). Absent
        channels receive near-zero weight via ``masked_fill``. Preserves the
        fixed montage width so EffNet ``in_channel`` is unchanged.
    ``pool``
        Collapse channels to ``(B, T)`` or ``(B, out_dim, T)`` if ``out_dim``
        is set — useful as a standalone probe / unit-test path.

    Input:  x ``(B, C, T)``, channel_mask ``(B, C)`` with 1=present / 0=absent.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int | None = None,
        mode: str = "reweight",
        hidden: int = 16,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.mode = str(mode)
        if self.mode not in ("reweight", "pool"):
            raise ValueError(f"mode must be 'reweight' or 'pool', got {mode!r}")
        # Score from per-channel summary stats (mean, std, energy) → logits.
        self.scorer = nn.Sequential(
            nn.Linear(3, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        self.project = None
        if out_dim is not None:
            self.project = nn.Conv1d(1, int(out_dim), kernel_size=1)

    def _channel_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, C, 3)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        energy = x.pow(2).mean(dim=-1)
        return torch.stack([mean, std, energy], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"ChannelAwareMaskedPool expected C={self.in_channels}, got {c}"
            )
        feats = self._channel_features(x)  # (B, C, 3)
        logits = self.scorer(feats).squeeze(-1)  # (B, C)
        if channel_mask is not None:
            if channel_mask.shape != (b, c):
                raise ValueError(
                    f"channel_mask shape {tuple(channel_mask.shape)} != {(b, c)}"
                )
            # Large negative for absent channels so softmax ≈ 0.
            logits = logits.masked_fill(channel_mask < 0.5, -1e4)
            # If an entire row is absent, fall back to uniform-zero (all -1e4
            # would NaN); clamp by giving equal tiny mass via identity fill.
            all_absent = channel_mask.sum(dim=-1) < 0.5
            if bool(all_absent.any()):
                logits = logits.clone()
                logits[all_absent] = 0.0
        weights = F.softmax(logits, dim=-1)  # (B, C)
        if channel_mask is not None:
            weights = weights * channel_mask.to(dtype=weights.dtype)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        w = weights.unsqueeze(-1)  # (B, C, 1)

        if self.mode == "reweight":
            # Soft channel gate; absent leads stay ~0 without hard destroy of grads
            # on present leads' relative importance.
            return x * w

        pooled = (x * w).sum(dim=1, keepdim=True)  # (B, 1, T)
        if self.project is not None:
            return self.project(pooled)
        return pooled.squeeze(1)
