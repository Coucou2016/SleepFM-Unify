"""Night-level temporal encoder over epoch embeddings (SleepFM-Unify)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sleepfm.models.sleepfm import symmetric_infonce


def _infer_n_layers(state: dict, kind: str) -> int:
    if kind == "transformer":
        ids = []
        for key in state:
            if ".layers." not in key:
                continue
            try:
                ids.append(int(key.split(".layers.")[1].split(".")[0]))
            except (IndexError, ValueError):
                continue
        return max(ids) + 1 if ids else 1
    ids = []
    for key in state:
        if "weight_ih_l" not in key or key.endswith("_reverse"):
            continue
        tail = key.split("weight_ih_l", 1)[1]
        num = tail.split("_")[0]
        try:
            ids.append(int(num))
        except ValueError:
            continue
    return max(ids) + 1 if ids else 1


class NightTemporalEncoder(nn.Module):
    """Small GRU or Transformer over a window of epoch embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        kind: str = "gru",
        dropout: float = 0.1,
        max_len: int = 512,
        window: Optional[int] = None,
    ):
        super().__init__()
        self.kind = kind
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.window = int(window) if window is not None else None
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.core = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.out = nn.Identity()
        else:
            self.core = nn.GRU(
                d_model,
                d_model,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.pos = None
            self.out = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        padding_mask: (B, L) True = pad
        input_mask: (B, L) True = replace with mask token (masked epoch prediction)
        """
        h = x
        if input_mask is not None:
            h = x.clone()
            h[input_mask] = self.mask_token.to(dtype=h.dtype)
        if self.kind == "transformer":
            h = h + self.pos[:, : h.size(1)]
            h = self.core(h, src_key_padding_mask=padding_mask)
            return h
        if padding_mask is not None:
            # Zero pads and pack so bidirectional GRU state does not leak across pads.
            h = h.clone()
            h[padding_mask.bool()] = 0
            lengths = (~padding_mask.bool()).sum(dim=1).clamp(min=1).to(dtype=torch.long)
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.core(packed)
            out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=h.size(1)
            )
            return self.out(out)
        out, _ = self.core(h)
        return self.out(out)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_or_path: Union[str, Path, dict, None],
        device: str = "cpu",
    ) -> Optional["NightTemporalEncoder"]:
        """Reload a trained temporal head, or return None if the checkpoint has none."""
        if ckpt_or_path is None:
            return None
        if isinstance(ckpt_or_path, dict):
            ckpt = ckpt_or_path
        else:
            ckpt = torch.load(str(ckpt_or_path), map_location=device, weights_only=False)
        state = ckpt.get("temporal_state_dict")
        if not state:
            return None
        cfg = dict(ckpt.get("temporal_cfg") or {})
        kind = cfg.get("kind")
        if kind is None:
            kind = (
                "transformer"
                if any("self_attn" in k or k == "pos" or k.startswith("pos") for k in state)
                else "gru"
            )
        d_model = cfg.get("d_model")
        if d_model is None and "mask_token" in state:
            d_model = int(state["mask_token"].shape[-1])
        n_layers = cfg.get("n_layers") or _infer_n_layers(state, str(kind))
        n_heads = int(cfg.get("n_heads") or 4)
        max_len = int(cfg.get("max_len") or 512)
        window = cfg.get("window")
        enc = cls(
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=n_heads,
            kind=str(kind),
            max_len=max_len,
            window=window,
        )
        enc.load_state_dict(state, strict=True)
        enc.to(device)
        enc.eval()
        return enc


def contextualize_sequence(
    encoder: NightTemporalEncoder,
    seq: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the temporal encoder over an epoch sequence ``(L, D)`` or ``(B, L, D)``."""
    squeeze = False
    if seq.ndim == 2:
        seq = seq.unsqueeze(0)
        squeeze = True
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
    max_len = int(encoder.max_len) if encoder.kind == "transformer" else seq.size(1)
    length = seq.size(1)
    if encoder.kind == "transformer" and length > max_len:
        pieces = []
        for start in range(0, length, max_len):
            sl = seq[:, start : start + max_len]
            pm = None if padding_mask is None else padding_mask[:, start : start + max_len]
            pieces.append(encoder(sl, padding_mask=pm))
        ctx = torch.cat(pieces, dim=1)
    else:
        ctx = encoder(seq, padding_mask=padding_mask)
    if squeeze:
        return ctx.squeeze(0)
    return ctx


def temporal_contrastive_loss(
    ctx: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    temp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Adjacent-epoch InfoNCE on contextual embeddings (B, L, D)."""
    if ctx.size(1) < 2:
        return ctx.new_zeros(())
    q = ctx[:, :-1]
    k = ctx[:, 1:]
    if padding_mask is not None:
        valid = (~padding_mask[:, :-1]) & (~padding_mask[:, 1:])
        q = q[valid]
        k = k[valid]
    else:
        q = q.reshape(-1, ctx.size(-1))
        k = k.reshape(-1, ctx.size(-1))
    if q.size(0) < 2:
        return ctx.new_zeros(())
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    if temp is None:
        temp = q.new_tensor(1.0)
    return symmetric_infonce(q, k, temp)


def masked_epoch_prediction_loss(
    encoder: NightTemporalEncoder,
    x: torch.Tensor,
    mask_prob: float = 0.15,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE: reconstruct masked epoch embeddings (stop-grad targets)."""
    b, length, _ = x.shape
    input_mask = torch.rand(b, length, device=x.device) < mask_prob
    if padding_mask is not None:
        input_mask = input_mask & (~padding_mask)
    if not bool(input_mask.any()):
        return x.new_zeros(())
    ctx = encoder(x, padding_mask=padding_mask, input_mask=input_mask)
    target = x.detach()
    return F.mse_loss(ctx[input_mask], target[input_mask])


def temporal_losses(
    encoder: NightTemporalEncoder,
    seq: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    mask_prob: float = 0.15,
    temp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    ctx = encoder(seq, padding_mask=padding_mask)
    l_adj = temporal_contrastive_loss(ctx, padding_mask=padding_mask, temp=temp)
    l_mask = masked_epoch_prediction_loss(
        encoder, seq, mask_prob=mask_prob, padding_mask=padding_mask
    )
    loss = l_adj + l_mask
    logs = {
        "loss_temporal_adj": float(l_adj.detach().item()),
        "loss_temporal_mask": float(l_mask.detach().item()),
        "loss_temporal": float(loss.detach().item()),
    }
    return loss, logs
