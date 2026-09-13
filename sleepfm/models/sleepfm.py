"""Multi-modal SleepFM with per-modality encoders (baseline + SleepFM-Unify)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleepfm.models.encoders import EffNet

ModalityName = Literal["bas", "ecg", "respiratory"]
ContrastiveMode = Literal["pairwise", "leave_one_out"]
DownstreamSpace = Literal["concat", "shared", "private"]


def symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE; a, b are (N, D). Returns 0 if N < 2."""
    if a.size(0) < 2:
        return a.new_zeros(())
    logits = torch.matmul(a, b.T) * temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def orthogonality_loss(
    shared: torch.Tensor,
    private: torch.Tensor,
    sample_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """||Zs^T Zp||_F^2 on centered, column-normalized features (finite for small N)."""
    if sample_mask is not None:
        keep = sample_mask > 0.5
        if keep.sum() < 2:
            return shared.new_zeros(())
        shared = shared[keep]
        private = private[keep]
    if shared.size(0) < 2:
        return shared.new_zeros(())
    s = shared - shared.mean(dim=0, keepdim=True)
    p = private - private.mean(dim=0, keepdim=True)
    s = s / s.norm(dim=0, keepdim=True).clamp_min(1e-6)
    p = p / p.norm(dim=0, keepdim=True).clamp_min(1e-6)
    gram = s.transpose(0, 1) @ p
    return gram.pow(2).mean()


class MultiModalSleepFM(nn.Module):
    """Three encoders (BAS/EEG, ECG, respiratory) + contrastive training interface.

    SleepFM-Unify (``unify=True``): each encoder keeps a 512-d backbone, then
    ``proj_shared`` / ``proj_private`` (default 256+256). Contrastive losses use
    **shared** embeddings only. Downstream concat is shared||private per modality
    (still 512-d, paper-comparable) unless ``downstream_space`` is overridden.
    """

    MODALITY_ORDER: List[ModalityName] = ["bas", "ecg", "respiratory"]

    def __init__(
        self,
        channels: Dict[str, int],
        embedding_dim: int = 512,
        temperature_init: float = 0.0,
        unify: bool = False,
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        downstream_space: DownstreamSpace = "concat",
    ):
        super().__init__()
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.unify = bool(unify)
        self.shared_dim = int(shared_dim if shared_dim is not None else (256 if self.unify else embedding_dim))
        self.private_dim = int(private_dim if private_dim is not None else (256 if self.unify else 0))
        self.downstream_space: DownstreamSpace = downstream_space
        self.temperature = nn.Parameter(torch.tensor(temperature_init, dtype=torch.float32))

        self.encoders = nn.ModuleDict(
            {
                name: EffNet(in_channel=channels[name], embedding_dim=embedding_dim)
                for name in self.MODALITY_ORDER
                if name in channels
            }
        )
        if self.unify:
            self.proj_shared = nn.ModuleDict(
                {
                    name: nn.Linear(embedding_dim, self.shared_dim)
                    for name in self.encoders
                }
            )
            self.proj_private = nn.ModuleDict(
                {
                    name: nn.Linear(embedding_dim, self.private_dim)
                    for name in self.encoders
                }
            )
        else:
            self.proj_shared = None
            self.proj_private = None

    @property
    def downstream_dim_per_modality(self) -> int:
        if not self.unify or self.downstream_space == "concat":
            if self.unify:
                return self.shared_dim + self.private_dim
            return self.embedding_dim
        if self.downstream_space == "shared":
            return self.shared_dim
        return self.private_dim

    def _signal_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for name in self.MODALITY_ORDER:
            if name in batch and name in self.encoders and torch.is_tensor(batch[name]):
                out[name] = batch[name]
        return out

    def _present_mask(
        self,
        batch: Dict[str, torch.Tensor],
        signals: Dict[str, torch.Tensor],
        seq_shape: Optional[Tuple[int, int]],
    ) -> Optional[torch.Tensor]:
        mask = batch.get("present_mask")
        if mask is None:
            return None
        if seq_shape is not None and mask.ndim == 3:
            b, l = seq_shape
            mask = mask.reshape(b * l, mask.size(-1))
        elif seq_shape is not None and mask.ndim == 2 and mask.size(0) == seq_shape[0]:
            mask = mask.unsqueeze(1).expand(seq_shape[0], seq_shape[1], mask.size(-1))
            mask = mask.reshape(seq_shape[0] * seq_shape[1], mask.size(-1))
        return mask

    def _flatten_signals(
        self, signals: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple[int, int]]]:
        seq_shape = None
        flat: Dict[str, torch.Tensor] = {}
        for name, tensor in signals.items():
            if tensor.ndim == 4:
                b, l, c, t = tensor.shape
                seq_shape = (b, l)
                flat[name] = tensor.reshape(b * l, c, t)
            else:
                flat[name] = tensor
        return flat, seq_shape

    def encode_backbone(
        self,
        batch: Dict[str, torch.Tensor],
        present_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode each modality; skip missing rows so BatchNorm is not polluted."""
        signals = self._signal_batch(batch)
        flat, seq_shape = self._flatten_signals(signals)
        if present_mask is None:
            present_mask = self._present_mask(batch, signals, seq_shape)
        channel_mask = batch.get("channel_mask")
        out: Dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in flat:
                continue
            x = flat[name]
            # Channel mask (DONE vs TODO):
            # DONE — dataset/collate emit per-lead masks; here we zero padded/absent
            #   leads before the 1D CNN so BatchNorm/conv do not treat pad as signal.
            # TODO — true variable-channel encoders / mask-aware channel attention or
            #   pooling that resizes the channel axis (beyond zero-fill + fixed in_ch).
            if isinstance(channel_mask, dict) and name in channel_mask:
                cm = channel_mask[name]
                if seq_shape is not None and cm.ndim == 2 and cm.size(0) == seq_shape[0]:
                    cm = cm.unsqueeze(1).expand(seq_shape[0], seq_shape[1], cm.size(-1))
                    cm = cm.reshape(seq_shape[0] * seq_shape[1], cm.size(-1))
                if cm.ndim == 2 and x.ndim == 3 and cm.size(0) == x.size(0):
                    x = x * cm.unsqueeze(-1).to(dtype=x.dtype)
            bsz = x.size(0)
            dim = self.embedding_dim
            if present_mask is None:
                out[name] = encoder(x)
                continue
            col = self.MODALITY_ORDER.index(name)
            valid = present_mask[:, col] > 0.5
            emb = x.new_zeros(bsz, dim)
            if bool(valid.any()):
                emb[valid] = encoder(x[valid])
            out[name] = emb
        return out

    def encode_factorized(
        self,
        batch: Dict[str, torch.Tensor],
        normalize: bool = True,
        present_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        signals = self._signal_batch(batch)
        flat, seq_shape = self._flatten_signals(signals)
        if present_mask is None:
            present_mask = self._present_mask(batch, signals, seq_shape)
        # Prefer already-flattened tensors when callers pass flat dicts; keep channel_mask.
        enc_batch: Dict[str, torch.Tensor] = dict(flat) if flat else dict(batch)
        if "channel_mask" in batch:
            enc_batch["channel_mask"] = batch["channel_mask"]
        backbone = self.encode_backbone(enc_batch, present_mask=present_mask)
        shared: Dict[str, torch.Tensor] = {}
        private: Dict[str, torch.Tensor] = {}
        for name, h in backbone.items():
            if self.unify:
                s = self.proj_shared[name](h)
                p = self.proj_private[name](h)
            else:
                s = h
                p = h.new_zeros(h.size(0), 0)
            if normalize:
                s = F.normalize(s, dim=-1)
                if p.numel() > 0:
                    p = F.normalize(p, dim=-1)
            # Missing rows stay exactly zero (do not L2-normalize zero vectors to NaN).
            if present_mask is not None:
                col = self.MODALITY_ORDER.index(name)
                keep = (present_mask[:, col] > 0.5).unsqueeze(-1).to(dtype=s.dtype)
                s = s * keep
                if p.numel() > 0:
                    p = p * keep
            shared[name] = s
            private[name] = p
        return shared, private

    def encode(
        self,
        batch: Dict[str, torch.Tensor],
        normalize: bool = True,
        space: str = "downstream",
    ) -> Dict[str, torch.Tensor]:
        signals = self._signal_batch(batch)
        flat, seq_shape = self._flatten_signals(signals)
        present_mask = self._present_mask(batch, signals, seq_shape)

        if not self.unify or space == "backbone":
            embeddings = self.encode_backbone(batch, present_mask=present_mask)
            if normalize:
                out = {}
                for k, v in embeddings.items():
                    if present_mask is not None:
                        col = self.MODALITY_ORDER.index(k)
                        keep = present_mask[:, col] > 0.5
                        n = F.normalize(v, dim=-1)
                        n = torch.where(keep.unsqueeze(-1), n, torch.zeros_like(n))
                        out[k] = n
                    else:
                        out[k] = F.normalize(v, dim=-1)
                embeddings = out
            return embeddings

        shared, private = self.encode_factorized(
            batch, normalize=normalize, present_mask=present_mask
        )
        if space in ("shared", "contrastive"):
            return shared
        if space == "private":
            return private
        use = self.downstream_space if space == "downstream" else space
        if use == "shared":
            return shared
        if use == "private":
            return private
        return {name: torch.cat([shared[name], private[name]], dim=-1) for name in shared}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encode(batch)

    def _emb_list(
        self,
        encoded: Dict[str, torch.Tensor],
        present_mask: Optional[torch.Tensor],
        drop_names: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], List[str]]:
        drop = set(drop_names or [])
        names = [m for m in self.MODALITY_ORDER if m in encoded and m not in drop]
        embs = [encoded[m] for m in names]
        mask = None
        if present_mask is not None:
            idx = [self.MODALITY_ORDER.index(m) for m in names]
            mask = present_mask[:, idx]
            keep = []
            keep_embs = []
            keep_cols = []
            for i, name in enumerate(names):
                if mask[:, i].sum() > 0:
                    keep.append(name)
                    keep_embs.append(embs[i])
                    keep_cols.append(i)
            names, embs = keep, keep_embs
            mask = mask[:, keep_cols] if keep_cols else None
        return embs, mask, names

    def _run_contrastive(
        self,
        encoded: Dict[str, torch.Tensor],
        present_mask: Optional[torch.Tensor],
        mode: ContrastiveMode,
    ) -> tuple[torch.Tensor, dict]:
        emb_list, mask, _names = self._emb_list(encoded, present_mask)
        n_mod = len(emb_list)
        temp = torch.exp(self.temperature.clamp(min=0.0))
        if n_mod == 0:
            ref = next(iter(self.encoders.values())).fc.weight
            return ref.new_zeros(()), {"mode": mode, "num_modalities": 0}
        # Single modality: LOO has no "others"; fall back to pairwise (no-op) → zero.
        if n_mod == 1:
            ref = emb_list[0]
            return ref.new_zeros(()), {
                "mode": mode,
                "num_modalities": 1,
                "note": "single_modality_skipped",
            }
        if mode == "pairwise":
            return self._pairwise_loss(emb_list, temp, n_mod, present_mask=mask)
        if mode == "leave_one_out":
            return self._leave_one_out_loss(emb_list, temp, n_mod, present_mask=mask)
        raise ValueError(f"Unknown contrastive mode: {mode}")

    def contrastive_loss(
        self,
        batch: Dict[str, torch.Tensor],
        mode: ContrastiveMode = "leave_one_out",
        present_mask: Optional[torch.Tensor] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict]:
        signals = self._signal_batch(batch)
        flat, seq_shape = self._flatten_signals(signals)
        if present_mask is None:
            present_mask = self._present_mask(batch, signals, seq_shape)
        pad = batch.get("padding_mask")
        if pad is not None and seq_shape is not None:
            b, l = seq_shape
            pad_flat = pad.reshape(b, l)
            if present_mask is None:
                present_mask = torch.ones(
                    b * l, len(self.MODALITY_ORDER), device=pad.device, dtype=torch.float32
                )
            present_mask = present_mask.clone()
            present_mask[pad_flat.reshape(-1)] = 0

        if encoded is None:
            enc_in = dict(flat)
            if "channel_mask" in batch:
                enc_in["channel_mask"] = batch["channel_mask"]
            if self.unify:
                encoded, _private = self.encode_factorized(
                    enc_in, normalize=True, present_mask=present_mask
                )
            else:
                if present_mask is not None:
                    enc_in["present_mask"] = present_mask
                encoded = self.encode(enc_in, normalize=True)

        return self._run_contrastive(encoded, present_mask, mode)

    def _pairwise_loss(
        self,
        emb_list: List[torch.Tensor],
        temp: torch.Tensor,
        n_mod: int,
        present_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        pairs = [(i, j) for i in range(n_mod) for j in range(i + 1, n_mod)]
        if present_mask is None or bool((present_mask > 0.5).all()):
            loss = 0.0
            for i, j in pairs:
                logits_ij = torch.matmul(emb_list[i], emb_list[j].T) * temp
                logits_ji = logits_ij.T
                labels = torch.arange(logits_ij.size(0), device=logits_ij.device)
                loss = loss + F.cross_entropy(logits_ij, labels)
                loss = loss + F.cross_entropy(logits_ji, labels)
            loss = loss / max(len(pairs) * 2, 1)
            return loss, {"mode": "pairwise", "num_pairs": len(pairs)}

        loss = emb_list[0].new_zeros(())
        n_terms = 0
        for i, j in pairs:
            valid = (present_mask[:, i] > 0.5) & (present_mask[:, j] > 0.5)
            if int(valid.sum()) < 2:
                continue
            loss = loss + symmetric_infonce(emb_list[i][valid], emb_list[j][valid], temp)
            n_terms += 1
        if n_terms == 0:
            return loss, {"mode": "pairwise", "num_pairs": 0}
        return loss / n_terms, {"mode": "pairwise", "num_pairs": n_terms}

    def _leave_one_out_loss(
        self,
        emb_list: List[torch.Tensor],
        temp: torch.Tensor,
        n_mod: int,
        present_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        if n_mod < 2:
            return emb_list[0].new_zeros(()), {
                "mode": "leave_one_out",
                "num_modalities": n_mod,
                "note": "single_modality_skipped",
            }
        if present_mask is None or bool((present_mask > 0.5).all()):
            loss = 0.0
            for i in range(n_mod):
                others = [emb_list[j] for j in range(n_mod) if j != i]
                other_emb = torch.stack(others, dim=0).mean(dim=0)
                logits_ij = torch.matmul(emb_list[i], other_emb.T) * temp
                logits_ji = logits_ij.T
                labels = torch.arange(logits_ij.size(0), device=logits_ij.device)
                loss = loss + F.cross_entropy(logits_ij, labels)
                loss = loss + F.cross_entropy(logits_ji, labels)
            loss = loss / max(n_mod * 2, 1)
            return loss, {"mode": "leave_one_out", "num_modalities": n_mod}

        stacked = torch.stack(emb_list, dim=1)
        loss = emb_list[0].new_zeros(())
        n_terms = 0
        for i in range(n_mod):
            others_mask = present_mask.clone()
            others_mask[:, i] = 0
            weights = others_mask.unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1e-6)
            other_emb = (stacked * weights).sum(dim=1) / denom
            valid = (present_mask[:, i] > 0.5) & (others_mask.sum(dim=1) > 0.5)
            if int(valid.sum()) < 2:
                continue
            loss = loss + symmetric_infonce(emb_list[i][valid], other_emb[valid], temp)
            n_terms += 1
        if n_terms == 0:
            # All samples are single-modality: safe no-op (pairwise needs ≥2 mods too).
            return loss, {
                "mode": "leave_one_out",
                "num_modalities": 0,
                "note": "no_loo_pairs_single_modality_samples",
            }
        return loss / n_terms, {"mode": "leave_one_out", "num_modalities": n_terms}

    def _apply_modality_dropout(
        self,
        present_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        p: float,
        mode: str = "batch",
    ) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[torch.Tensor]]:
        """
        Apply modality dropout.

        Returns (new_mask, dropped_name_or_None, original_present_mask).
        ``original_present_mask`` is a clone taken *before* dropout (for L_miss).

        mode:
          - ``"batch"``: drop one column for the whole batch (legacy ablation).
          - ``"sample"``: per-sample random non-empty subset of originally present mods.
        """
        if present_mask is None:
            original = torch.ones(
                batch_size, len(self.MODALITY_ORDER), device=device, dtype=torch.float32
            )
        else:
            original = present_mask.clone()

        if p <= 0 or not self.training:
            return present_mask if present_mask is not None else original, None, original

        if mode == "sample":
            # Per-row: keep a random non-empty subset of originally present modalities.
            new_mask = original.clone()
            for i in range(batch_size):
                if torch.rand((), device=device) >= p:
                    continue
                present_idx = (original[i] > 0.5).nonzero(as_tuple=False).view(-1)
                if present_idx.numel() == 0:
                    continue
                # 2^K - 1 non-empty subsets; sample uniformly among them.
                k = int(present_idx.numel())
                choice = int(torch.randint(1, 2**k, (1,), device=device).item())
                keep = torch.zeros(len(self.MODALITY_ORDER), device=device)
                for bit, col in enumerate(present_idx.tolist()):
                    if choice & (1 << bit):
                        keep[col] = 1.0
                new_mask[i] = keep
            return new_mask, "sample_wise", original

        # Batch-level: drop one shared column with probability p.
        if torch.rand((), device=device) >= p:
            return original.clone(), None, original
        present_mask = original.clone()
        col_present = (present_mask > 0.5).any(dim=0).nonzero(as_tuple=False).view(-1)
        if col_present.numel() < 2:
            return present_mask, None, original
        pick = col_present[torch.randint(0, col_present.numel(), (1,), device=device)]
        idx = int(pick.item())
        present_mask[:, idx] = 0
        return present_mask, self.MODALITY_ORDER[idx], original

    def _private_variance_loss(
        self,
        private: Dict[str, torch.Tensor],
        present_mask: Optional[torch.Tensor],
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """VICReg-style variance hinge on private embeddings (anti-collapse)."""
        ref = next(iter(private.values()))
        total = ref.new_zeros(())
        n = 0
        for i, name in enumerate(self.MODALITY_ORDER):
            if name not in private or private[name].size(-1) == 0:
                continue
            p = private[name]
            if present_mask is not None:
                keep = present_mask[:, i] > 0.5
                if int(keep.sum()) < 2:
                    continue
                p = p[keep]
            if p.size(0) < 2:
                continue
            std = torch.sqrt(p.var(dim=0, unbiased=False) + 1e-4)
            total = total + F.relu(1.0 - std).mean()
            n += 1
        if n == 0:
            return ref.new_zeros(())
        return total / n

    def pretrain_loss(
        self,
        batch: Dict[str, torch.Tensor],
        mode: ContrastiveMode = "leave_one_out",
        loss_weights: Optional[Dict[str, float]] = None,
        modality_dropout: float = 0.0,
        modality_dropout_mode: str = "sample",
        temporal_encoder: Optional[nn.Module] = None,
        temporal_mask_prob: float = 0.15,
    ) -> tuple[torch.Tensor, dict]:
        """Mixed Unify loss: LOO + pairwise + orth + miss + private + optional temporal."""
        weights = {
            "loo": 1.0 if mode == "leave_one_out" else 0.0,
            "pairwise": 1.0 if mode == "pairwise" else 0.0,
            "orth": 0.0,
            "temporal": 0.0,
            "miss": 0.0,
            "private": 0.0,
        }
        if loss_weights:
            weights.update({k: float(v) for k, v in loss_weights.items()})

        signals = self._signal_batch(batch)
        flat, seq_shape = self._flatten_signals(signals)
        present_mask = self._present_mask(batch, signals, seq_shape)
        ref = next(iter(flat.values()))
        bsz = ref.size(0)
        present_mask, dropped, original_present_mask = self._apply_modality_dropout(
            present_mask,
            bsz,
            ref.device,
            modality_dropout,
            mode=modality_dropout_mode,
        )
        pad = batch.get("padding_mask")
        if pad is not None and seq_shape is not None:
            if present_mask is None:
                present_mask = torch.ones(
                    bsz, len(self.MODALITY_ORDER), device=ref.device, dtype=torch.float32
                )
            present_mask = present_mask.clone()
            present_mask[pad.reshape(-1)] = 0
            original_present_mask = original_present_mask.clone()
            original_present_mask[pad.reshape(-1)] = 0

        # Encode with *original* presence so dropout teachers stay real and BN skips
        # only naturally missing rows. Contrastive / LOO use post-dropout ``present_mask``.
        enc_in = dict(flat)
        if "channel_mask" in batch:
            enc_in["channel_mask"] = batch["channel_mask"]
        shared, private = self.encode_factorized(
            enc_in, normalize=True, present_mask=original_present_mask
        )

        logs: dict = {
            "dropped_modality": dropped,
            "modality_dropout_mode": modality_dropout_mode,
        }

        total = ref.new_zeros(())
        if weights.get("loo", 0) > 0:
            loo, meta = self._run_contrastive(shared, present_mask, "leave_one_out")
            logs.update({f"leave_one_out_{k}": v for k, v in meta.items()})
            logs["loss_loo"] = float(loo.detach().item())
            total = total + weights["loo"] * loo
        if weights.get("pairwise", 0) > 0:
            pw, meta = self._run_contrastive(shared, present_mask, "pairwise")
            logs.update({f"pairwise_{k}": v for k, v in meta.items()})
            logs["loss_pairwise"] = float(pw.detach().item())
            total = total + weights["pairwise"] * pw

        if weights.get("miss", 0) > 0 and dropped is not None and dropped != "sample_wise":
            remain_names = [m for m in self.MODALITY_ORDER if m in shared and m != dropped]
            if remain_names and dropped in shared:
                remain_stack = torch.stack([shared[m] for m in remain_names], dim=1)
                drop_idx = self.MODALITY_ORDER.index(dropped)
                remain_idx = [self.MODALITY_ORDER.index(m) for m in remain_names]
                w = present_mask[:, remain_idx].to(dtype=remain_stack.dtype)
                w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pred = (remain_stack * w.unsqueeze(-1)).sum(dim=1) / w_sum
                remain_ok = w.sum(dim=1) > 0.5
                orig_ok = original_present_mask[:, drop_idx] > 0.5
                valid = remain_ok & orig_ok
                if bool(valid.any()) and int(valid.sum()) >= 2:
                    temp = torch.exp(self.temperature.clamp(min=0.0))
                    miss = symmetric_infonce(
                        pred[valid], shared[dropped][valid], temp
                    )
                    logs["loss_miss"] = float(miss.detach().item())
                    total = total + weights["miss"] * miss
                else:
                    logs["loss_miss"] = 0.0
        elif weights.get("miss", 0) > 0 and dropped == "sample_wise":
            miss_acc = ref.new_zeros(())
            n_miss = 0
            temp = torch.exp(self.temperature.clamp(min=0.0))
            for drop_name in self.MODALITY_ORDER:
                if drop_name not in shared:
                    continue
                drop_idx = self.MODALITY_ORDER.index(drop_name)
                was_present = original_present_mask[:, drop_idx] > 0.5
                now_missing = present_mask[:, drop_idx] < 0.5
                remain_names = [
                    m for m in self.MODALITY_ORDER if m in shared and m != drop_name
                ]
                if not remain_names:
                    continue
                remain_idx = [self.MODALITY_ORDER.index(m) for m in remain_names]
                remain_stack = torch.stack([shared[m] for m in remain_names], dim=1)
                w = present_mask[:, remain_idx].to(dtype=remain_stack.dtype)
                remain_ok = w.sum(dim=1) > 0.5
                valid = was_present & now_missing & remain_ok
                if int(valid.sum()) < 2:
                    continue
                w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pred = (remain_stack * w.unsqueeze(-1)).sum(dim=1) / w_sum
                miss_acc = miss_acc + symmetric_infonce(
                    pred[valid], shared[drop_name][valid], temp
                )
                n_miss += 1
            if n_miss:
                miss_acc = miss_acc / n_miss
                logs["loss_miss"] = float(miss_acc.detach().item())
                total = total + weights["miss"] * miss_acc
            else:
                logs["loss_miss"] = 0.0

        if self.unify and weights.get("orth", 0) > 0:
            orth = ref.new_zeros(())
            n_o = 0
            for i, name in enumerate(self.MODALITY_ORDER):
                if name not in shared or private[name].size(-1) == 0:
                    continue
                col = None if present_mask is None else present_mask[:, i]
                orth = orth + orthogonality_loss(shared[name], private[name], col)
                n_o += 1
            if n_o:
                orth = orth / n_o
            logs["loss_orth"] = float(orth.detach().item())
            total = total + weights["orth"] * orth

        if self.unify and weights.get("private", 0) > 0:
            priv = self._private_variance_loss(private, present_mask)
            logs["loss_private"] = float(priv.detach().item())
            total = total + weights["private"] * priv

        if temporal_encoder is not None and seq_shape is not None and weights.get("temporal", 0) > 0:
            from sleepfm.models.temporal import temporal_losses

            b, l = seq_shape
            parts = []
            for name in self.MODALITY_ORDER:
                if name in shared:
                    parts.append(shared[name].view(b, l, -1))
            if parts:
                seq = torch.stack(parts, dim=0).mean(dim=0)
                t_loss, t_logs = temporal_losses(
                    temporal_encoder,
                    seq,
                    padding_mask=pad if pad is None else pad.reshape(b, l),
                    mask_prob=temporal_mask_prob,
                    temp=torch.exp(self.temperature.clamp(min=0.0)),
                )
                logs.update(t_logs)
                total = total + weights["temporal"] * t_loss

        logs["loss_total"] = float(total.detach().item()) if total.numel() else 0.0
        return total, logs

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        channels: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ) -> "MultiModalSleepFM":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        from sleepfm.models.official_adapter import (
            build_model_from_official,
            is_official_checkpoint,
        )

        if is_official_checkpoint(ckpt):
            model, report = build_model_from_official(
                path,
                channels=channels,
                device=device,
                strict=False,
                allow_channel_mismatch=False,
            )
            if not report.loaded:
                raise RuntimeError(
                    "Official SleepFM checkpoint could not be mapped into "
                    "MultiModalSleepFM.\n" + report.summary()
                )
            # Surface channel montage in stdout so evaluate paths are not silent.
            for line in report.messages:
                if "CHANNEL" in line or line.startswith("WARNING"):
                    print(line)
            return model

        from sleepfm.models.channel_meta import (
            describe_montage,
            warn_official_vs_paper,
        )

        ch = channels or ckpt.get("channels")
        inferred = ckpt.get("inferred_channels")
        if isinstance(inferred, dict) and inferred:
            for line in warn_official_vs_paper(inferred, requested=channels or ch):
                print(line)
        state = ckpt.get("model_state_dict", ckpt)
        unify = ckpt.get("unify")
        if unify is None:
            unify = any(k.startswith("proj_shared") for k in state)
        shared_dim = ckpt.get("shared_dim")
        private_dim = ckpt.get("private_dim")
        if unify and shared_dim is None:
            w = state.get("proj_shared.bas.weight")
            if w is not None:
                shared_dim = int(w.shape[0])
        if unify and private_dim is None:
            w = state.get("proj_private.bas.weight")
            if w is not None:
                private_dim = int(w.shape[0])
        model = cls(
            channels=ch,
            embedding_dim=ckpt.get("embedding_dim", 512),
            unify=bool(unify),
            shared_dim=shared_dim,
            private_dim=private_dim,
            downstream_space=ckpt.get("downstream_space", "concat"),
        )
        model.load_state_dict(state, strict=True)
        if "temperature" in ckpt:
            model.temperature.data.fill_(ckpt["temperature"])
        if isinstance(ch, dict):
            print(f"Loaded checkpoint channels={describe_montage(ch)}")
        return model
