"""Retrieval Recall@k for contrastive embeddings (paper Sec 3.1 style).

Scripts use the full split as the paired gallery by default; ``limit_gallery``
caps N for large CinC/SHHS runs via **seeded random subsample** (not a prefix).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def recall_at_k(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Fraction of queries whose top-k gallery neighbors include the paired index.

    query, gallery: (N, D) L2-normalized; row i in query matches row i in gallery.
    """
    if query.size(0) < 2:
        return float("nan")
    sim = torch.matmul(query, gallery.T)
    k_eff = min(k, sim.size(1))
    topk = sim.topk(k_eff, dim=1).indices
    labels = torch.arange(query.size(0), device=query.device).unsqueeze(1)
    hits = (topk == labels).any(dim=1).float().mean().item()
    return float(hits)


def modality_retrieval_metrics(
    embeddings: Dict[str, torch.Tensor],
    k: int = 10,
) -> Dict[str, float]:
    """Pairwise retrieval: modality A queries modality B gallery."""
    names = sorted(embeddings.keys())
    out: Dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r_ab = recall_at_k(embeddings[a], embeddings[b], k=k)
            r_ba = recall_at_k(embeddings[b], embeddings[a], k=k)
            key = f"recall@{k}_{a}_to_{b}"
            out[key] = (r_ab + r_ba) / 2.0
    return out


def random_recall_baseline(n: int, k: int = 10) -> float:
    """Expected Recall@k if the true match ranks uniformly among n-1 negatives."""
    if n <= 1:
        return float("nan")
    return min(k, n - 1) / (n - 1)


def limit_gallery(
    embeddings: Dict[str, torch.Tensor],
    max_gallery: Optional[int] = None,
    seed: Optional[int] = 0,
    mode: str = "rng",
) -> Dict[str, torch.Tensor]:
    """Cap paired query/gallery size (full split by default).

    Parameters
    ----------
    max_gallery:
        If None or >= N, return embeddings unchanged.
    seed:
        RNG seed for ``mode="rng"`` (default 0). Ignored for ``mode="prefix"``.
    mode:
        ``"rng"`` — seeded ``randperm`` subsample (same indices across modalities).
        ``"prefix"`` — first ``max_gallery`` rows (legacy; biased if loader order is fixed).
    """
    if not embeddings or max_gallery is None:
        return embeddings
    n = next(iter(embeddings.values())).size(0)
    if n <= max_gallery:
        return embeddings
    if mode == "prefix":
        return {k: v[:max_gallery] for k, v in embeddings.items()}
    if mode != "rng":
        raise ValueError(f"limit_gallery mode must be 'rng' or 'prefix', got {mode!r}")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed if seed is not None else 0))
    idx = torch.randperm(n, generator=g)[:max_gallery]
    # Keep ascending order so paired rows stay aligned visually in dumps;
    # pairing is by shared index, not by original time order.
    idx, _ = torch.sort(idx)
    return {k: v[idx] for k, v in embeddings.items()}
