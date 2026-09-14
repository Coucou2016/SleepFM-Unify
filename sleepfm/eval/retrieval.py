"""Retrieval Recall@k for contrastive embeddings (paper Sec 3.1 style).

Scripts sample gallery indices from the full dataset **before** encoding when
``max_gallery`` is set. Similarity uses chunked matmul to avoid N×N OOM.

Pair metrics only use samples where **both** modalities are present
(co-presence filtering); random baseline is ``k / N_pair``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def recall_at_k(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k: int = 10,
    chunk_size: int = 512,
) -> float:
    """
    Fraction of queries whose top-k gallery neighbors include the paired index.

    query, gallery: (N, D) L2-normalized; row i in query matches row i in gallery.
    Uses chunked similarity to avoid materializing a full N×N matrix.
    """
    n = int(query.size(0))
    if n < 2:
        return float("nan")
    k_eff = min(k, gallery.size(0))
    device = query.device
    hits = torch.zeros(n, device=device)
    labels = torch.arange(n, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = torch.matmul(query[start:end], gallery.T)
        topk = sim.topk(k_eff, dim=1).indices
        lab = labels[start:end].unsqueeze(1)
        hits[start:end] = (topk == lab).any(dim=1).float()
    return float(hits.mean().item())


def random_recall_baseline(n: int, k: int = 10) -> float:
    """Expected Recall@k if the true match ranks uniformly among n gallery items."""
    if n <= 0:
        return float("nan")
    return min(k, n) / float(n)


def _pair_both_present(
    present_mask: torch.Tensor,
    modality_order: Sequence[str],
    a: str,
    b: str,
) -> torch.Tensor:
    ia = list(modality_order).index(a)
    ib = list(modality_order).index(b)
    return (present_mask[:, ia] > 0.5) & (present_mask[:, ib] > 0.5)


def modality_retrieval_metrics(
    embeddings: Dict[str, torch.Tensor],
    k: int = 10,
    chunk_size: int = 512,
    present_mask: Optional[torch.Tensor] = None,
    modality_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Pairwise retrieval with co-presence filtering and directed pair scores.

    For each unordered pair {A, B}, only rows where both A and B are present
    enter the gallery/query. Reports:
      - ``recall@k_{a}_to_{b}`` / ``recall@k_{b}_to_{a}`` (directed)
      - ``recall@k_{a}_{b}_avg`` (macro of the two directions)
      - ``n_pair_{a}_{b}`` and ``random_baseline_{a}_{b}`` (= k / N_pair)
    Without ``present_mask``, falls back to using all rows (legacy).
    """
    names = sorted(embeddings.keys())
    order = list(modality_order) if modality_order is not None else names
    out: Dict[str, float] = {}
    directed = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ea, eb = embeddings[a], embeddings[b]
            if present_mask is not None:
                both = _pair_both_present(present_mask, order, a, b)
                n_pair = int(both.sum())
                if n_pair < 2:
                    out[f"recall@{k}_{a}_to_{b}"] = float("nan")
                    out[f"recall@{k}_{b}_to_{a}"] = float("nan")
                    out[f"recall@{k}_{a}_{b}_avg"] = float("nan")
                    out[f"n_pair_{a}_{b}"] = float(n_pair)
                    out[f"random_baseline_{a}_{b}"] = float("nan")
                    continue
                ea = ea[both]
                eb = eb[both]
            else:
                n_pair = int(ea.size(0))
            r_ab = recall_at_k(ea, eb, k=k, chunk_size=chunk_size)
            r_ba = recall_at_k(eb, ea, k=k, chunk_size=chunk_size)
            out[f"recall@{k}_{a}_to_{b}"] = r_ab
            out[f"recall@{k}_{b}_to_{a}"] = r_ba
            avg = (r_ab + r_ba) / 2.0
            out[f"recall@{k}_{a}_{b}_avg"] = avg
            # Legacy bidirectional key (same as avg) for older scripts.
            out[f"recall@{k}_{a}_to_{b}_bidir"] = avg
            out[f"n_pair_{a}_{b}"] = float(n_pair)
            out[f"random_baseline_{a}_{b}"] = random_recall_baseline(n_pair, k=k)
            if not (r_ab != r_ab):  # not NaN
                directed.extend([r_ab, r_ba])
    if directed:
        out[f"recall@{k}_macro_avg"] = float(sum(directed) / len(directed))
    return out


def sample_gallery_indices(
    n: int,
    max_gallery: Optional[int],
    seed: Optional[int] = 0,
    mode: str = "rng",
) -> Optional[np.ndarray]:
    """Pick subset indices from the full dataset *before* loading/encoding."""
    if max_gallery is None or n <= max_gallery:
        return None
    if mode == "prefix":
        return np.arange(max_gallery, dtype=np.int64)
    if mode != "rng":
        raise ValueError(f"gallery mode must be 'rng' or 'prefix', got {mode!r}")
    rng = np.random.default_rng(int(seed if seed is not None else 0))
    idx = rng.choice(n, size=max_gallery, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def subset_dataset(dataset: Dataset, indices: Optional[Sequence[int]]) -> Dataset:
    if indices is None:
        return dataset
    return Subset(dataset, list(indices))


def limit_gallery(
    embeddings: Dict[str, torch.Tensor],
    max_gallery: Optional[int] = None,
    seed: Optional[int] = 0,
    mode: str = "rng",
    present_mask: Optional[torch.Tensor] = None,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    """Cap paired query/gallery size after encoding (legacy helper).

    Prefer ``sample_gallery_indices`` + ``Subset`` *before* encode for large splits.
    When ``present_mask`` is passed, returns ``(embeddings, present_mask)`` sliced together.
    """
    if not embeddings or max_gallery is None:
        if present_mask is not None:
            return embeddings, present_mask
        return embeddings
    n = next(iter(embeddings.values())).size(0)
    if n <= max_gallery:
        if present_mask is not None:
            return embeddings, present_mask
        return embeddings
    if mode == "prefix":
        idx = torch.arange(max_gallery)
    elif mode == "rng":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed if seed is not None else 0))
        idx = torch.randperm(n, generator=g)[:max_gallery]
        idx, _ = torch.sort(idx)
    else:
        raise ValueError(f"limit_gallery mode must be 'rng' or 'prefix', got {mode!r}")
    out = {k: v[idx] for k, v in embeddings.items()}
    if present_mask is not None:
        return out, present_mask[idx]
    return out


def encode_retrieval_embeddings(
    model,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    collate_fn=None,
    max_gallery: Optional[int] = None,
    gallery_seed: int = 0,
    gallery_mode: str = "rng",
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Encode a split for retrieval; also return stacked ``present_mask``.

    Returns ``(embeddings, present_mask)`` where ``present_mask`` is ``(N, M)``
    or ``None`` if the dataset never emitted masks.
    """
    from sleepfm.data.dataset import collate_multimodal

    if collate_fn is None:
        collate_fn = collate_multimodal
    indices = sample_gallery_indices(
        len(dataset), max_gallery, seed=gallery_seed, mode=gallery_mode
    )
    ds = subset_dataset(dataset, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_emb = {m: [] for m in model.MODALITY_ORDER}
    mask_chunks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (
                    {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else (v.to(device) if torch.is_tensor(v) else v)
                )
                for k, v in batch.items()
            }
            space = "shared" if getattr(model, "unify", False) else "downstream"
            z = model.encode(batch, space=space)
            mask = batch.get("present_mask")
            if mask is not None:
                mask_chunks.append(mask.detach().cpu())
            for m, t in z.items():
                if mask is not None:
                    col = model.MODALITY_ORDER.index(m)
                    t = t * mask[:, col : col + 1].to(dtype=t.dtype)
                all_emb[m].append(t.cpu())
    embeddings = {m: torch.cat(chunks, dim=0) for m, chunks in all_emb.items() if chunks}
    present = torch.cat(mask_chunks, dim=0) if mask_chunks else None
    return embeddings, present
