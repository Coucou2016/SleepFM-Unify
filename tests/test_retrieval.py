"""Retrieval metric unit tests."""

import torch

from sleepfm.eval.retrieval import (
    encode_retrieval_embeddings,
    limit_gallery,
    modality_retrieval_metrics,
    random_recall_baseline,
    recall_at_k,
    sample_gallery_indices,
)
from sleepfm.models.sleepfm import MultiModalSleepFM


def test_perfect_retrieval():
    n = 8
    z = torch.zeros(n, 16)
    z[torch.arange(n), torch.arange(n)] = 1.0
    r = recall_at_k(z, z, k=3)
    assert r == 1.0


def test_random_baseline_formula():
    # True match is one of n gallery items (uniform): min(k,n)/n
    assert abs(random_recall_baseline(100, k=10) - 10 / 100) < 1e-6
    assert abs(random_recall_baseline(10, k=10) - 1.0) < 1e-6


def test_chunked_recall_matches_full():
    torch.manual_seed(0)
    n, d = 64, 8
    q = torch.randn(n, d)
    q = q / q.norm(dim=-1, keepdim=True)
    g = q.clone()
    assert abs(recall_at_k(q, g, k=5, chunk_size=7) - recall_at_k(q, g, k=5, chunk_size=512)) < 1e-6


def test_sample_gallery_indices_before_encode():
    idx = sample_gallery_indices(100, max_gallery=10, seed=3, mode="rng")
    assert idx is not None and len(idx) == 10
    assert idx.min() >= 0 and idx.max() < 100
    assert list(idx) == sorted(idx.tolist())
    # Different from prefix
    prefix = sample_gallery_indices(100, max_gallery=10, seed=3, mode="prefix")
    assert not (idx == prefix).all()
    assert sample_gallery_indices(5, max_gallery=10) is None


def test_limit_gallery():
    z = torch.randn(20, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    embs = {"bas": z, "ecg": z}
    capped = limit_gallery(embs, max_gallery=5, seed=0, mode="rng")
    assert capped["bas"].size(0) == 5
    assert limit_gallery(embs, max_gallery=None)["bas"].size(0) == 20
    assert limit_gallery(embs, max_gallery=50)["bas"].size(0) == 20
    r = recall_at_k(capped["bas"], capped["ecg"], k=1)
    assert 0.0 <= r <= 1.0


def test_limit_gallery_rng_reproducible_and_not_prefix():
    z = torch.randn(40, 8)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    embs = {"bas": z, "ecg": z.clone()}
    a = limit_gallery(embs, max_gallery=8, seed=7, mode="rng")
    b = limit_gallery(embs, max_gallery=8, seed=7, mode="rng")
    c = limit_gallery(embs, max_gallery=8, seed=8, mode="rng")
    prefix = limit_gallery(embs, max_gallery=8, mode="prefix")
    assert torch.equal(a["bas"], b["bas"])
    assert not torch.equal(a["bas"], c["bas"])
    assert torch.equal(a["bas"], a["ecg"])
    assert not torch.equal(a["bas"], prefix["bas"])


def test_limit_gallery_invalid_mode():
    z = torch.randn(10, 4)
    try:
        limit_gallery({"bas": z}, max_gallery=3, mode="bogus")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "rng" in str(exc)


def test_copresence_filters_missing_modality_from_pair():
    """A sample missing B must never enter A↔B gallery/query."""
    n, d = 6, 4
    order = ["bas", "ecg", "respiratory"]
    # Distinct one-hot-ish embeddings so recall is well-defined on the subset.
    bas = torch.zeros(n, d)
    ecg = torch.zeros(n, d)
    for i in range(n):
        bas[i, i % d] = 1.0
        ecg[i, (i + 1) % d] = 1.0
    bas = bas / bas.norm(dim=-1, keepdim=True)
    ecg = ecg / ecg.norm(dim=-1, keepdim=True)
    # Sample 0: ECG missing; sample 1: BAS missing; rest both present.
    present = torch.ones(n, 3)
    present[0, 1] = 0.0  # ecg absent
    present[1, 0] = 0.0  # bas absent
    metrics = modality_retrieval_metrics(
        {"bas": bas, "ecg": ecg},
        k=1,
        present_mask=present,
        modality_order=order,
    )
    assert metrics["n_pair_bas_ecg"] == 4.0  # rows 2..5
    assert abs(metrics["random_baseline_bas_ecg"] - 1.0 / 4.0) < 1e-6
    assert "recall@1_bas_to_ecg" in metrics
    assert "recall@1_ecg_to_bas" in metrics
    assert "recall@1_bas_ecg_avg" in metrics
    # If we wrongly included missing rows, N would be 6.
    assert metrics["n_pair_bas_ecg"] != 6.0


def test_encode_retrieval_returns_present_mask(tiny_data_dir):
    model = MultiModalSleepFM(
        channels={"bas": 10, "ecg": 2, "respiratory": 7}, embedding_dim=16
    )
    from sleepfm.data.dataset import SleepEpochDataset

    ds = SleepEpochDataset(tiny_data_dir, split="pretrain")
    emb, mask = encode_retrieval_embeddings(
        model, ds, torch.device("cpu"), batch_size=4, max_gallery=8
    )
    assert mask is not None
    assert mask.ndim == 2 and mask.size(0) == next(iter(emb.values())).size(0)
    metrics = modality_retrieval_metrics(
        emb, k=1, present_mask=mask, modality_order=model.MODALITY_ORDER
    )
    assert any(k.startswith("n_pair_") for k in metrics)
