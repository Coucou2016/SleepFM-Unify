"""Retrieval metric unit tests."""

import torch

from sleepfm.eval.retrieval import limit_gallery, random_recall_baseline, recall_at_k


def test_perfect_retrieval():
    n = 8
    z = torch.zeros(n, 16)
    z[torch.arange(n), torch.arange(n)] = 1.0
    r = recall_at_k(z, z, k=3)
    assert r == 1.0


def test_random_baseline_formula():
    assert abs(random_recall_baseline(100, k=10) - 10 / 99) < 1e-6


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
    # Distinct rows so different index sets are detectable after gather.
    z = torch.randn(40, 8)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    embs = {"bas": z, "ecg": z.clone()}
    a = limit_gallery(embs, max_gallery=8, seed=7, mode="rng")
    b = limit_gallery(embs, max_gallery=8, seed=7, mode="rng")
    c = limit_gallery(embs, max_gallery=8, seed=8, mode="rng")
    prefix = limit_gallery(embs, max_gallery=8, mode="prefix")
    assert torch.equal(a["bas"], b["bas"])
    assert not torch.equal(a["bas"], c["bas"])
    # Same indices across modalities
    assert torch.equal(a["bas"], a["ecg"])
    # RNG default should differ from prefix for this sized tensor
    assert not torch.equal(a["bas"], prefix["bas"])


def test_limit_gallery_invalid_mode():
    z = torch.randn(10, 4)
    try:
        limit_gallery({"bas": z}, max_gallery=3, mode="bogus")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "rng" in str(exc)
