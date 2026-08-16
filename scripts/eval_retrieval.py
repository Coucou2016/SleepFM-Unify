"""Evaluate retrieval Recall@k on a split (full-split gallery by default)."""

import argparse

import torch
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.eval.retrieval import limit_gallery, modality_retrieval_metrics, random_recall_baseline
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM retrieval Recall@k")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=None,
        help="Cap paired gallery/query size (default: all embeddings in the split). "
        "Uses seeded RNG subsample by default (not a prefix).",
    )
    parser.add_argument(
        "--gallery-seed",
        type=int,
        default=None,
        help="Seed for gallery RNG subsample (default: config seed).",
    )
    parser.add_argument(
        "--gallery-mode",
        type=str,
        choices=["rng", "prefix"],
        default="rng",
        help="How to cap gallery: rng (default) or prefix (legacy).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    gallery_seed = args.gallery_seed if args.gallery_seed is not None else int(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)
    model.eval()

    ds = SleepEpochDataset(args.data_dir or cfg["data_dir"], split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_multimodal)

    all_emb = {m: [] for m in model.MODALITY_ORDER}
    n_got = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            space = "shared" if getattr(model, "unify", False) else "downstream"
            z = model.encode(batch, space=space)
            for m, t in z.items():
                all_emb[m].append(t.cpu())
            if z:
                n_got += next(iter(z.values())).size(0)
            # Collect beyond the cap so RNG subsample is not a prefix of loader order.
            if args.max_gallery is not None and n_got >= max(args.max_gallery * 4, args.max_gallery):
                break
    embeddings = {m: torch.cat(chunks, dim=0) for m, chunks in all_emb.items() if chunks}
    embeddings = limit_gallery(
        embeddings,
        max_gallery=args.max_gallery,
        seed=gallery_seed,
        mode=args.gallery_mode,
    )

    metrics = modality_retrieval_metrics(embeddings, k=args.k)
    n = next(iter(embeddings.values())).size(0)
    baseline = random_recall_baseline(n, k=args.k)
    print(f"Split={args.split} N={n} random Recall@{args.k}≈{baseline:.4f}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
