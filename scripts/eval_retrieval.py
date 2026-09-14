"""Evaluate retrieval Recall@k on a split (full-split gallery by default)."""

import argparse

import torch

from sleepfm.data.dataset import SleepEpochDataset
from sleepfm.eval.retrieval import (
    encode_retrieval_embeddings,
    modality_retrieval_metrics,
    random_recall_baseline,
)
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
        "Samples indices from the full dataset before encoding (seeded RNG by default).",
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
    embeddings, present_mask = encode_retrieval_embeddings(
        model,
        ds,
        device,
        batch_size=args.batch_size,
        max_gallery=args.max_gallery,
        gallery_seed=gallery_seed,
        gallery_mode=args.gallery_mode,
    )

    metrics = modality_retrieval_metrics(
        embeddings,
        k=args.k,
        present_mask=present_mask,
        modality_order=model.MODALITY_ORDER,
    )
    # Prefer per-pair baselines when co-presence filtering is active.
    pair_baselines = {k: v for k, v in metrics.items() if k.startswith("random_baseline_")}
    if pair_baselines:
        print(f"Split={args.split} co-presence filtered (per-pair N / baseline):")
        for key, base in sorted(pair_baselines.items()):
            pair = key.replace("random_baseline_", "")
            n_key = f"n_pair_{pair}"
            n_pair = int(metrics.get(n_key, 0))
            print(f"  {pair}: N_pair={n_pair} random Recall@{args.k}≈{base:.4f}")
    else:
        n = next(iter(embeddings.values())).size(0)
        baseline = random_recall_baseline(n, k=args.k)
        print(f"Split={args.split} N={n} random Recall@{args.k}≈{baseline:.4f}")
    for k, v in sorted(metrics.items()):
        if k.startswith("random_baseline_") or k.startswith("n_pair_"):
            continue
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
