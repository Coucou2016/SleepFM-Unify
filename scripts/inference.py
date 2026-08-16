"""Extract embeddings from a checkpoint for new epochs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Extract SleepFM embeddings")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    model.to(device)
    model.eval()

    ds = SleepEpochDataset(args.data_dir, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_multimodal)

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            z = model.encode(batch)
            combined = torch.cat([z[m] for m in model.MODALITY_ORDER if m in z], dim=-1)
            chunks.append(combined.cpu().numpy())

    out = np.concatenate(chunks, axis=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved embeddings shape {out.shape} to {args.output}")


if __name__ == "__main__":
    main()
