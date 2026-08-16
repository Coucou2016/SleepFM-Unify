"""Night-level AHI-bin / sleep-efficiency placeholders from index.json labels."""

import argparse
import json

import torch

from sleepfm.data.label_coverage import (
    apply_metric_gate,
    compute_label_coverage,
    gate_claimed_metrics,
)
from sleepfm.eval.night import epoch_sequence_kappa, night_eval_pack, probe_night_tasks
from sleepfm.models.channel_meta import assert_channels_compatible
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.models.temporal import NightTemporalEncoder
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM night-level evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-channel-mismatch",
        action="store_true",
        help="Override fail-fast when checkpoint channels != data meta (5/1/3 vs 10/2/7)",
    )
    parser.add_argument(
        "--force-metrics",
        action="store_true",
        help="Claim night staging/AHI even when label coverage gate would block them",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(args.checkpoint, device=str(device))
    assert_channels_compatible(
        model.channels,
        data_dir=data_dir,
        allow_mismatch=args.allow_channel_mismatch,
        context="evaluate_night",
    )
    model.to(device)
    temporal = NightTemporalEncoder.from_checkpoint(args.checkpoint, device=str(device))

    coverage = compute_label_coverage(data_dir)
    gate = gate_claimed_metrics(coverage)
    for msg in gate.messages:
        print(msg)

    tr = night_eval_pack(
        model,
        data_dir,
        args.train_split,
        device,
        batch_size=args.batch_size,
        temporal_encoder=temporal,
    )
    te = night_eval_pack(
        model,
        data_dir,
        args.test_split,
        device,
        batch_size=args.batch_size,
        temporal_encoder=temporal,
    )
    metrics = probe_night_tasks(tr["X"], tr["summaries"], te["X"], te["summaries"])
    metrics["temporal_head"] = "loaded" if temporal is not None else "mean_pool_fallback"
    if tr["y_stage"].size and te["y_stage"].size and (gate.claim_night_staging_kappa or args.force_metrics):
        metrics["staging_epoch_kappa"] = epoch_sequence_kappa(
            tr["X_epochs"], tr["y_stage"], te["X_epochs"], te["y_stage"]
        )
    metrics["train_nights"] = tr["keys"]
    metrics["test_nights"] = te["keys"]
    metrics["train_summaries"] = tr["summaries"]
    metrics["test_summaries"] = te["summaries"]
    if not args.force_metrics:
        metrics = apply_metric_gate(
            metrics,
            gate,
            staging_keys=("staging_epoch_kappa",),
            apnea_keys=(),
            night_ahi_keys=("ahi_bin_auroc", "ahi_bin", "ahi"),
        )
        if not gate.claim_night_ahi:
            for k in list(metrics.keys()):
                if "ahi" in k.lower() and k != "label_gate":
                    if isinstance(metrics[k], dict) and metrics[k].get("skipped"):
                        continue
                    metrics[k] = {
                        "skipped": True,
                        "reason": "insufficient respiratory/apnea label coverage",
                        "prior_value_removed": True,
                    }
    else:
        metrics["label_gate"] = {**gate.to_dict(), "forced": True}
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
