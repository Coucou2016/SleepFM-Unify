"""Downstream sleep staging and apnea detection via logistic regression on embeddings."""

import argparse
import json

import torch

from sleepfm.data.dataset import SleepEpochDataset
from sleepfm.data.label_coverage import (
    apply_metric_gate,
    compute_label_coverage,
    gate_claimed_metrics,
)
from sleepfm.data.splits import downstream_isolation_ok
from sleepfm.eval.downstream import (
    build_embedding_matrix,
    can_train_binary,
    evaluate_apnea,
    evaluate_sleep_staging,
    train_logistic_regression,
    tune_lr_c,
)
from sleepfm.models.channel_meta import assert_channels_compatible
from sleepfm.models.sleepfm import MultiModalSleepFM
from sleepfm.utils.config import load_config
from sleepfm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="SleepFM downstream evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-channel-mismatch",
        action="store_true",
        help="Override fail-fast when checkpoint channels != data meta (5/1/3 vs 10/2/7)",
    )
    parser.add_argument(
        "--force-metrics",
        action="store_true",
        help="Claim staging/apnea even when label coverage gate would block them",
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
        context="evaluate_downstream",
    )
    model.to(device)

    ds_cfg = cfg["downstream"]
    train_split = ds_cfg.get("train_split", "train")
    test_split = ds_cfg.get("test_split", "test")
    valid_split = ds_cfg.get("valid_split", "valid")

    iso = downstream_isolation_ok(data_dir)
    failed = [k for k, v in iso.items() if not v]
    if failed:
        print(f"WARNING: split isolation checks failed: {failed}")

    coverage = compute_label_coverage(data_dir)
    gate = gate_claimed_metrics(coverage)
    print("Label coverage:", json.dumps(coverage.to_dict(), indent=2))
    for msg in gate.messages:
        print(msg)

    train_ds = SleepEpochDataset(data_dir, split=train_split, return_labels=True)
    test_ds = SleepEpochDataset(data_dir, split=test_split, return_labels=True)

    X_train, y_train = build_embedding_matrix(model, train_ds, device, args.batch_size)
    X_test, y_test = build_embedding_matrix(model, test_ds, device, args.batch_size)

    lr_cfg = dict(ds_cfg)
    if ds_cfg.get("tune_c_on_valid") and (gate.claim_staging or args.force_metrics):
        try:
            valid_ds = SleepEpochDataset(data_dir, split=valid_split, return_labels=True)
            X_val, y_val = build_embedding_matrix(model, valid_ds, device, args.batch_size)
            best_c = tune_lr_c(X_val, y_val["stage_id"], {**ds_cfg, "task": "staging"})
            lr_cfg["C"] = best_c
            print(f"Tuned L2 C on {valid_split}: {best_c}")
        except KeyError:
            print(f"Skipping C tuning: split '{valid_split}' not in index.json")

    out = {}
    if gate.claim_staging or args.force_metrics:
        clf_stage = train_logistic_regression(X_train, y_train["stage_id"], lr_cfg)
        out["staging"] = evaluate_sleep_staging(clf_stage, X_test, y_test["stage_id"])
        print("Sleep staging:", out["staging"])
    else:
        out["staging"] = {
            "skipped": True,
            "reason": "insufficient AASM staging label coverage",
        }
        print("Sleep staging: SKIPPED (label gate)")

    if (gate.claim_apnea or args.force_metrics) and can_train_binary(y_train["apnea"]):
        apnea_cfg = lr_cfg
        if ds_cfg.get("tune_c_on_valid"):
            try:
                valid_ds = SleepEpochDataset(data_dir, split=valid_split, return_labels=True)
                X_val, y_val = build_embedding_matrix(model, valid_ds, device, args.batch_size)
                if can_train_binary(y_val["apnea"]):
                    apnea_cfg = {
                        **lr_cfg,
                        "C": tune_lr_c(X_val, y_val["apnea"], {**ds_cfg, "task": "apnea"}),
                    }
            except KeyError:
                pass
        clf_apnea = train_logistic_regression(X_train, y_train["apnea"], apnea_cfg)
        out["apnea"] = evaluate_apnea(clf_apnea, X_test, y_test["apnea"])
        print("Apnea (SDB):", out["apnea"])
    elif not gate.claim_apnea and not args.force_metrics:
        out["apnea"] = {
            "skipped": True,
            "reason": "insufficient respiratory/apnea label coverage",
        }
        print("Apnea (SDB): SKIPPED (label gate — CinC often lacks respiratory events)")
    else:
        out["apnea"] = {
            "skipped": True,
            "reason": "need >=2 classes in train split; increase demo size or apnea_rate",
        }
        print("Apnea (SDB): skipped (need >=2 classes in train split; increase demo size or apnea_rate)")

    if not args.force_metrics:
        out = apply_metric_gate(out, gate)
    else:
        out["label_gate"] = {**gate.to_dict(), "forced": True}
    print("label_gate:", json.dumps(out.get("label_gate", {}), indent=2))


if __name__ == "__main__":
    main()
